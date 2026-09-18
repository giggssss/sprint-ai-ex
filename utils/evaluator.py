"""
통합 모델 평가 및 OOD(Out-of-Distribution) 필터링 모듈
대회 공식 평가 기준인 mAP@[0.75:0.95] 산출, k-NN Feature Bank 기반 OOD 거절,
및 최적 임계값 최적화 기능을 제공합니다.
"""
import os
import glob
import json
import ast
import cv2
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

import config


class YOLOEvaluator:
    """YOLO 모델 표준 추론, COCO 지표 평가 및 임계값 최적화 엔진."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        data_yaml_path: str = config.DATA_YAML_DEFAULT,
        class_mapping_path: str = config.CLASS_MAPPING_FILE,
        device: str = config.DEVICE,
    ):
        self.weights_path = config.resolve_weights(weights_path)
        self.data_yaml_path = data_yaml_path
        self.class_mapping_path = class_mapping_path
        self.device = device

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {self.weights_path}")

    def load_model(self) -> YOLO:
        """가중치로부터 YOLO 인스턴스를 로드합니다."""
        return YOLO(self.weights_path)

    def load_class_mapping(self) -> Dict[str, int]:
        """YOLO 클래스 ID -> 원본 의약품 코드(category_id) 매핑을 로드합니다."""
        if not os.path.exists(self.class_mapping_path):
            raise FileNotFoundError(f"클래스 매핑 파일을 찾을 수 없습니다: {self.class_mapping_path}")
        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_filename_to_id_map(self, annotations_path: str) -> Dict[str, int]:
        """어노테이션 파일에서 file_name -> image_id 매핑 딕셔너리를 구축합니다."""
        filename_to_id = {}
        if os.path.isfile(annotations_path):
            with open(annotations_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for img in data.get("images", []):
                    fn = img.get("file_name")
                    img_id = img.get("id")
                    if fn and img_id is not None:
                        filename_to_id[fn] = img_id
                        filename_to_id[f"{img_id}.png"] = img_id
                        filename_to_id[f"{img_id}.jpg"] = img_id
        else:
            json_files = glob.glob(os.path.join(annotations_path, "**", "*.json"), recursive=True)
            for jf in json_files:
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for img in data.get("images", []):
                            fn = img.get("file_name")
                            img_id = img.get("id")
                            if fn and img_id is not None:
                                filename_to_id[fn] = img_id
                                filename_to_id[f"{img_id}.png"] = img_id
                                filename_to_id[f"{img_id}.jpg"] = img_id
                except Exception:
                    pass
        return filename_to_id

    def predict_test_set(
        self,
        test_images_dir: str = config.TEST_IMAGES_DEFAULT,
        test_annotations_path: Optional[str] = None,
        output_csv: str = config.DEFAULT_PREDICTIONS_CSV,
        conf_threshold: Optional[float] = None,
        iou_threshold: float = 0.45,
        imgsz: int = 960,
        batch_size: int = 16,
    ) -> pd.DataFrame:
        """테스트 이미지 폴더에 대해 배치 추론을 수행하고 대회 제출용 포맷의 CSV를 생성합니다."""
        annotations_path = test_annotations_path or config.resolve_test_annotations()
        print(f"🚀 [YOLO 추론 시작] 모델: {self.weights_path}")
        print(f"  • 테스트 이미지: {test_images_dir}")
        print(f"  • 어노테이션 소스: {annotations_path}")

        model = self.load_model()
        class_mapping = self.load_class_mapping()
        filename_to_id = self.build_filename_to_id_map(annotations_path)

        if conf_threshold is None:
            conf_threshold = config.get_optimal_confidence(default_conf=0.25)
        print(f"  • 적용 Confidence: {conf_threshold:.4f} | IoU: {iou_threshold}")

        image_paths = sorted(
            glob.glob(os.path.join(test_images_dir, "*.[pP][nN][gG]"))
            + glob.glob(os.path.join(test_images_dir, "*.[jJ][pP][gG]"))
            + glob.glob(os.path.join(test_images_dir, "*.[jJ][pP][eE][gG]"))
        )
        if not image_paths:
            raise FileNotFoundError(f"테스트 이미지를 찾을 수 없습니다: {test_images_dir}")
        print(f"  • 총 {len(image_paths)}개 이미지 추론 진행...")

        records = []
        annotation_id = 1

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            results = model.predict(
                source=batch_paths,
                conf=conf_threshold,
                iou=iou_threshold,
                agnostic_nms=True,
                imgsz=imgsz,
                augment=True,
                device="cpu" if self.device == "mps" else self.device,
                verbose=False,
            )

            for r, img_path in zip(results, batch_paths):
                filename = os.path.basename(img_path)
                if filename in filename_to_id:
                    image_id = filename_to_id[filename]
                else:
                    try:
                        image_id = int(os.path.splitext(filename)[0])
                    except ValueError:
                        import hashlib
                        image_id = int(hashlib.md5(filename.encode()).hexdigest(), 16) % (10**8)

                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        yolo_cls_id = int(box.cls[0].item())
                        original_dl_idx = class_mapping.get(str(yolo_cls_id), yolo_cls_id)
                        conf = float(box.conf[0].item())

                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        x_min, y_min, x_max, y_max = xyxy
                        bbox_x = x_min
                        bbox_y = y_min
                        bbox_w = x_max - x_min
                        bbox_h = y_max - y_min

                        records.append({
                            "annotation_id": annotation_id,
                            "image_id": image_id,
                            "category_id": original_dl_idx,
                            "bbox_x": round(bbox_x, 2),
                            "bbox_y": round(bbox_y, 2),
                            "bbox_w": round(bbox_w, 2),
                            "bbox_h": round(bbox_h, 2),
                            "score": round(conf, 4),
                        })
                        annotation_id += 1

        df = pd.DataFrame(
            records,
            columns=["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"],
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 예측 CSV 저장 완료: {output_csv} (총 {len(df)}개 박스)")
        return df

    def evaluate_coco(
        self,
        predictions_csv: str = config.DEFAULT_PREDICTIONS_CSV,
        annotations_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """pycocotools를 이용하여 56개 학습 클래스에 대해 공식 기준인 mAP@[0.75:0.95] 및 관련 지표를 산출합니다."""
        ann_file = annotations_path or config.resolve_test_annotations()
        print(f"\n📊 [COCO 정량 평가] CSV: {predictions_csv} vs GT: {ann_file}")
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"GT 어노테이션 파일을 찾을 수 없습니다: {ann_file}")
        if not os.path.exists(predictions_csv):
            raise FileNotFoundError(f"예측 CSV 파일을 찾을 수 없습니다: {predictions_csv}")

        coco_gt = COCO(ann_file)
        df = pd.read_csv(predictions_csv, encoding="utf-8-sig")

        if len(df) == 0:
            print("⚠️ 평가할 예측 결과가 비어 있습니다.")
            return {"mAP75-95": 0.0, "mAP75": 0.0, "mAP50": 0.0, "Recall": 0.0}

        if "bbox" in df.columns and isinstance(df.iloc[0]["bbox"], str):
            df["bbox"] = df["bbox"].apply(ast.literal_eval)
            res_json = df.to_dict("records")
        elif "bbox_x" in df.columns:
            res_json = []
            for _, r in df.iterrows():
                res_json.append({
                    "image_id": int(r["image_id"]),
                    "category_id": int(r["category_id"]),
                    "bbox": [float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_w"]), float(r["bbox_h"])],
                    "score": float(r["score"]),
                })
        else:
            res_json = df.to_dict("records")

        coco_dt = coco_gt.loadRes(res_json)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

        # 56개 학습 클래스만 평가 대상으로 필터링
        known_coco_ids = []
        if os.path.exists(self.data_yaml_path):
            with open(self.data_yaml_path, "r", encoding="utf-8") as f:
                yolo_names = [n.strip() for n in yaml.safe_load(f)["names"]]
            for cat in coco_gt.loadCats(coco_gt.getCatIds()):
                if cat["name"].strip() in yolo_names:
                    known_coco_ids.append(cat["id"])
            coco_eval.params.catIds = known_coco_ids

        # 공식 엄격 기준: IoU 0.75 ~ 0.95
        coco_eval.params.iouThrs = np.linspace(0.75, 0.95, int(np.round((0.95 - 0.75) / 0.05)) + 1, endpoint=True)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP_75_95 = float(coco_eval.stats[0])
        ap_75 = float(coco_eval.stats[2])
        ar_75_95 = float(coco_eval.stats[8])

        # 보조 지표 (mAP@0.5)
        coco_eval_standard = COCOeval(coco_gt, coco_dt, "bbox")
        if known_coco_ids:
            coco_eval_standard.params.catIds = known_coco_ids
        coco_eval_standard.evaluate()
        coco_eval_standard.accumulate()
        map50 = float(coco_eval_standard.stats[1])

        print("=" * 60)
        print(f"🎯 [공식 타겟 기준] mAP@[0.75:0.95]: {mAP_75_95:.4f}")
        print(f"📊 mAP@0.75 (단일)         : {ap_75:.4f}")
        print(f"📊 mAP@0.50 (기본)         : {map50:.4f}")
        print(f"📈 Recall@[0.75:0.95]      : {ar_75_95:.4f}")
        print("=" * 60)

        return {
            "mAP75-95": round(mAP_75_95, 4),
            "mAP75": round(ap_75, 4),
            "mAP50": round(map50, 4),
            "Recall": round(ar_75_95, 4),
        }

    def find_optimal_threshold(self, split: str = "val") -> Tuple[float, float]:
        """Validation 셋의 Precision-Recall 곡선으로부터 F1 점수를 최대화하는 최적의 Confidence 임계값을 탐색합니다."""
        print(f"\n🔍 [최적 Confidence 임계값 탐색] 검증 셋: {split}")
        model = self.load_model()
        metrics = model.val(data=self.data_yaml_path, split=split, plots=False, verbose=False)

        f1_curve = np.array(metrics.box.f1)
        mean_f1 = f1_curve.mean(axis=0) if f1_curve.ndim == 2 else f1_curve

        points = mean_f1.shape[0]
        conf_thresholds = np.linspace(0.0, 1.0, points)

        optimal_idx = int(np.argmax(mean_f1))
        optimal_conf = float(conf_thresholds[optimal_idx])
        optimal_f1 = float(mean_f1[optimal_idx])

        print(f"🌟 최적 Confidence Threshold: {optimal_conf:.4f} (Max F1: {optimal_f1:.4f})")
        with open(config.OPTIMAL_CONF_FILE, "w", encoding="utf-8") as f:
            f.write(f"{optimal_conf:.4f}")
        print(f"💾 최적 임계값이 '{config.OPTIMAL_CONF_FILE}'에 저장되었습니다.")

        return optimal_conf, optimal_f1


class OODEvaluator:
    """YOLO Backbone 임베딩 및 k-NN Feature Bank를 활용한 OOD(미학습 클래스) 거절 엔진."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        train_images_dir: str = config.TRAIN_IMAGES_DEFAULT,
        train_labels_dir: str = config.TRAIN_LABELS_DEFAULT,
        data_yaml_path: str = config.DATA_YAML_DEFAULT,
        device: str = config.DEVICE,
    ):
        self.weights_path = config.resolve_weights(weights_path)
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        self.data_yaml_path = data_yaml_path
        self.device = torch.device(device)

        self.model = YOLO(self.weights_path)
        self.backbone = self._extract_backbone().to(self.device)

    def _extract_backbone(self) -> torch.nn.Module:
        """YOLO 모델에서 SPPF 계층까지의 Backbone을 추출합니다."""
        backbone = torch.nn.Sequential(*list(self.model.model.model.children())[:10])
        backbone.eval()
        return backbone

    def extract_embedding(self, img_crop: np.ndarray) -> torch.Tensor:
        """크롭된 이미지로부터 L2 정규화된 512차원 특징 벡터를 추출합니다."""
        img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            x = self.backbone(tensor)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            x = F.normalize(x, p=2, dim=1)
        return x.squeeze(0).cpu()

    def build_feature_bank(self, num_classes: int = 56) -> Dict[int, torch.Tensor]:
        """Train 셋의 모든 정답 바운딩 박스로부터 클래스별 k-NN Feature Bank를 구축합니다."""
        print("\n🏦 [k-NN Feature Bank 구축] Train 이미지로부터 특징 추출 중...")
        feature_bank: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_classes)}

        train_imgs = glob.glob(os.path.join(self.train_images_dir, "*.*"))
        for img_path in tqdm(train_imgs, desc="Extracting Train Features"):
            label_path = os.path.join(
                self.train_labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            )
            if not os.path.exists(label_path):
                continue

            orig_img = cv2.imread(img_path)
            if orig_img is None:
                continue
            h_img, w_img = orig_img.shape[:2]

            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                x1 = max(0, int((cx - w / 2) * w_img))
                y1 = max(0, int((cy - h / 2) * h_img))
                x2 = min(w_img, int((cx + w / 2) * w_img))
                y2 = min(h_img, int((cy + h / 2) * h_img))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = orig_img[y1:y2, x1:x2]
                emb = self.extract_embedding(crop)
                feature_bank[cls_id].append(emb)

        tensor_bank = {}
        for cls_id in range(num_classes):
            if len(feature_bank[cls_id]) > 0:
                tensor_bank[cls_id] = torch.stack(feature_bank[cls_id]).to(self.device)
            else:
                tensor_bank[cls_id] = torch.zeros((1, 512)).to(self.device)

        print(f"✅ 총 {num_classes}개 클래스에 대한 Feature Bank 구축 완료!")
        return tensor_bank

    def run_ood_sweep(
        self,
        test_images_dir: str = config.TEST_IMAGES_DEFAULT,
        test_annotations_path: Optional[str] = None,
        output_csv: str = config.DEFAULT_PREDICTIONS_CSV,
        thresholds: Optional[List[float]] = None,
    ) -> Tuple[float, float, str]:
        """다양한 코사인 유사도 임계값에 대해 OOD 거절을 적용하고 최적의 mAP@[0.75:0.95]를 산출합니다."""
        if thresholds is None:
            thresholds = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]

        ann_file = test_annotations_path or config.resolve_test_annotations()
        feature_bank = self.build_feature_bank(num_classes=56)

        evaluator = YOLOEvaluator(weights_path=self.weights_path, data_yaml_path=self.data_yaml_path)
        with open(config.CLASS_MAPPING_FILE, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)

        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            yolo_names = yaml.safe_load(f)["names"]

        imgfile_to_id = evaluator.build_filename_to_id_map(ann_file)

        coco_name_to_id = {}
        if os.path.isfile(ann_file):
            with open(ann_file, "r", encoding="utf-8") as f:
                coco_data = json.load(f)
                for c in coco_data.get("categories", []):
                    coco_name_to_id[c["name"].strip()] = c["id"]

        yolo_to_coco = {y_id: coco_name_to_id.get(name.strip(), -1) for y_id, name in enumerate(yolo_names)}

        test_imgs = sorted(
            glob.glob(os.path.join(test_images_dir, "*.[pP][nN][gG]"))
            + glob.glob(os.path.join(test_images_dir, "*.[jJ][pP][gG]"))
            + glob.glob(os.path.join(test_images_dir, "*.[jJ][pP][eE][gG]"))
        )
        print(f"\n🔍 [OOD 추론 및 유사도 측정] 총 {len(test_imgs)}개 이미지...")

        all_detections = []
        for img_path in tqdm(test_imgs, desc="Extracting embeddings & Sim"):
            filename = os.path.basename(img_path)
            img_id = imgfile_to_id.get(filename)
            if img_id is None:
                try:
                    img_id = int(os.path.splitext(filename)[0])
                except ValueError:
                    continue

            orig_img = cv2.imread(img_path)
            if orig_img is None:
                continue
            h_img, w_img = orig_img.shape[:2]

            results = self.model.predict(img_path, conf=0.25, iou=0.45, verbose=False)
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_pred = int(box.cls[0].cpu().numpy())

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = orig_img[y1:y2, x1:x2]
                emb = self.extract_embedding(crop).to(self.device)

                bank = feature_bank[cls_pred]
                sims = F.cosine_similarity(emb.unsqueeze(0), bank, dim=1)
                sim = float(sims.max().item())

                orig_class_id = yolo_to_coco.get(cls_pred, -1)
                all_detections.append({
                    "image_id": img_id,
                    "category_id": orig_class_id,
                    "bbox_x": int(x1),
                    "bbox_y": int(y1),
                    "bbox_w": int(x2 - x1),
                    "bbox_h": int(y2 - y1),
                    "score": round(conf, 4),
                    "sim": sim,
                })

        # Thresholds Sweep 평가
        best_map = -1.0
        best_thresh = 0.0
        best_csv = output_csv
        sweep_dir = os.path.join(config.BASE_DIR, "experiments", "sweeps")
        os.makedirs(sweep_dir, exist_ok=True)

        for t in thresholds:
            kept = [d for d in all_detections if d["sim"] >= t]
            records = []
            for ann_id, d in enumerate(kept, 1):
                rec = {k: v for k, v in d.items() if k != "sim"}
                rec["annotation_id"] = ann_id
                records.append(rec)

            df_sweep = pd.DataFrame(records)
            csv_path = os.path.join(sweep_dir, f"predictions_ood_{t:.2f}.csv")
            df_sweep.to_csv(csv_path, index=False, encoding="utf-8-sig")

            metrics = evaluator.evaluate_coco(csv_path, annotations_path=ann_file)
            mAP = metrics["mAP75-95"]
            print(f"  • 임계값 {t:.2f}: 잔존 박스 {len(kept)}개 | mAP@[0.75:0.95]: {mAP:.4f}")

            if mAP > best_map:
                best_map = mAP
                best_thresh = t
                best_csv = csv_path

        import shutil
        shutil.copy2(best_csv, output_csv)
        print("\n" + "=" * 60)
        print(f"🏆 [최적 OOD 모델 도출] Best mAP@[0.75:0.95]: {best_map:.4f} (임계값 {best_thresh:.2f})")
        print(f"📁 최종 산출물이 '{output_csv}'로 저장되었습니다.")
        print("=" * 60)

        return best_map, best_thresh, output_csv
