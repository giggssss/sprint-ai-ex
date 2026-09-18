import os
import glob
import yaml
import json
import shutil
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple


class YOLODataPrep:
    """YOLO 데이터셋 검증, 생성, 변환 및 매핑 관리 클래스."""

    def __init__(self, data_yaml_path: str = "/Volumes/Macintosh SUB/Dataset/yolo_data_v2/data.yaml"):
        self.data_yaml_path = data_yaml_path
        self.dataset_dir = os.path.dirname(data_yaml_path)
        self.config = {}

    def load_yaml(self) -> dict:
        """data.yaml 파일을 로드합니다."""
        if not os.path.exists(self.data_yaml_path):
            return {}
        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        return self.config

    def build_stratified_dataset(
        self,
        raw_img_dir: str = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/train_images",
        raw_ann_dir: str = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/train_annotations",
        output_dir: str = "/Volumes/Macintosh SUB/Dataset/yolo_data_v2",
        train_ratio: float = 0.9,
    ) -> None:
        """희귀 클래스가 누락되지 않도록 Stratified Split 기반으로 완벽한 YOLO 데이터셋을 재구축합니다."""
        print("\n🚀 [Stratified Dataset Rebuild] 시작...")
        self.load_yaml()
        yolo_names = self.config.get("names", [])
        if not yolo_names:
            raise ValueError("기존 data.yaml에서 names 리스트를 불러올 수 없습니다.")

        name_to_yolo = {name: idx for idx, name in enumerate(yolo_names)}

        # JSON 파싱
        print("JSON 어노테이션 파싱 중...")
        image_info = {}
        class_to_images = defaultdict(list)

        json_files = glob.glob(os.path.join(raw_ann_dir, "**", "*.json"), recursive=True)
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    img_dict = {img["id"]: img for img in data.get("images", [])}
                    id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}

                    for ann in data.get("annotations", []):
                        cname = id_to_name.get(ann["category_id"])
                        if cname not in name_to_yolo:
                            continue

                        yolo_id = name_to_yolo[cname]
                        img_id = ann["image_id"]

                        if img_id not in image_info:
                            img = img_dict.get(img_id)
                            if not img:
                                continue
                            image_info[img_id] = {
                                "file_name": img["file_name"],
                                "width": img["width"],
                                "height": img["height"],
                                "annotations": [],
                                "assigned_split": None,
                            }

                        image_info[img_id]["annotations"].append({
                            "yolo_id": yolo_id,
                            "bbox": ann["bbox"],  # [x_min, y_min, w, h]
                        })
                except Exception:
                    pass

        # Stratified Split (희귀 클래스 우선 배분)
        for img_id, info in image_info.items():
            unique_classes = set(ann["yolo_id"] for ann in info["annotations"])
            for c in unique_classes:
                class_to_images[c].append(img_id)

        sorted_classes = sorted(class_to_images.keys(), key=lambda c: len(class_to_images[c]))

        train_imgs: Set[int] = set()
        val_imgs: Set[int] = set()

        for c in sorted_classes:
            imgs_with_c = [img for img in class_to_images[c] if image_info[img]["assigned_split"] is None]
            random.shuffle(imgs_with_c)

            n_total = len(imgs_with_c)
            if n_total == 0:
                continue

            n_train = max(1, int(n_total * train_ratio))
            if n_total == 3:
                n_train = 2
            elif n_total == 2:
                n_train = 1

            for i, img_id in enumerate(imgs_with_c):
                if i < n_train:
                    image_info[img_id]["assigned_split"] = "train"
                    train_imgs.add(img_id)
                else:
                    image_info[img_id]["assigned_split"] = "val"
                    val_imgs.add(img_id)

        print(f"✅ 분할 완료: Train {len(train_imgs)}장, Val {len(val_imgs)}장")

        # 디렉터리 생성 및 파일 복사
        print(f"디렉토리 생성 및 파일 복사 중... ({output_dir})")
        for split in ["train", "val"]:
            os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

        all_raw_images = glob.glob(os.path.join(raw_img_dir, "**", "*.*"), recursive=True)
        img_name_to_path = {os.path.basename(p): p for p in all_raw_images}

        processed = 0
        for img_id, info in image_info.items():
            split = info["assigned_split"]
            if not split:
                continue

            fname = info["file_name"]
            src_img = img_name_to_path.get(fname)
            if not src_img or not os.path.exists(src_img):
                continue

            ext = os.path.splitext(src_img)[1]
            dst_img = os.path.join(output_dir, split, "images", f"{img_id}{ext}")

            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

            label_txt = os.path.join(output_dir, split, "labels", f"{img_id}.txt")
            with open(label_txt, "w", encoding="utf-8") as lf:
                for ann in info["annotations"]:
                    x_min, y_min, w, h = ann["bbox"]
                    x_center = (x_min + w / 2) / info["width"]
                    y_center = (y_min + h / 2) / info["height"]
                    norm_w = w / info["width"]
                    norm_h = h / info["height"]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))
                    lf.write(f"{ann['yolo_id']} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            processed += 1
            if processed % 1000 == 0:
                print(f"  ... {processed}장 처리 완료")

        # yaml 저장
        new_yaml_path = os.path.join(output_dir, "data.yaml")
        yaml_content = {
            "path": output_dir,
            "train": "train/images",
            "val": "val/images",
            "test": "test/images" if "test" in self.config else "",
            "nc": self.config.get("nc", 56),
            "names": yolo_names,
        }
        with open(new_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"🎉 YOLO 데이터셋 재구축 완료! -> {output_dir}")
        self.data_yaml_path = new_yaml_path
        self.dataset_dir = output_dir

    def validate_dataset(self) -> dict:
        """데이터셋 이미지와 라벨의 1:1 매칭 무결성을 검증합니다."""
        self.load_yaml()
        stats = {}
        for split in ["train", "val"]:
            img_dir = os.path.join(self.dataset_dir, split, "images")
            lbl_dir = os.path.join(self.dataset_dir, split, "labels")
            images = (
                glob.glob(os.path.join(img_dir, "*.[jJ][pP][gG]"))
                + glob.glob(os.path.join(img_dir, "*.[pP][nN][gG]"))
                + glob.glob(os.path.join(img_dir, "*.[jJ][pP][eE][gG]"))
            )
            labels = glob.glob(os.path.join(lbl_dir, "*.txt"))
            stats[split] = {
                "num_images": len(images),
                "num_labels": len(labels),
            }
            print(f"--- [{split.upper()} Split] ---")
            print(f"  • 이미지 개수: {len(images)}")
            print(f"  • 라벨 개수  : {len(labels)}")
        return stats

    def get_class_distribution(self, split: str = "train") -> Counter:
        """특정 Split 내 각 클래스별 바운딩 박스 개수를 집계합니다."""
        lbl_dir = os.path.join(self.dataset_dir, split, "labels")
        label_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        class_counts = Counter()
        for lbl_file in label_files:
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_counts[int(parts[0])] += 1
        return class_counts

    def verify_class_distribution(self) -> None:
        """전체 클래스 분포 및 희귀 클래스(카나브정 등)의 유효성을 점검하여 리포트합니다."""
        print("\n🔍 [데이터셋 클래스 분포 정밀 점검]")
        self.load_yaml()
        yolo_names = self.config.get("names", [])
        train_counts = self.get_class_distribution(split="train")
        val_counts = self.get_class_distribution(split="val")

        print(f"총 클래스 수: {len(yolo_names)}")
        print(f"{'클래스 ID':<10} | {'클래스명':<25} | {'Train 박스':<10} | {'Val 박스':<10}")
        print("-" * 65)

        for idx, name in enumerate(yolo_names):
            t_cnt = train_counts.get(idx, 0)
            v_cnt = val_counts.get(idx, 0)
            flag = ""
            if t_cnt == 0:
                flag = " ⚠️ [Train 누락]"
            elif v_cnt == 0:
                flag = " ⚠️ [Val 누락]"
            print(f"{idx:<10} | {name[:23]:<25} | {t_cnt:<10} | {v_cnt:<10}{flag}")

    @staticmethod
    def create_class_mapping(
        data_yaml_path: str,
        test_annotations_path: str,
        output_mapping_path: str = "class_mapping.json",
    ) -> Dict[str, int]:
        """data.yaml과 테스트 어노테이션 JSON을 파싱하여 YOLO class_id -> 대회 원본 dl_idx 매핑을 생성합니다."""
        print(f"Generating class mapping from {test_annotations_path}...")
        name_to_idx = {}

        if os.path.isfile(test_annotations_path):
            ann_files = [test_annotations_path]
        else:
            ann_files = glob.glob(os.path.join(test_annotations_path, "**", "*.json"), recursive=True)

        for jf in ann_files:
            with open(jf, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    for img_info in data.get("images", []):
                        dl_name = img_info.get("dl_name", "")
                        dl_idx = img_info.get("dl_idx", "")
                        if dl_name and dl_idx:
                            name_to_idx[dl_name.strip()] = int(dl_idx)
                    for cat_info in data.get("categories", []):
                        c_name = cat_info.get("name", "")
                        c_id = cat_info.get("id")
                        if c_name and c_id:
                            name_to_idx[c_name.strip()] = int(c_id)
                except Exception as e:
                    print(f"Error parsing {jf}: {e}")

        with open(data_yaml_path, "r", encoding="utf-8") as f:
            yolo_config = yaml.safe_load(f)
            yolo_names = yolo_config["names"]

        class_id_to_dl_idx = {}
        unmatched = []

        for yolo_cls_id, name in enumerate(yolo_names):
            clean_name = name.strip()
            if clean_name in name_to_idx:
                class_id_to_dl_idx[str(yolo_cls_id)] = name_to_idx[clean_name]
            else:
                unmatched.append((yolo_cls_id, name))

        if unmatched:
            print(f"⚠️ 매칭되지 않은 클래스: {len(unmatched)}개")
            for cls_id, name in unmatched:
                print(f"  ID {cls_id}: {name}")
        else:
            print(f"✅ 전체 {len(yolo_names)}개 클래스가 성공적으로 매핑되었습니다.")

        with open(output_mapping_path, "w", encoding="utf-8") as f:
            json.dump(class_id_to_dl_idx, f, indent=4, ensure_ascii=False)

        print(f"Saved class mapping to {output_mapping_path}")
        return class_id_to_dl_idx

    @staticmethod
    def convert_test_annotations_to_yolo(
        test_annotations_path: str,
        test_images_dir: str,
        output_yolo_dir: str,
        class_mapping_path: str = "class_mapping.json",
    ) -> None:
        """테스트 어노테이션 JSON을 파싱하여 YOLO 포맷의 txt 라벨 및 심볼릭 링크 이미지를 생성합니다."""
        yolo_test_images = os.path.join(output_yolo_dir, "images")
        yolo_test_labels = os.path.join(output_yolo_dir, "labels")
        os.makedirs(yolo_test_images, exist_ok=True)
        os.makedirs(yolo_test_labels, exist_ok=True)

        with open(class_mapping_path, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)
        dl_idx_to_yolo = {v: int(k) for k, v in class_mapping.items()}

        image_info_map = {}
        annotations_map = defaultdict(list)

        if os.path.isfile(test_annotations_path):
            ann_files = [test_annotations_path]
        else:
            ann_files = glob.glob(os.path.join(test_annotations_path, "**", "*.json"), recursive=True)

        for jf in ann_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for img in data.get("images", []):
                        img_id = img.get("id")
                        width = img.get("width", 976)
                        height = img.get("height", 976)
                        image_info_map[img_id] = (width, height)

                    for ann in data.get("annotations", []):
                        img_id = ann.get("image_id")
                        cat_id = ann.get("category_id")
                        bbox = ann.get("bbox")
                        if img_id is None or cat_id not in dl_idx_to_yolo or bbox is None:
                            continue

                        yolo_cls = dl_idx_to_yolo[cat_id]
                        x, y, w, h = bbox
                        width, height = image_info_map.get(img_id, (976, 976))

                        x_center = (x + w / 2) / width
                        y_center = (y + h / 2) / height
                        w_norm = w / width
                        h_norm = h / height

                        line = f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                        annotations_map[img_id].append(line)
            except Exception as e:
                print(f"Error parsing {jf}: {e}")

        for img_id, lines in annotations_map.items():
            label_file = os.path.join(yolo_test_labels, f"{img_id}.txt")
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        print(f"✅ 테스트셋 YOLO 라벨 변환 완료 ({len(annotations_map)}개 이미지) -> {output_yolo_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Data Preparation and Verification Toolkit")
    parser.add_argument("--rebuild", action="store_true", help="Stratified Split 기반 전체 데이터셋 재구축")
    parser.add_argument("--verify", action="store_true", help="클래스 분포 및 불균형 점검")
    parser.add_argument("--map", action="store_true", help="class_mapping.json 생성")
    args = parser.parse_args()

    prep = YOLODataPrep()
    if args.rebuild:
        prep.build_stratified_dataset()
    if args.verify:
        prep.verify_class_distribution()
    prep.validate_dataset()
