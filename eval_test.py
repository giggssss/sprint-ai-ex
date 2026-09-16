import os
import glob
import json
import pandas as pd
from ultralytics import YOLO

def generate_predictions_csv(weights_path, test_images_dir, test_annotations_dir, class_mapping_path, output_csv):
    """
    테스트 이미지들에 대해 YOLO 모델 추론을 수행하고,
    original dl_idx(오리지널 약품코드)로 클래스를 매핑하여 지정된 포맷의 CSV를 생성합니다.
    """
    # 1. 모델 로드
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    print(f"Loading YOLO model from {weights_path}...")
    model = YOLO(weights_path)
    
    # 2. 클래스 매핑 로드 (yolo class index -> original dl_idx)
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)
        
    # 3. 이미지 ID 매핑 생성 (filename -> image_id)
    # JSON에서 image_id 추출
    filename_to_image_id = {}
    json_files = glob.glob(os.path.join(test_annotations_dir, "**", "*.json"), recursive=True)
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for img in data.get("images", []):
                    fn = img.get("file_name")
                    img_id = img.get("id")
                    if fn and img_id is not None:
                        filename_to_image_id[fn] = img_id
            except Exception:
                pass
                
    # 4. 테스트 이미지 로드
    image_paths = glob.glob(os.path.join(test_images_dir, "*.png")) + \
                  glob.glob(os.path.join(test_images_dir, "*.jpg"))
    print(f"Found {len(image_paths)} test images.")
    
    records = []
    annotation_id = 1
    
    # 5. 추론 수행
    # 추론은 배치로 수행하는 것이 빠름
    batch_size = 16
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        # device='cpu' to avoid MPS issues if any, or just let ultralytics decide
        results = model.predict(source=batch_paths, conf=0.01, iou=0.6, device='cpu', verbose=False)
        
        for r, img_path in zip(results, batch_paths):
            filename = os.path.basename(img_path)
            # 매핑된 ID가 없으면 임의의 ID 부여(안전장치)
            image_id = filename_to_image_id.get(filename, abs(hash(filename)) % (10**8))
            
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    yolo_cls_id = int(box.cls[0].item())
                    # 원래 약품 코드로 변환
                    original_dl_idx = class_mapping.get(str(yolo_cls_id), yolo_cls_id)
                    
                    conf = float(box.conf[0].item())
                    # bbox: [x, y, w, h] 포맷
                    # YOLO xywh는 center x, center y 기반이므로 xyxy에서 변환
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
                        "score": round(conf, 4)
                    })
                    annotation_id += 1
                    
        print(f"Processed {min(i+batch_size, len(image_paths))} / {len(image_paths)} images")

    # 6. CSV 저장
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Prediction CSV saved to {output_csv} with {len(df)} bounding boxes.")

if __name__ == "__main__":
    WEIGHTS = "models/best.pt"
    TEST_IMAGES = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_images"
    TEST_ANNS = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_annotations"
    CLASS_MAP = "class_mapping.json"
    OUTPUT_CSV = "predictions.csv"
    
    generate_predictions_csv(WEIGHTS, TEST_IMAGES, TEST_ANNS, CLASS_MAP, OUTPUT_CSV)
