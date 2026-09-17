import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from ultralytics import YOLO
from tqdm import tqdm
import json

WEIGHTS = "runs/detect/runs/detect/train_yolo11s_v2_dataset/weights/best.pt"
TRAIN_IMAGES = "/Volumes/Macintosh SUB/Dataset/yolo_data_v2/train/images"
TRAIN_LABELS = "/Volumes/Macintosh SUB/Dataset/yolo_data_v2/train/labels"
TEST_IMAGES = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_images"
TEST_ANNS = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_annotations"
CLASS_MAP = "class_mapping.json"
OUTPUT_CSV = "predictions.csv"

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

def get_backbone(model):
    # Extracts the YOLO backbone up to the SPPF layer
    # model.model.model is the sequential module
    # Layer 9 is usually SPPF in YOLOv8/11
    backbone = torch.nn.Sequential(*list(model.model.model.children())[:10])
    backbone.eval()
    return backbone

def extract_embedding(backbone, img_crop):
    # img_crop: BGR numpy array
    img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        x = backbone(img) # shape: (1, C, H, W)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1) # shape: (1, C)
        x = F.normalize(x, p=2, dim=1) # L2 normalize
    return x.squeeze(0).cpu()

def build_feature_bank(backbone, num_classes=56):
    print("Building Feature Bank from train set...")
    feature_bank = {i: [] for i in range(num_classes)}
    
    train_imgs = glob.glob(os.path.join(TRAIN_IMAGES, "*.*"))
    for img_path in tqdm(train_imgs, desc="Extracting Train Features"):
        label_path = os.path.join(TRAIN_LABELS, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        if not os.path.exists(label_path): continue
        
        orig_img = cv2.imread(img_path)
        if orig_img is None: continue
        h_img, w_img = orig_img.shape[:2]
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            
            x1 = int((cx - w/2) * w_img)
            y1 = int((cy - h/2) * h_img)
            x2 = int((cx + w/2) * w_img)
            y2 = int((cy + h/2) * h_img)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 <= x1 or y2 <= y1: continue
            
            crop = orig_img[y1:y2, x1:x2]
            emb = extract_embedding(backbone, crop)
            feature_bank[cls_id].append(emb)
            
    # Convert lists to tensors
    for cls_id in range(num_classes):
        if len(feature_bank[cls_id]) > 0:
            feature_bank[cls_id] = torch.stack(feature_bank[cls_id]).to(device)
        else:
            print(f"Warning: No train samples for class {cls_id}")
            feature_bank[cls_id] = torch.zeros((1, 512)).to(device)
            
    return feature_bank

def run_ood_inference_all_thresholds(model_path, backbone, feature_bank, thresholds=[0.85, 0.90, 0.92, 0.95, 0.97]):
    print(f"\n--- Running OOD Inference for Thresholds: {thresholds} ---")
    model = YOLO(model_path)
    
    with open(CLASS_MAP, 'r') as f:
        class_mapping = json.load(f)
        
    import yaml
    with open('/Volumes/Macintosh SUB/Dataset/yolo_data_v2/data.yaml', 'r') as f:
        yolo_names = yaml.safe_load(f)['names']
        
    test_anns = glob.glob(os.path.join(TEST_ANNS, "**", "*.json"), recursive=True)
    imgfile_to_id = {}
    coco_name_to_id = {}
    for jf in test_anns:
        with open(jf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for img_info in data.get('images', []):
                    imgfile_to_id[img_info['file_name']] = img_info['id']
                for c in data.get('categories', []):
                    coco_name_to_id[c['name']] = c['id']
            except:
                pass
                
    yolo_to_coco = {}
    for yolo_id, name in enumerate(yolo_names):
        yolo_to_coco[yolo_id] = coco_name_to_id.get(name, -1)

    all_detections = []
    test_imgs = glob.glob(os.path.join(TEST_IMAGES, "*.*"))
    
    for img_path in tqdm(test_imgs, desc="Extracting embeddings"):
        filename = os.path.basename(img_path)
        try:
            img_id = int(os.path.splitext(filename)[0])
        except ValueError:
            continue
        orig_img = cv2.imread(img_path)
        if orig_img is None: continue
        h_img, w_img = orig_img.shape[:2]
        
        results = model.predict(img_path, conf=0.25, iou=0.45, verbose=False)
        boxes = results[0].boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_pred = int(box.cls[0].cpu().numpy())
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 <= x1 or y2 <= y1: continue
            
            crop = orig_img[y1:y2, x1:x2]
            emb = extract_embedding(backbone, crop).to(device)
            
            bank = feature_bank[cls_pred]
            sims = F.cosine_similarity(emb.unsqueeze(0), bank, dim=1)
            sim = sims.max().item()
            
            orig_class_id = yolo_to_coco.get(cls_pred, -1)
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            all_detections.append({
                "image_id": img_id,
                "category_id": orig_class_id,
                "bbox": [int(x1), int(y1), int(bbox_w), int(bbox_h)],
                "score": conf,
                "sim": sim
            })
            
    csv_paths = []
    for t in thresholds:
        kept = [d for d in all_detections if d["sim"] >= t]
        dropped = len(all_detections) - len(kept)
        print(f"\nThreshold {t}: Kept {len(kept)}, Dropped {dropped}")
        
        df = pd.DataFrame([{k:v for k,v in d.items() if k != 'sim'} for d in kept])
        csv_name = f"predictions_ood_{t:.2f}.csv"
        df.to_csv(csv_name, index=False)
        csv_paths.append((t, csv_name))
        
    return csv_paths

def evaluate_csv(csv_path):
    print(f"\n--- Evaluating {csv_path} ---")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    coco_gt = COCO("test_coco.json")
    
    # Read predictions
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("No predictions to evaluate.")
        return 0.0
        
    # Convert string '[x, y, w, h]' back to list if pandas parsed it as string
    import ast
    if isinstance(df.iloc[0]['bbox'], str):
        df['bbox'] = df['bbox'].apply(ast.literal_eval)
        
    res_json = df.to_dict('records')
    
    coco_dt = coco_gt.loadRes(res_json)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # We only want to evaluate on the 56 known classes.
    import yaml
    with open('/Volumes/Macintosh SUB/Dataset/yolo_data_v2/data.yaml', 'r') as f:
        yolo_names = yaml.safe_load(f)['names']
        
    known_coco_ids = []
    for cat in coco_gt.loadCats(coco_gt.getCatIds()):
        if cat['name'] in yolo_names:
            known_coco_ids.append(cat['id'])
            
    coco_eval.params.catIds = known_coco_ids
    
    # Calculate mAP (default iouThrs is 0.5:0.05:0.95)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    mAP_50_95 = coco_eval.stats[0] # AP at 0.50:0.95
    mAP_50 = coco_eval.stats[1]    # AP at IoU=0.50
    mAP_75 = coco_eval.stats[2]    # AP at IoU=0.75
    
    print(f"🔥 mAP@[0.50:0.95]: {mAP_50_95:.4f}")
    print(f"🔥 mAP@0.50: {mAP_50:.4f}")
    print(f"🔥 mAP@0.75: {mAP_75:.4f}")
    return mAP_50

if __name__ == "__main__":
    model = YOLO(WEIGHTS)
    backbone = get_backbone(model).to(device)
    
    feature_bank = build_feature_bank(backbone, num_classes=56)
    
    best_map = 0
    best_thresh = 0
    csv_paths = run_ood_inference_all_thresholds(WEIGHTS, backbone, feature_bank)
    for t, csv_path in csv_paths:
        mAP = evaluate_csv(csv_path)
        if mAP > best_map:
            best_map = mAP
            best_thresh = t
            
    print(f"\n🚀 Best mAP@0.50: {best_map:.4f} at Threshold {best_thresh:.2f}")
