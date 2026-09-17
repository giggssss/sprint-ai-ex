import os
import glob
import yaml
import json
import shutil
import random
from collections import Counter, defaultdict


class YOLODataPrep:
    """YOLO 데이터셋 검증, 생성 및 data.yaml 관리 클래스."""

    def __init__(self, data_yaml_path: str = "/Volumes/Macintosh SUB/Dataset/yolo_data/data.yaml"):
        self.data_yaml_path = data_yaml_path
        self.dataset_dir = os.path.dirname(data_yaml_path)
        self.config = {}

    def load_yaml(self) -> dict:
        if not os.path.exists(self.data_yaml_path):
            return {}
        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        return self.config

    def build_stratified_dataset(self, 
                                 raw_img_dir="/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/train_images",
                                 raw_ann_dir="/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/train_annotations",
                                 output_dir="/Volumes/Macintosh SUB/Dataset/yolo_data_v2",
                                 train_ratio=0.9):
        """희귀 클래스가 누락되지 않도록 Stratified Split 기반으로 완벽한 YOLO 데이터셋을 재구축합니다."""
        print("\n🚀 [Stratified Dataset Rebuild] 시작...")
        
        self.load_yaml()
        yolo_names = self.config.get('names', [])
        if not yolo_names:
            raise ValueError("기존 data.yaml에서 names 리스트를 불러올 수 없습니다.")
            
        name_to_yolo = {name: idx for idx, name in enumerate(yolo_names)}

        # 2. Parse all JSONs
        print("JSON 어노테이션 파싱 중...")
        image_info = {} 
        class_to_images = defaultdict(list)
        
        json_files = glob.glob(os.path.join(raw_ann_dir, "**", "*.json"), recursive=True)
        for jf in json_files:
            with open(jf, "r") as f:
                try:
                    data = json.load(f)
                    img_dict = {img['id']: img for img in data.get('images', [])}
                    id_to_name = {c['id']: c['name'] for c in data.get('categories', [])}
                    
                    for ann in data.get('annotations', []):
                        cname = id_to_name.get(ann['category_id'])
                        if cname not in name_to_yolo:
                            continue
                        
                        yolo_id = name_to_yolo[cname]
                        img_id = ann['image_id']
                        
                        if img_id not in image_info:
                            img = img_dict.get(img_id)
                            if not img: continue
                            image_info[img_id] = {
                                'file_name': img['file_name'],
                                'width': img['width'],
                                'height': img['height'],
                                'annotations': [],
                                'assigned_split': None
                            }
                        
                        image_info[img_id]['annotations'].append({
                            'yolo_id': yolo_id,
                            'bbox': ann['bbox'] # [x_min, y_min, w, h]
                        })
                except Exception as e:
                    pass

        # 3. Stratified Split Logic (희귀 클래스 우선 배분)
        for img_id, info in image_info.items():
            unique_classes = set(ann['yolo_id'] for ann in info['annotations'])
            for c in unique_classes:
                class_to_images[c].append(img_id)
                
        # 희귀 클래스(빈도 오름차순)부터 먼저 처리하여 무조건 분배
        sorted_classes = sorted(class_to_images.keys(), key=lambda c: len(class_to_images[c]))
        
        train_imgs = set()
        val_imgs = set()
        
        for c in sorted_classes:
            # 아직 배정되지 않은 이미지들
            imgs_with_c = [img for img in class_to_images[c] if image_info[img]['assigned_split'] is None]
            random.shuffle(imgs_with_c)
            
            n_total = len(imgs_with_c)
            if n_total == 0:
                continue
                
            n_train = max(1, int(n_total * train_ratio))
            # 극소수 클래스 수동 보정 (카나브정 등)
            if n_total == 3:
                n_train = 2
            elif n_total == 2:
                n_train = 1
                
            for i, img_id in enumerate(imgs_with_c):
                if i < n_train:
                    image_info[img_id]['assigned_split'] = 'train'
                    train_imgs.add(img_id)
                else:
                    image_info[img_id]['assigned_split'] = 'val'
                    val_imgs.add(img_id)
                    
        print(f"✅ 분할 완료: Train {len(train_imgs)}장, Val {len(val_imgs)}장")
        
        # 4. 파일 생성 및 복사
        print(f"디렉토리 생성 및 파일 복사 중... ({output_dir})")
        for split in ['train', 'val']:
            os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)
        
        # 빠른 파일 매핑을 위해 이미지 딕셔너리 생성
        all_raw_images = glob.glob(os.path.join(raw_img_dir, "**", "*.*"), recursive=True)
        img_name_to_path = {os.path.basename(p): p for p in all_raw_images}
        
        processed = 0
        for img_id, info in image_info.items():
            split = info['assigned_split']
            if not split: continue
            
            fname = info['file_name']
            src_img = img_name_to_path.get(fname)
            if not src_img or not os.path.exists(src_img):
                continue
                
            ext = os.path.splitext(src_img)[1]
            dst_img = os.path.join(output_dir, split, "images", f"{img_id}{ext}")
            
            # 파일 복사
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
                
            # 라벨 생성 (YOLO Format)
            label_txt = os.path.join(output_dir, split, "labels", f"{img_id}.txt")
            with open(label_txt, "w") as lf:
                for ann in info['annotations']:
                    x_min, y_min, w, h = ann['bbox']
                    # Normalize
                    x_center = (x_min + w / 2) / info['width']
                    y_center = (y_min + h / 2) / info['height']
                    norm_w = w / info['width']
                    norm_h = h / info['height']
                    # 클리핑 (0~1 사이)
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))
                    lf.write(f"{ann['yolo_id']} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
            processed += 1
            if processed % 1000 == 0:
                print(f"  ... {processed}장 처리 완료")
                
        # 5. 기존 yaml 기반으로 새 yaml 생성
        new_yaml_path = os.path.join(output_dir, "data.yaml")
        yaml_content = {
            'path': output_dir,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images' if 'test' in self.config else '',
            'nc': self.config.get('nc', 56),
            'names': yolo_names
        }
        with open(new_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
            
        print(f"🎉 YOLO 데이터셋 재구축 완료! -> {output_dir}")
        self.data_yaml_path = new_yaml_path
        self.dataset_dir = output_dir

    def validate_dataset(self) -> dict:
        self.load_yaml()
        stats = {}
        for split in ["train", "val"]:
            img_dir = os.path.join(self.dataset_dir, split, "images")
            lbl_dir = os.path.join(self.dataset_dir, split, "labels")
            images = glob.glob(os.path.join(img_dir, "*.[jJ][pP][gG]")) + \
                     glob.glob(os.path.join(img_dir, "*.[pP][nN][gG]")) + \
                     glob.glob(os.path.join(img_dir, "*.[jJ][pP][eE][gG]"))
            labels = glob.glob(os.path.join(lbl_dir, "*.txt"))
            img_basenames = {os.path.splitext(os.path.basename(p))[0] for p in images}
            lbl_basenames = {os.path.splitext(os.path.basename(p))[0] for p in labels}
            unmatched_imgs = img_basenames - lbl_basenames
            unmatched_lbls = lbl_basenames - img_basenames
            stats[split] = {
                "num_images": len(images),
                "num_labels": len(labels),
            }
            print(f"--- [{split.upper()} Split] ---")
            print(f"  • 이미지 개수: {len(images)}")
            print(f"  • 라벨 개수  : {len(labels)}")
        return stats

    def get_class_distribution(self, split: str = "train") -> Counter:
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Stratified Split 기반 전체 데이터셋 재구축")
    args = parser.parse_args()
    
    prep = YOLODataPrep()
    if args.rebuild:
        prep.build_stratified_dataset()
        
    prep.validate_dataset()
