import os
import json
import glob
import shutil

def convert_test_data():
    test_ann_dir = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_annotations"
    test_img_dir = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_images"
    
    yolo_test_images = "/Volumes/Macintosh SUB/Dataset/yolo_data/test/images"
    yolo_test_labels = "/Volumes/Macintosh SUB/Dataset/yolo_data/test/labels"
    
    os.makedirs(yolo_test_images, exist_ok=True)
    os.makedirs(yolo_test_labels, exist_ok=True)
    
    # 1. Load class mapping and invert it (dl_idx -> yolo_cls)
    mapping_path = "class_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)
    
    # class_mapping is {yolo_cls (str): dl_idx (int)}
    dl_idx_to_yolo = {v: int(k) for k, v in class_mapping.items()}
    
    # 2. Parse all JSON files to group bounding boxes by image_id
    # Also keep track of image dimensions if available. If not, we might need to read the image size.
    # YOLO format requires normalized coordinates: x_center, y_center, width, height
    # test images are usually 976x976 or similar. Let's read from JSON images section.
    
    json_files = glob.glob(os.path.join(test_ann_dir, "**", "*.json"), recursive=True)
    image_info_map = {} # image_id -> {width, height}
    annotations_map = {} # image_id -> list of yolo formatted lines
    
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Parse images info
                for img in data.get("images", []):
                    img_id = img.get("id")
                    width = img.get("width", 976)
                    height = img.get("height", 976)
                    image_info_map[img_id] = (width, height)
                    
                # Parse annotations
                for ann in data.get("annotations", []):
                    img_id = ann.get("image_id")
                    cat_id = ann.get("category_id")
                    bbox = ann.get("bbox") # [x, y, w, h]
                    
                    if img_id is None or cat_id not in dl_idx_to_yolo or bbox is None:
                        continue
                        
                    yolo_cls = dl_idx_to_yolo[cat_id]
                    
                    x, y, w, h = bbox
                    if img_id not in image_info_map:
                        width, height = 976, 976 # fallback
                    else:
                        width, height = image_info_map[img_id]
                        
                    # Normalize
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    line = f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                    
                    if img_id not in annotations_map:
                        annotations_map[img_id] = []
                    annotations_map[img_id].append(line)
                    
            except Exception as e:
                print(f"Error parsing {jf}: {e}")
                
    # 3. Create labels and symlink images
    print(f"Parsed annotations for {len(annotations_map)} images.")
    valid_images = 0
    
    for img_id, lines in annotations_map.items():
        src_img_png = os.path.join(test_img_dir, f"{img_id}.png")
        src_img_jpg = os.path.join(test_img_dir, f"{img_id}.jpg")
        
        src_img = None
        if os.path.exists(src_img_png):
            src_img = src_img_png
        elif os.path.exists(src_img_jpg):
            src_img = src_img_jpg
            
        if src_img:
            dst_img = os.path.join(yolo_test_images, os.path.basename(src_img))
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
                
            dst_lbl = os.path.join(yolo_test_labels, f"{img_id}.txt")
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            valid_images += 1
            
    print(f"Successfully prepared {valid_images} test images and labels for YOLO validation.")

if __name__ == "__main__":
    convert_test_data()
