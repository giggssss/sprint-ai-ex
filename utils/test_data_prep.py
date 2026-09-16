import os
import json
import glob
import yaml

def create_class_mapping(data_yaml_path, test_annotations_dir, output_mapping_path):
    """
    data.yaml의 클래스 이름과 test_annotations의 JSON 파일들의 dl_name을 대조하여
    YOLO class_id (0~55) -> original dl_idx 맵핑을 생성합니다.
    """
    print("Loading YOLO classes from data.yaml...")
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    yolo_names = data_config.get("names", [])
    
    # 맵핑을 위한 딕셔너리
    name_to_idx = {}
    
    print(f"Parsing JSON files in {test_annotations_dir}...")
    json_files = glob.glob(os.path.join(test_annotations_dir, "**", "*.json"), recursive=True)
    
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for img_info in data.get("images", []):
                    dl_name = img_info.get("dl_name", "")
                    dl_idx = img_info.get("dl_idx", "")
                    
                    if dl_name and dl_idx:
                        # Clean up names for better matching if needed, though exact match is preferred
                        name_to_idx[dl_name.strip()] = int(dl_idx)
                        
            except Exception as e:
                print(f"Error parsing {jf}: {e}")
                
    # YOLO 클래스 이름(0-55)을 기반으로 매핑 테이블 생성
    class_id_to_dl_idx = {}
    unmapped_classes = []
    
    for i, name in enumerate(yolo_names):
        name = name.strip()
        if name in name_to_idx:
            class_id_to_dl_idx[i] = name_to_idx[name]
        else:
            unmapped_classes.append(name)
            
    # 매핑되지 않은 경우, 부분 일치 시도
    for name in unmapped_classes[:]:
        for dl_name, dl_idx in name_to_idx.items():
            if name in dl_name or dl_name in name:
                class_id_to_dl_idx[yolo_names.index(name)] = dl_idx
                unmapped_classes.remove(name)
                break

    print(f"Successfully mapped {len(class_id_to_dl_idx)} / {len(yolo_names)} classes.")
    if unmapped_classes:
        print(f"Unmapped classes: {unmapped_classes}")
        
    with open(output_mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_id_to_dl_idx, f, ensure_ascii=False, indent=4)
        
    print(f"Mapping saved to {output_mapping_path}")
    return class_id_to_dl_idx

if __name__ == "__main__":
    DATA_YAML = "/Volumes/Macintosh SUB/Dataset/yolo_data/data.yaml"
    TEST_ANN_DIR = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_annotations"
    OUTPUT_MAP = "class_mapping.json"
    
    create_class_mapping(DATA_YAML, TEST_ANN_DIR, OUTPUT_MAP)
