import os
import json
import streamlit as st
import pandas as pd
import glob
from PIL import Image, ImageDraw
import random
from ultralytics import YOLO
import yaml

st.set_page_config(page_title="Prediction Viewer", layout="wide")

st.title("🔍 모델 추론 결과 확인 툴 (Test Set Viewer)")

# -----------------
# 1. 공통 헬퍼 함수
# -----------------
@st.cache_resource
def load_yolo_model(weights_path):
    if os.path.exists(weights_path):
        return YOLO(weights_path)
    return None

@st.cache_data
def load_class_mapping(mapping_path):
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def get_dl_idx_to_name(data_yaml_path, mapping_path):
    dl_idx_to_name = {}
    if os.path.exists(data_yaml_path) and os.path.exists(mapping_path):
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            yolo_names = yaml.safe_load(f).get("names", [])
        with open(mapping_path, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)
            
        for yolo_idx_str, dl_idx in class_mapping.items():
            yolo_idx = int(yolo_idx_str)
            if yolo_idx < len(yolo_names):
                dl_idx_to_name[dl_idx] = yolo_names[yolo_idx]
    return dl_idx_to_name

def load_data(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty and not list(df.columns):
            return None
        return df
    except pd.errors.EmptyDataError:
        return None

@st.cache_data
def get_image_list(img_dir):
    paths = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
    return {os.path.basename(p): p for p in paths}

def generate_colors(num_colors):
    random.seed(42)
    return ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(num_colors)]

# -----------------
# 2. 전역 설정 및 로드
# -----------------
CSV_PATH = "predictions.csv"
TEST_IMG_DIR = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_images"
WEIGHTS_PATH = "models/best.pt"
MAPPING_PATH = "class_mapping.json"
DATA_YAML_PATH = "/Volumes/Macintosh SUB/Dataset/yolo_data/data.yaml"

# 최적의 Confidence Threshold 읽어오기 (기본값 0.25)
OPTIMAL_CONF = 0.25
if os.path.exists("optimal_conf.txt"):
    with open("optimal_conf.txt", "r") as f:
        try:
            OPTIMAL_CONF = float(f.read().strip())
        except:
            pass

df = load_data(CSV_PATH)
image_map = get_image_list(TEST_IMG_DIR)
model = load_yolo_model(WEIGHTS_PATH)
class_mapping = load_class_mapping(MAPPING_PATH)
dl_idx_to_name = get_dl_idx_to_name(DATA_YAML_PATH, MAPPING_PATH)

# 사이드바 모드 선택
st.sidebar.header("설정")
mode = st.sidebar.radio("확인 방식 선택", ["직접 파일 업로드 (Live Inference)", "기존 테스트셋 조회 (CSV)"])

if mode == "기존 테스트셋 조회 (CSV)":
    if df is None or not image_map:
        st.warning("테스트셋 결과(CSV)나 이미지 디렉토리를 찾을 수 없습니다. '직접 파일 업로드' 기능을 사용해 보세요.")
    else:
        st.sidebar.info("CSV 포맷 제한으로 인해, 현재는 폴더 내 이미지를 직접 선택하면 해당 이미지에 매칭되는 결과를 보여줍니다.")
        selected_filename = st.sidebar.selectbox("테스트 이미지 선택", list(image_map.keys()))

        @st.cache_data
        def get_filename_to_image_id_mapping(test_annotations_dir):
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
            return filename_to_image_id

        TEST_ANNS = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_annotations"
        mapping = get_filename_to_image_id_mapping(TEST_ANNS)
        
        target_image_id = mapping.get(selected_filename)
        if target_image_id is None:
            try:
                target_image_id = int(os.path.splitext(selected_filename)[0])
            except ValueError:
                import hashlib
                target_image_id = int(hashlib.md5(selected_filename.encode()).hexdigest(), 16) % (10**8)
            
        img_df = df[df['image_id'] == target_image_id]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            img_path = image_map[selected_filename]
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            
            unique_categories = df['category_id'].unique() if not df.empty else []
            colors = generate_colors(len(unique_categories) + 100)
            cat_to_color = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}
            
            if not img_df.empty:
                for _, row in img_df.iterrows():
                    x, y, w, h = row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']
                    cat = row['category_id']
                    score = row['score']
                    color = cat_to_color.get(cat, "#ff0000")
                    
                    draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
                    text = f"ID:{cat} ({score:.2f})"
                    try:
                        t_bbox = draw.textbbox((x, y), text)
                        draw.rectangle([t_bbox[0], t_bbox[1], t_bbox[2], t_bbox[3]], fill=color)
                    except AttributeError:
                        pass
                    draw.text((x, y), text, fill="white")
                    
            st.image(image, use_container_width=True, caption=f"Selected: {selected_filename} (Image ID: {target_image_id})")

        with col2:
            st.subheader("예측 데이터 (CSV)")
            st.write(f"탐지된 객체 수: {len(img_df)}")
            
            if not img_df.empty:
                formatted_df = pd.DataFrame()
                formatted_df['카테고리 아이디'] = img_df['category_id']
                formatted_df['카테고리 이름'] = img_df['category_id'].map(dl_idx_to_name)
                formatted_df['스코어'] = img_df['score']
                formatted_df['박스'] = img_df.apply(lambda row: f"[{row['bbox_x']}, {row['bbox_y']}, {row['bbox_w']}, {row['bbox_h']}]", axis=1)
                st.dataframe(formatted_df)
            else:
                st.dataframe(img_df)
            
            if st.button("전체 예측 결과(CSV) 보기"):
                st.dataframe(df.head(100))

elif mode == "직접 파일 업로드 (Live Inference)":
    st.subheader("새로운 이미지를 업로드하여 모델 결과를 확인합니다.")
    uploaded_file = st.file_uploader("이미지 업로드 (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        if model is None:
            st.error("학습된 YOLO 모델(`models/best.pt`)을 찾을 수 없습니다. 학습을 먼저 진행해주세요.")
        else:
            col1, col2 = st.columns([2, 1])
            
            # 이미지 열기
            image = Image.open(uploaded_file).convert("RGB")
            
            # 추론
            with st.spinner(f"모델 추론 중... (Threshold: {OPTIMAL_CONF:.4f}, Agnostic NMS, imgsz=960, TTA)"):
                # YOLO는 PIL 이미지를 직접 입력받을 수 있습니다.
                results = model.predict(image, conf=OPTIMAL_CONF, iou=0.45, agnostic_nms=True, imgsz=960, augment=True, device='cpu', verbose=False)
                
            draw = ImageDraw.Draw(image)
            records = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        yolo_cls_id = int(box.cls[0].item())
                        original_dl_idx = class_mapping.get(str(yolo_cls_id), yolo_cls_id)
                        conf = float(box.conf[0].item())
                        
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        x_min, y_min, x_max, y_max = xyxy
                        w = x_max - x_min
                        h = y_max - y_min
                        
                        records.append({
                            "category_id": original_dl_idx,
                            "bbox_x": round(x_min, 2),
                            "bbox_y": round(y_min, 2),
                            "bbox_w": round(w, 2),
                            "bbox_h": round(h, 2),
                            "score": round(conf, 4)
                        })
                        
                        # 시각화 박스 그리기
                        color = "#ff0000" # 업로드 이미지에서는 단일 색상 또는 랜덤
                        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                        text = f"ID:{original_dl_idx} ({conf:.2f})"
                        try:
                            t_bbox = draw.textbbox((x_min, y_min), text)
                            draw.rectangle([t_bbox[0], t_bbox[1], t_bbox[2], t_bbox[3]], fill=color)
                        except AttributeError:
                            pass
                        draw.text((x_min, y_min), text, fill="white")
                        
            with col1:
                st.image(image, use_container_width=True, caption=f"Uploaded: {uploaded_file.name}")
                
            with col2:
                st.write(f"탐지된 객체 수: {len(records)}")
                if records:
                    live_df = pd.DataFrame(records)
                    formatted_live = pd.DataFrame()
                    formatted_live['카테고리 아이디'] = live_df['category_id']
                    formatted_live['카테고리 이름'] = live_df['category_id'].map(dl_idx_to_name)
                    formatted_live['스코어'] = live_df['score']
                    formatted_live['박스'] = live_df.apply(lambda row: f"[{row['bbox_x']}, {row['bbox_y']}, {row['bbox_w']}, {row['bbox_h']}]", axis=1)
                    st.dataframe(formatted_live)
                else:
                    st.info("탐지된 객체가 없습니다. (Confidence Threshold 0.01 기준)")
