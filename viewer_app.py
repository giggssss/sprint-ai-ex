import os
import streamlit as st
import pandas as pd
import glob
from PIL import Image, ImageDraw, ImageFont
import random

st.set_page_config(page_title="Prediction Viewer", layout="wide")

st.title("🔍 모델 추론 결과 확인 툴 (Test Set Viewer)")

@st.cache_data
def load_data(csv_path):
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)

@st.cache_data
def get_image_list(img_dir):
    paths = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
    # filename to path mapping
    return {os.path.basename(p): p for p in paths}

def generate_colors(num_colors):
    random.seed(42)
    return ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(num_colors)]

CSV_PATH = "predictions.csv"
TEST_IMG_DIR = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_images"

df = load_data(CSV_PATH)
image_map = get_image_list(TEST_IMG_DIR)

if df is None:
    st.error(f"'{CSV_PATH}' 파일을 찾을 수 없습니다. eval_test.py를 먼저 실행해주세요.")
    st.stop()

if not image_map:
    st.error(f"테스트 이미지 디렉터리 '{TEST_IMG_DIR}'에서 이미지를 찾을 수 없습니다.")
    st.stop()

st.sidebar.header("설정")
# image_id나 파일명으로 선택하기
image_ids_in_csv = df['image_id'].unique()
st.sidebar.write(f"CSV에 포함된 이미지(id) 수: {len(image_ids_in_csv)}")

# 사용자가 filename 기반으로 선택할 수 있게 하려면 JSON의 mapping이 필요하지만
# df 에는 image_id만 존재. 
# 여기서는 파일리스트 기준으로 보여주고, df 에서 매칭되는 bbox를 표시할 수도 있음.
# 하지만 image_id가 파일명과 어떻게 매핑되었는지 UI에서 바로 알기 어려우므로,
# 단순히 df 내의 image_id 하나를 선택하면 해당 이미지를 찾도록 하거나(복잡),
# 사용자가 직접 이미지를 고르게 함.
# 만약 eval_test.py에서 image_id가 파일명 해시 또는 추출된 ID라면 
# 역산이 불가능할 수 있음. 
# 잠시만요, eval_test.py에서 csv의 image_id는 JSON의 고유 id를 사용했음.
# 이미지의 파일명과 매핑하려면 df를 filename과 연결해야하는데... 
# 이를 위해 eval_test.py에서 filename도 임시로 csv에 같이 저장했으면 좋겠지만 포맷 제약이 있음.

# 해결: CSV 파일과 파일 시스템을 대조하기 어렵다면, 
# 여기서는 폴더 내 모든 이미지를 리스트하고 그 파일명을 해싱/매핑해서 CSV를 필터링
st.sidebar.info("CSV 포맷 제한으로 인해, 현재는 폴더 내 이미지를 직접 선택하면 해당 이미지에 매칭되는 결과를 보여줍니다.")
selected_filename = st.sidebar.selectbox("테스트 이미지 선택", list(image_map.keys()))

# JSON parsing logic in viewer to find image_id for selected filename
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
    # fallback
    target_image_id = abs(hash(selected_filename)) % (10**8)

# Filter df
img_df = df[df['image_id'] == target_image_id]

col1, col2 = st.columns([2, 1])

with col1:
    img_path = image_map[selected_filename]
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    unique_categories = df['category_id'].unique()
    colors = generate_colors(len(unique_categories) + 100) # 넉넉히
    cat_to_color = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}
    
    if not img_df.empty:
        for _, row in img_df.iterrows():
            x, y, w, h = row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']
            cat = row['category_id']
            score = row['score']
            color = cat_to_color.get(cat, "#ff0000")
            
            # draw box
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            # draw text
            text = f"ID:{cat} ({score:.2f})"
            # 배경 그리기 (가독성)
            # Pillow의 textbbox를 사용하거나 대략 계산
            try:
                t_bbox = draw.textbbox((x, y), text)
                draw.rectangle([t_bbox[0], t_bbox[1], t_bbox[2], t_bbox[3]], fill=color)
            except AttributeError:
                pass
            draw.text((x, y), text, fill="white")
            
    st.image(image, use_column_width=True, caption=f"Selected: {selected_filename} (Image ID: {target_image_id})")

with col2:
    st.subheader("예측 데이터 (CSV)")
    st.write(f"탐지된 객체 수: {len(img_df)}")
    st.dataframe(img_df)
    
    if st.button("전체 예측 결과(CSV) 보기"):
        st.dataframe(df.head(100)) # 너무 크면 일부만
