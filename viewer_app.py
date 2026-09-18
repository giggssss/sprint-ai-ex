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
st.caption("🎯 **공식 평가 기준**: `mAP@[0.75:0.95]` (Strict IoU)")

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
def get_dl_idx_to_name(data_yaml_path, mapping_path, test_anns_path=None):
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
                
    if test_anns_path and os.path.exists(test_anns_path):
        try:
            with open(test_anns_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
                for cat in coco.get("categories", []):
                    cid = cat.get("id")
                    cname = cat.get("name")
                    if cid is not None and cname:
                        dl_idx_to_name[cid] = cname
        except Exception:
            pass
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
TEST_ANNS = "/Volumes/Macintosh SUB/Dataset/sprint_ai_project1_data/test_annotations_v260710.json"
if not os.path.exists(TEST_ANNS):
    TEST_ANNS = "test_coco.json"

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
dl_idx_to_name = get_dl_idx_to_name(DATA_YAML_PATH, MAPPING_PATH, TEST_ANNS)

@st.cache_data
def load_test_ground_truths(test_annotations_path):
    if not os.path.exists(test_annotations_path):
        return {}
    try:
        with open(test_annotations_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        img_to_gts = {}
        for ann in coco.get("annotations", []):
            iid = ann.get("image_id")
            if iid not in img_to_gts:
                img_to_gts[iid] = []
            img_to_gts[iid].append({
                "category_id": ann.get("category_id"),
                "bbox": ann.get("bbox")
            })
        return img_to_gts
    except Exception:
        return {}

all_ground_truths = load_test_ground_truths(TEST_ANNS)

@st.cache_data
def get_train_reference_crops():
    data_yaml = '/Volumes/Macintosh SUB/Dataset/yolo_data/data.yaml'
    mapping_path = 'class_mapping.json'
    if not os.path.exists(data_yaml) or not os.path.exists(mapping_path):
        return {}
        
    with open(data_yaml) as f:
        names = yaml.safe_load(f).get('names', [])
    with open(mapping_path) as f:
        class_mapping_data = json.load(f)
        
    yolo_to_dl = {int(k): int(v) for k, v in class_mapping_data.items()}
    
    search_dirs = [
        ('/Volumes/Macintosh SUB/Dataset/yolo_data_v2/train/labels', '/Volumes/Macintosh SUB/Dataset/yolo_data_v2/train/images'),
        ('/Volumes/Macintosh SUB/Dataset/yolo_data_v2/val/labels', '/Volumes/Macintosh SUB/Dataset/yolo_data_v2/val/images'),
        ('/Volumes/Macintosh SUB/Dataset/yolo_data/train/labels', '/Volumes/Macintosh SUB/Dataset/yolo_data/train/images')
    ]
    
    yolo_to_ref_info = {}
    for lbl_dir, img_dir in search_dirs:
        if not os.path.exists(lbl_dir) or not os.path.exists(img_dir):
            continue
        for lf in glob.glob(os.path.join(lbl_dir, '*.txt')):
            base = os.path.splitext(os.path.basename(lf))[0]
            ip = os.path.join(img_dir, base + '.png')
            if not os.path.exists(ip):
                ip = os.path.join(img_dir, base + '.jpg')
            if not os.path.exists(ip):
                continue
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    cls_id = int(parts[0])
                    if cls_id not in yolo_to_ref_info:
                        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        yolo_to_ref_info[cls_id] = (ip, xc, yc, bw, bh)
            if len(yolo_to_ref_info) == len(names):
                break
                
    dl_to_crop = {}
    for yolo_cls, (ip, xc, yc, bw, bh) in yolo_to_ref_info.items():
        dl_idx = yolo_to_dl.get(yolo_cls, yolo_cls)
        try:
            with Image.open(ip) as img:
                img_rgb = img.convert('RGB')
                W, H = img_rgb.size
                x_c, y_c = xc * W, yc * H
                w_box, h_box = bw * W, bh * H
                pad_x = int(w_box * 0.25)
                pad_y = int(h_box * 0.25)
                x1 = max(0, int(x_c - w_box / 2 - pad_x))
                y1 = max(0, int(y_c - h_box / 2 - pad_y))
                x2 = min(W, int(x_c + w_box / 2 + pad_x))
                y2 = min(H, int(y_c + h_box / 2 + pad_y))
                crop = img_rgb.crop((x1, y1, x2, y2))
                dl_to_crop[dl_idx] = crop
        except Exception:
            pass
            
    return dl_to_crop

train_ref_crops = get_train_reference_crops()

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def render_detailed_object_comparison(img_path, img_df, gts, dl_idx_to_name, train_ref_crops, iou_thresh=0.5):
    if not os.path.exists(img_path):
        return
        
    try:
        full_image = Image.open(img_path).convert("RGB")
    except Exception:
        return
        
    img_W, img_H = full_image.size
    
    pred_items = []
    if img_df is not None and not img_df.empty:
        for _, row in img_df.iterrows():
            if 'bbox_x' in row and pd.notna(row['bbox_x']):
                p_box = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
            elif 'bbox' in row and pd.notna(row['bbox']):
                import ast
                b = ast.literal_eval(row['bbox']) if isinstance(row['bbox'], str) else row['bbox']
                p_box = [b[0], b[1], b[2], b[3]]
            else:
                continue
            pred_items.append({
                'box': p_box,
                'score': float(row.get('score', 1.0)),
                'category_id': int(row.get('category_id'))
            })
            
    gt_items = []
    if gts:
        for g in gts:
            gt_items.append({
                'box': [float(v) for v in g['bbox']],
                'category_id': int(g['category_id'])
            })
            
    sorted_preds = sorted(pred_items, key=lambda x: x['score'], reverse=True)
    matched_results = []
    used_gts = set()
    
    for p in sorted_preds:
        best_iou = 0.0
        best_gt_idx = None
        for g_idx, g in enumerate(gt_items):
            if g_idx in used_gts:
                continue
            iou = compute_iou(p['box'], g['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
                
        if best_gt_idx is not None and best_iou >= iou_thresh:
            used_gts.add(best_gt_idx)
            matched_gt = gt_items[best_gt_idx]
            if p['category_id'] == matched_gt['category_id']:
                res_type = "TP (Correct)"
            else:
                res_type = "FP (Misclassified)"
            matched_results.append({
                'type': res_type,
                'score': p['score'],
                'iou': best_iou,
                'pred': p,
                'gt': matched_gt
            })
        else:
            matched_results.append({
                'type': "FP (Background / Low IoU)",
                'score': p['score'],
                'iou': best_iou,
                'pred': p,
                'gt': None
            })
            
    for g_idx, g in enumerate(gt_items):
        if g_idx not in used_gts:
            matched_results.append({
                'type': "FN (Missed Ground Truth)",
                'score': None,
                'iou': 0.0,
                'pred': None,
                'gt': g
            })
            
    if not matched_results:
        st.info("비교 분석할 객체(예측 또는 정답)가 없습니다.")
        return

    st.markdown("---")
    st.subheader("🔬 객체별 정밀 대조 및 분석 (Prediction vs Ground Truth vs Train Reference)")
    st.caption("각 검출 객체별로 예측 위치(빨간색), 실제 정답 위치(초록색), 학습 데이터 기준 참조 이미지를 1:1로 비교합니다.")
    
    tp_cnt = sum(1 for r in matched_results if "TP" in r["type"])
    fp_cnt = sum(1 for r in matched_results if "FP" in r["type"])
    fn_cnt = sum(1 for r in matched_results if "FN" in r["type"])
    
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        obj_filter = st.radio("표시 필터", ["전체 객체 (All)", "오류만 (FP / FN)", "정상 탐지만 (TP)"], horizontal=True, key=f"filter_{os.path.basename(img_path)}")
    with filter_col2:
        st.markdown(
            f"총 **{len(matched_results)}**개 객체 (<span style='color:#2e7d32; font-weight:bold;'>TP: {tp_cnt}</span> | "
            f"<span style='color:#d32f2f; font-weight:bold;'>FP: {fp_cnt}</span> | "
            f"<span style='color:#d32f2f; font-weight:bold;'>FN: {fn_cnt}</span>)",
            unsafe_allow_html=True
        )

    displayed_count = 0
    for idx, item in enumerate(matched_results, start=1):
        res_type = item['type']
        if obj_filter == "오류만 (FP / FN)" and "TP" in res_type:
            continue
        if obj_filter == "정상 탐지만 (TP)" and "TP" not in res_type:
            continue
            
        displayed_count += 1
        score = item['score']
        score_str = f"{score:.3f}" if score is not None else "N/A"
        iou_val = item['iou']
        
        p = item['pred']
        g = item['gt']
        
        pred_cat = p['category_id'] if p else None
        gt_cat = g['category_id'] if g else None
        
        pred_name = dl_idx_to_name.get(pred_cat, f"ID:{pred_cat}") if pred_cat is not None else "없음 (미탐지)"
        gt_name = dl_idx_to_name.get(gt_cat, f"ID:{gt_cat}") if gt_cat is not None else "없음 (정답 라벨 없음)"
        
        if p and g:
            u_x1 = min(p['box'][0], g['box'][0])
            u_y1 = min(p['box'][1], g['box'][1])
            u_x2 = max(p['box'][0] + p['box'][2], g['box'][0] + g['box'][2])
            u_y2 = max(p['box'][1] + p['box'][3], g['box'][1] + g['box'][3])
        elif p:
            u_x1, u_y1, u_x2, u_y2 = p['box'][0], p['box'][1], p['box'][0] + p['box'][2], p['box'][1] + p['box'][3]
        else:
            u_x1, u_y1, u_x2, u_y2 = g['box'][0], g['box'][1], g['box'][0] + g['box'][2], g['box'][1] + g['box'][3]
            
        u_w = max(10, u_x2 - u_x1)
        u_h = max(10, u_y2 - u_y1)
        pad_x = int(u_w * 0.25)
        pad_y = int(u_h * 0.25)
        c_x1 = max(0, int(u_x1 - pad_x))
        c_y1 = max(0, int(u_y1 - pad_y))
        c_x2 = min(img_W, int(u_x2 + pad_x))
        c_y2 = min(img_H, int(u_y2 + pad_y))
        
        crop_base = full_image.crop((c_x1, c_y1, c_x2, c_y2))
        
        # 1. 모델 예측 (RED)
        crop_pred = crop_base.copy()
        if p:
            draw_p = ImageDraw.Draw(crop_pred)
            px1 = p['box'][0] - c_x1
            py1 = p['box'][1] - c_y1
            px2 = px1 + p['box'][2]
            py2 = py1 + p['box'][3]
            draw_p.rectangle([px1, py1, px2, py2], outline="#ff0000", width=4)
            
        # 2. 정답 (GREEN)
        crop_gt = crop_base.copy()
        if g:
            draw_g = ImageDraw.Draw(crop_gt)
            gx1 = g['box'][0] - c_x1
            gy1 = g['box'][1] - c_y1
            gx2 = gx1 + g['box'][2]
            gy2 = gy1 + g['box'][3]
            draw_g.rectangle([gx1, gy1, gx2, gy2], outline="#00aa00", width=4)
            
        # 3. Train Reference Crop
        ref_cat = pred_cat if pred_cat is not None else gt_cat
        crop_ref = train_ref_crops.get(ref_cat)
        ref_name = dl_idx_to_name.get(ref_cat, f"ID:{ref_cat}")
        
        is_tp = "TP" in res_type
        type_color = "#2e7d32" if is_tp else "#d32f2f" # TP는 초록색, False case(FP/FN)는 빨간색
        
        st.markdown(
            f"#### <span style='color: {type_color}; font-weight: bold;'>[{idx}] 유형: {res_type}</span> | Score: **{score_str}**",
            unsafe_allow_html=True
        )
        if is_tp:
            st.markdown(
                f"**Pred Class:** <span style='color: #2e7d32; font-weight: 600;'>{pred_name}</span> | **GT Class:** <span style='color: #2e7d32; font-weight: 600;'>{gt_name}</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"**Pred Class:** <span style='color: #d32f2f; font-weight: 600;'>{pred_name}</span> | **GT Class:** <span style='color: #2e7d32; font-weight: 600;'>{gt_name}</span>",
                unsafe_allow_html=True
            )
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("**1. 모델 예측 (Test Image)**")
            st.image(crop_pred, use_container_width=True)
            pred_caption_color = "#2e7d32" if is_tp else "#d32f2f"
            st.markdown(f"<p style='color: {pred_caption_color}; font-size: 0.85rem; margin-top: -8px;'>예측: {pred_name}</p>", unsafe_allow_html=True)
            
        with col_c2:
            st.markdown("**2. 정답 (Test Image)**")
            st.image(crop_gt, use_container_width=True)
            st.markdown(f"<p style='color: #2e7d32; font-size: 0.85rem; margin-top: -8px;'>정답: {gt_name}</p>", unsafe_allow_html=True)
            
        with col_c3:
            st.markdown("**3. 예측 클래스의 실제 생김새 (Train Reference)**")
            if crop_ref is not None:
                st.image(crop_ref, use_container_width=True)
                st.markdown(f"<p style='color: #555555; font-size: 0.85rem; margin-top: -8px;'>참고: {ref_name}</p>", unsafe_allow_html=True)
            else:
                st.info("Train Reference 없음 (미학습 클래스 또는 OOD)")
        st.divider()

    if displayed_count == 0:
        st.info("선택한 필터 조건에 부합하는 객체가 없습니다.")

@st.cache_data
def get_filename_to_image_id_mapping(test_annotations_path):
    filename_to_image_id = {}
    if os.path.isfile(test_annotations_path):
        with open(test_annotations_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for img in data.get("images", []):
                    fn = img.get("file_name")
                    img_id = img.get("id")
                    if fn and img_id is not None:
                        filename_to_image_id[fn] = img_id
                        filename_to_image_id[f"{img_id}.png"] = img_id
            except Exception:
                pass
    else:
        json_files = glob.glob(os.path.join(test_annotations_path, "**", "*.json"), recursive=True)
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    for img in data.get("images", []):
                        fn = img.get("file_name")
                        img_id = img.get("id")
                        if fn and img_id is not None:
                            filename_to_image_id[fn] = img_id
                            filename_to_image_id[f"{img_id}.png"] = img_id
                except Exception:
                    pass
    return filename_to_image_id

mapping = get_filename_to_image_id_mapping(TEST_ANNS)
image_id_to_filename = {v: k for k, v in mapping.items() if not k.endswith(".png") or k in image_map}

def render_image_with_boxes(img_path, boxes_df, caption, cat_to_color, dl_idx_to_name, gts=None, show_pred=True, show_gt=True):
    if not os.path.exists(img_path):
        st.error(f"이미지를 찾을 수 없습니다: {img_path}")
        return
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # 1. Draw GT boxes in Green (#00c853)
    if show_gt and gts:
        for g in gts:
            gx, gy, gw, gh = [float(v) for v in g['bbox']]
            cat = g['category_id']
            cat_name = dl_idx_to_name.get(cat, "")
            name_str = f" ({cat_name})" if cat_name else ""
            gt_color = "#00c853"
            draw.rectangle([gx, gy, gx + gw, gy + gh], outline=gt_color, width=4)
            gt_text = f"[GT] ID:{cat}{name_str}"
            try:
                t_bbox = draw.textbbox((gx, max(0, gy - 20)), gt_text)
                draw.rectangle([t_bbox[0], t_bbox[1], t_bbox[2], t_bbox[3]], fill=gt_color)
                draw.text((gx, max(0, gy - 20)), gt_text, fill="white")
            except AttributeError:
                draw.text((gx, max(0, gy - 20)), gt_text, fill=gt_color)
                
    # 2. Draw Prediction boxes in Red (#ff1744)
    if show_pred and boxes_df is not None and not boxes_df.empty:
        for _, row in boxes_df.iterrows():
            if 'bbox_x' in row and pd.notna(row['bbox_x']):
                x, y, w, h = row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']
            elif 'bbox' in row and pd.notna(row['bbox']):
                import ast
                b = ast.literal_eval(row['bbox']) if isinstance(row['bbox'], str) else row['bbox']
                x, y, w, h = b[0], b[1], b[2], b[3]
            else:
                continue
            cat = row['category_id']
            score = row.get('score', 1.0)
            pred_color = "#ff1744"
            draw.rectangle([x, y, x + w, y + h], outline=pred_color, width=3)
            cat_name = dl_idx_to_name.get(cat, "")
            name_str = f" ({cat_name})" if cat_name else ""
            text = f"[Pred] ID:{cat}{name_str} ({score:.2f})"
            try:
                t_bbox = draw.textbbox((x, y), text)
                draw.rectangle([t_bbox[0], t_bbox[1], t_bbox[2], t_bbox[3]], fill=pred_color)
                draw.text((x, y), text, fill="white")
            except AttributeError:
                draw.text((x, y), text, fill=pred_color)
                
    st.image(image, use_container_width=True, caption=caption)

unique_categories = df['category_id'].unique() if df is not None and not df.empty else []
colors = generate_colors(len(unique_categories) + 100)
cat_to_color = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}

# 사이드바 모드 선택
st.sidebar.header("설정")
mode = st.sidebar.radio("확인 방식 선택", [
    "기존 테스트셋 조회 (CSV)", 
    "🚨 Edge Case / 이상치 분석",
    "직접 파일 업로드 (Live Inference)"
])

if mode == "기존 테스트셋 조회 (CSV)":
    if df is None or not image_map:
        st.warning("테스트셋 결과(CSV)나 이미지 디렉토리를 찾을 수 없습니다. '직접 파일 업로드' 기능을 사용해 보세요.")
    else:
        st.sidebar.info("CSV 포맷 제한으로 인해, 현재는 폴더 내 이미지를 직접 선택하면 해당 이미지에 매칭되는 결과를 보여줍니다.")
        selected_filename = st.sidebar.selectbox("테스트 이미지 선택", list(image_map.keys()))
        
        target_image_id = mapping.get(selected_filename)
        if target_image_id is None:
            try:
                target_image_id = int(os.path.splitext(selected_filename)[0])
            except ValueError:
                import hashlib
                target_image_id = int(hashlib.md5(selected_filename.encode()).hexdigest(), 16) % (10**8)
            
        img_df = df[df['image_id'] == target_image_id]
        gts_for_img = all_ground_truths.get(target_image_id, [])
        
        col1, col2 = st.columns([2, 1])
        with col1:
            img_path = image_map[selected_filename]
            
            # 박스 표시 토글 (기본: 둘 다 표시)
            box_col1, box_col2 = st.columns(2)
            with box_col1:
                show_pred = st.checkbox("🟥 모델 예측 (Pred) 박스", value=True, key=f"csv_p_{target_image_id}")
            with box_col2:
                show_gt = st.checkbox("🟩 실제 정답 (GT) 박스", value=True, key=f"csv_g_{target_image_id}")
                
            render_image_with_boxes(
                img_path, img_df, f"Selected: {selected_filename} (Image ID: {target_image_id})", 
                cat_to_color, dl_idx_to_name, 
                gts=gts_for_img, show_pred=show_pred, show_gt=show_gt
            )

        with col2:
            st.markdown(
                f"### 📊 데이터 요약\n"
                f"- 🟥 **예측(Pred) 객체 수**: `{len(img_df)}개`\n"
                f"- 🟩 **정답(GT) 객체 수**: `{len(gts_for_img)}개`"
            )
            tab_pred, tab_gt = st.tabs([f"🎯 예측 데이터 ({len(img_df)})", f"✅ 정답 데이터 ({len(gts_for_img)})"])
            with tab_pred:
                if not img_df.empty:
                    formatted_df = pd.DataFrame()
                    formatted_df['카테고리 ID'] = img_df['category_id']
                    formatted_df['카테고리 이름'] = img_df['category_id'].map(dl_idx_to_name)
                    formatted_df['스코어'] = img_df['score']
                    formatted_df['박스'] = img_df.apply(lambda row: f"[{row['bbox_x']}, {row['bbox_y']}, {row['bbox_w']}, {row['bbox_h']}]", axis=1)
                    st.dataframe(formatted_df)
                else:
                    st.info("이 이미지에서 탐지된 객체가 없습니다 (0 Detection).")
            with tab_gt:
                if gts_for_img:
                    gt_df = pd.DataFrame()
                    gt_df['카테고리 ID'] = [g['category_id'] for g in gts_for_img]
                    gt_df['카테고리 이름'] = [dl_idx_to_name.get(g['category_id'], '') for g in gts_for_img]
                    gt_df['정답 박스'] = [f"[{round(g['bbox'][0], 1)}, {round(g['bbox'][1], 1)}, {round(g['bbox'][2], 1)}, {round(g['bbox'][3], 1)}]" for g in gts_for_img]
                    st.dataframe(gt_df)
                else:
                    st.info("정답 라벨이 없습니다 (순수 음성/배경).")
            
            if st.button("전체 예측 결과(CSV) 미리보기"):
                st.dataframe(df.head(100))

        # 각 이미지별 객체 3단 상세 대조 (예측 vs 정답 vs Train Reference)
        render_detailed_object_comparison(img_path, img_df, gts_for_img, dl_idx_to_name, train_ref_crops)

elif mode == "🚨 Edge Case / 이상치 분석":
    st.subheader("🚨 Edge Case 및 데이터 이상치 분석 센터")
    
    edge_tab1, edge_tab2, edge_tab3 = st.tabs([
        "1️⃣ 미탐지(0 Detection) 테스트 이미지", 
        "2️⃣ 낮은 신뢰도 예측 (Score < 0.50)", 
        "3️⃣ 데이터셋 라벨 불일치 리포트"
    ])
    
    with edge_tab1:
        st.markdown("#### 🔍 미탐지 (Detections = 0) 이미지")
        if df is not None and os.path.exists(TEST_ANNS):
            with open(TEST_ANNS, "r", encoding="utf-8") as f:
                anns_data = json.load(f)
            all_test_imgs = {img['id']: img['file_name'] for img in anns_data.get('images', [])}
            pred_ids = set(df['image_id'].unique())
            no_det_ids = sorted(list(set(all_test_imgs.keys()) - pred_ids))
            
            st.info(f"전체 {len(all_test_imgs)}개 테스트 이미지 중 탐지 박스가 0개인 이미지는 총 **{len(no_det_ids)}장**입니다.")
            
            if no_det_ids:
                selected_no_det = st.selectbox(
                    "미탐지 대상 이미지 선택",
                    no_det_ids,
                    format_func=lambda x: f"Image ID: {x} | {all_test_imgs.get(x, '')}"
                )
                
                # Check ground truth in annotation
                gt_anns = [a for a in anns_data.get('annotations', []) if a['image_id'] == selected_no_det]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fn = all_test_imgs.get(selected_no_det, "")
                    img_p = image_map.get(fn, "")
                    if img_p and os.path.exists(img_p):
                        image = Image.open(img_p).convert("RGB")
                        st.image(image, use_container_width=True, caption=f"No Detection: {fn} (ID: {selected_no_det})")
                    else:
                        st.warning(f"이미지 파일을 찾을 수 없습니다: {fn}")
                        
                with col2:
                    st.markdown("##### 📋 분석 결과")
                    st.write(f"- **예측 박스 수**: `0개`")
                    st.write(f"- **정답(Ground Truth) 라벨 수**: `{len(gt_anns)}개`")
                    if len(gt_anns) == 0:
                        st.success(
                            "✅ **정상 예측 (True Negative)**\n\n"
                            "이 이미지는 공식 어노테이션 파일에서도 정답 라벨이 0개인 순수 배경/음성(Negative) 또는 OOD 이미지입니다.\n\n"
                            "모델이 오탐(False Positive) 없이 0개를 예측하여 평가 점수(Precision/mAP)에 매우 긍정적입니다."
                        )
                    else:
                        st.error(
                            f"❌ **미탐지 (False Negative)**: 정답 객체가 {len(gt_anns)}개 존재하나 모델이 찾지 못했습니다."
                        )
                        st.dataframe(pd.DataFrame(gt_anns))
        else:
            st.warning("예측 결과(CSV) 또는 어노테이션 파일을 확인할 수 없습니다.")

    with edge_tab2:
        st.markdown("#### ⚠️ 낮은 신뢰도(Score) 예측 이미지")
        conf_cutoff = st.slider("신뢰도 상한선 (Cutoff Threshold)", 0.1, 0.7, 0.5, 0.05)
        
        if df is not None:
            low_score_df = df[df['score'] <= conf_cutoff]
            low_score_imgs = low_score_df['image_id'].unique()
            st.write(f"신뢰도 `{conf_cutoff:.2f}` 이하의 객체가 포함된 이미지: 총 **{len(low_score_imgs)}장** (전체 {len(low_score_df)}개 박스)")
            
            if len(low_score_imgs) > 0:
                selected_low_img_id = st.selectbox(
                    "낮은 신뢰도 이미지 선택",
                    low_score_imgs,
                    format_func=lambda x: f"Image ID: {x} | {image_id_to_filename.get(x, f'{x}.png')}"
                )
                
                target_boxes = df[df['image_id'] == selected_low_img_id]
                fn = image_id_to_filename.get(selected_low_img_id, f"{selected_low_img_id}.png")
                img_p = image_map.get(fn, "")
                
                low_gts = all_ground_truths.get(selected_low_img_id, [])
                col1, col2 = st.columns([2, 1])
                with col1:
                    if img_p and os.path.exists(img_p):
                        b_c1, b_c2 = st.columns(2)
                        with b_c1:
                            s_pred = st.checkbox("🟥 모델 예측 (Pred) 박스", value=True, key=f"low_p_{selected_low_img_id}")
                        with b_c2:
                            s_gt = st.checkbox("🟩 실제 정답 (GT) 박스", value=True, key=f"low_g_{selected_low_img_id}")
                        render_image_with_boxes(
                            img_p, target_boxes, f"ID: {selected_low_img_id} ({fn})", 
                            cat_to_color, dl_idx_to_name, 
                            gts=low_gts, show_pred=s_pred, show_gt=s_gt
                        )
                    else:
                        st.warning(f"이미지를 찾을 수 없습니다: {fn}")
                with col2:
                    st.markdown(
                        f"### 📊 데이터 요약\n"
                        f"- 🟥 **예측(Pred) 객체 수**: `{len(target_boxes)}개`\n"
                        f"- 🟩 **정답(GT) 객체 수**: `{len(low_gts)}개`"
                    )
                    f_df = pd.DataFrame()
                    f_df['카테고리 ID'] = target_boxes['category_id']
                    f_df['이름'] = target_boxes['category_id'].map(dl_idx_to_name)
                    f_df['스코어'] = target_boxes['score']
                    f_df['낮은 신뢰도 여부'] = target_boxes['score'] <= conf_cutoff
                    st.dataframe(f_df)
                
                # 3단 상세 대조 렌더링
                if img_p and os.path.exists(img_p):
                    render_detailed_object_comparison(img_p, target_boxes, low_gts, dl_idx_to_name, train_ref_crops)
        else:
            st.warning("예측 결과(CSV)가 없습니다.")

    with edge_tab3:
        st.markdown("#### 📑 데이터셋 라벨 불일치(Inconsistency) 분석 리포트")
        rep_path = "experiments/reports/inconsistency_report.csv"
        matched_path = "experiments/reports/matched_inconsistent_images.csv"
        
        if os.path.exists(rep_path) and os.path.exists(matched_path):
            st.info("이 데이터는 초기 데이터셋 검증 당시 라벨 파일과 폴더 구조 간 불일치(Obj item 누락 등)가 발생했던 내역입니다.")
            rep_df = pd.read_csv(rep_path)
            matched_df = pd.read_csv(matched_path)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("불일치 발생 폴더 수", len(rep_df))
                st.dataframe(rep_df)
            with col_b:
                st.metric("불일치 대상 이미지 수", len(matched_df))
                st.dataframe(matched_df)
        else:
            st.warning("`experiments/reports/` 내에 불일치 리포트 파일이 존재하지 않습니다.")

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
