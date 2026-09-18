"""
중앙 집중식 프로젝트 설정 모듈 (Configuration)
데이터 경로, 모델 가중치 기본 경로 및 파라미터 기본값을 관리합니다.
"""
import os
import torch

# 1. 기본 디렉터리 및 데이터셋 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DATASET_ROOT = "/Volumes/Macintosh SUB/Dataset"

# YOLOv2 데이터셋 (Stratified Split 적용)
DATA_YAML_DEFAULT = os.path.join(EXTERNAL_DATASET_ROOT, "yolo_data_v2", "data.yaml")
TRAIN_IMAGES_DEFAULT = os.path.join(EXTERNAL_DATASET_ROOT, "yolo_data_v2", "train", "images")
TRAIN_LABELS_DEFAULT = os.path.join(EXTERNAL_DATASET_ROOT, "yolo_data_v2", "train", "labels")
VAL_IMAGES_DEFAULT = os.path.join(EXTERNAL_DATASET_ROOT, "yolo_data_v2", "val", "images")

# 원본 테스트 데이터
TEST_IMAGES_DEFAULT = os.path.join(EXTERNAL_DATASET_ROOT, "sprint_ai_project1_data", "test_images")
TEST_ANNS_PRIMARY = os.path.join(EXTERNAL_DATASET_ROOT, "sprint_ai_project1_data", "test_annotations_v260710.json")
TEST_ANNS_FALLBACK = os.path.join(BASE_DIR, "test_coco.json")

# 2. 로컬 프로젝트 산출물 경로
CLASS_MAPPING_FILE = os.path.join(BASE_DIR, "class_mapping.json")
OPTIMAL_CONF_FILE = os.path.join(BASE_DIR, "optimal_conf.txt")
DEFAULT_PREDICTIONS_CSV = os.path.join(BASE_DIR, "predictions.csv")

# 3. 모델 가중치 기본 경로
DEFAULT_TRAINED_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "runs", "detect", "train_yolo11s_v2_dataset", "weights", "best.pt")
FALLBACK_WEIGHTS = os.path.join(BASE_DIR, "models", "best.pt")

# 4. 연산 디바이스 자동 감지
def get_device() -> str:
    """사용 가능한 최적의 PyTorch 디바이스 반환 ('mps', 'cuda', 'cpu')"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

# 5. 경로 유효성 검증 헬퍼
def resolve_test_annotations() -> str:
    """테스트 어노테이션 파일의 유효 경로를 반환합니다 (외장 SSD 우선, 없으면 로컬 폴백)."""
    if os.path.exists(TEST_ANNS_PRIMARY):
        return TEST_ANNS_PRIMARY
    if os.path.exists(TEST_ANNS_FALLBACK):
        return TEST_ANNS_FALLBACK
    return TEST_ANNS_PRIMARY

def resolve_weights(preferred_path: str = None) -> str:
    """사용 가능한 최적의 가중치 경로를 반환합니다."""
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path
    if os.path.exists(DEFAULT_TRAINED_WEIGHTS):
        return DEFAULT_TRAINED_WEIGHTS
    if os.path.exists(FALLBACK_WEIGHTS):
        return FALLBACK_WEIGHTS
    return preferred_path or FALLBACK_WEIGHTS

def get_optimal_confidence(default_conf: float = 0.25) -> float:
    """저장된 최적 Confidence 임계값을 로드하거나 기본값을 반환합니다."""
    if os.path.exists(OPTIMAL_CONF_FILE):
        try:
            with open(OPTIMAL_CONF_FILE, "r", encoding="utf-8") as f:
                val = float(f.read().strip())
                return val
        except Exception:
            pass
    return default_conf
