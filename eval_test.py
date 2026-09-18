"""
테스트셋 평가 및 제출용 CSV 생성 스크립트 (하위 호환 래퍼)
통합된 utils.evaluator.YOLOEvaluator 모듈을 호출합니다.
"""
import os
import config
from utils.evaluator import YOLOEvaluator


def generate_predictions_csv(
    weights_path: str = None,
    test_images_dir: str = config.TEST_IMAGES_DEFAULT,
    test_annotations_dir: str = None,
    class_mapping_path: str = config.CLASS_MAPPING_FILE,
    output_csv: str = config.DEFAULT_PREDICTIONS_CSV,
    conf_threshold: float = None,
):
    """지정된 가중치를 사용하여 테스트 이미지 예측을 수행하고 CSV로 저장합니다."""
    evaluator = YOLOEvaluator(weights_path=weights_path, class_mapping_path=class_mapping_path)
    return evaluator.predict_test_set(
        test_images_dir=test_images_dir,
        test_annotations_path=test_annotations_dir,
        output_csv=output_csv,
        conf_threshold=conf_threshold,
    )


def evaluate_predictions_coco(
    csv_path: str = config.DEFAULT_PREDICTIONS_CSV,
    annotations_path: str = None,
):
    """pycocotools를 사용하여 생성된 CSV 파일의 mAP@[0.75:0.95]를 산출합니다."""
    evaluator = YOLOEvaluator()
    metrics = evaluator.evaluate_coco(predictions_csv=csv_path, annotations_path=annotations_path)
    return metrics.get("mAP75-95", 0.0)


def check_map(weights_path: str = None):
    """YOLO validation 내장 기능을 통해 mAP 지표를 확인합니다."""
    from ultralytics import YOLO

    w = config.resolve_weights(weights_path)
    model = YOLO(w)
    metrics = model.val(data=config.DATA_YAML_DEFAULT, split="val")
    mAP_75_95 = 0.0
    if hasattr(metrics.box, "all_ap") and metrics.box.all_ap is not None:
        all_ap = metrics.box.all_ap
        if len(all_ap.shape) >= 2 and all_ap.shape[1] > 5:
            mAP_75_95 = float(all_ap[:, 5:].mean())
    print(f"🔥 mAP@[0.75:0.95]: {mAP_75_95:.4f}")
    return mAP_75_95


if __name__ == "__main__":
    weights = config.resolve_weights()
    anns = config.resolve_test_annotations()
    output_csv = config.DEFAULT_PREDICTIONS_CSV

    print(f"🚀 Running Evaluation with weights: {weights}")
    check_map(weights)
    generate_predictions_csv(weights_path=weights, test_annotations_dir=anns, output_csv=output_csv)
    evaluate_predictions_coco(csv_path=output_csv, annotations_path=anns)
