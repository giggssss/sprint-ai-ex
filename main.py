import argparse
import os
import sys

import config
from utils.data_prep import YOLODataPrep
from utils.experiment_manager import ExperimentTracker, ReportGenerator
from utils.evaluator import YOLOEvaluator, OODEvaluator
from train import run_training
from infer import run_inference


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Object Detection Master Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="prep",
        choices=["prep", "train", "infer", "eval", "ood", "report", "all"],
        help="Pipeline mode: prep (data check), train (train model), infer (run infer), eval (test set eval), ood (k-NN OOD filtering), report (gen report), all (full pipeline)",
    )
    # 데이터 관련 설정
    parser.add_argument("--data", type=str, default=config.DATA_YAML_DEFAULT, help="Path to data.yaml")
    parser.add_argument("--test-images", type=str, default=config.TEST_IMAGES_DEFAULT, help="Path to test images")
    parser.add_argument("--test-anns", type=str, default=None, help="Path to test annotations JSON")

    # 학습 설정
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Ultralytics YOLO model name or path (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project output path")
    parser.add_argument("--name", type=str, default="train_run", help="Run name")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.15, help="Mixup augmentation probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Flip left-right probability")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning before training")

    # 추론 및 평가 설정
    parser.add_argument("--weights", type=str, default=None, help="Weights file for inference/evaluation")
    parser.add_argument("--source", type=str, default=config.VAL_IMAGES_DEFAULT, help="Inference source path")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default=config.DEVICE, help=f"Computation device (default: {config.DEVICE})")
    parser.add_argument("--output-dir", type=str, default="runs/predict/val_results", help="Inference image output directory")
    parser.add_argument("--save-csv", type=str, default=config.DEFAULT_PREDICTIONS_CSV, help="Predictions output CSV path")

    args = parser.parse_args()

    resolved_weights = config.resolve_weights(args.weights)
    resolved_anns = args.test_anns or config.resolve_test_annotations()

    print("=" * 70)
    print(f"🎯 [YOLO Master Pipeline] 선택된 모드: {args.mode.upper()}")
    print(f"  • 데이터 설정 : {args.data}")
    print(f"  • 모델 가중치 : {resolved_weights}")
    print(f"  • 실행 디바이스: {args.device}")
    print("=" * 70)

    # 1. 데이터 검증 (Prep)
    if args.mode in ["prep", "all"]:
        print("\n[단계 1] 데이터셋 무결성 검증 및 라벨 통계 분석...")
        prep = YOLODataPrep(data_yaml_path=args.data)
        prep.validate_dataset()
        prep.verify_class_distribution()

    # 2. 모델 학습 (Train)
    if args.mode in ["train", "all"]:
        print("\n[단계 2] Ultralytics YOLO 모델 학습 개시...")
        args.model = args.model or resolved_weights
        run_training(args)

    # 3. 단순 이미지 추론 (Infer)
    if args.mode == "infer":
        print("\n[단계 3] 지정 소스 이미지에 대한 추론 실행...")
        args.weights = resolved_weights
        run_inference(args)

    # 4. 테스트셋 정량 평가 (Eval)
    if args.mode in ["eval", "all"]:
        print("\n[단계 3/4] 테스트셋 정량 평가 및 mAP@[0.75:0.95] 산출...")
        evaluator = YOLOEvaluator(weights_path=resolved_weights, data_yaml_path=args.data, device=args.device)
        evaluator.predict_test_set(
            test_images_dir=args.test_images,
            test_annotations_path=resolved_anns,
            output_csv=args.save_csv,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
        evaluator.evaluate_coco(predictions_csv=args.save_csv, annotations_path=resolved_anns)

    # 5. OOD k-NN 거절 스윕 평가 (OOD)
    if args.mode == "ood":
        print("\n[단계 3-B] k-NN Feature Bank 기반 OOD 거절 필터링 스윕...")
        ood_evaluator = OODEvaluator(weights_path=resolved_weights, data_yaml_path=args.data, device=args.device)
        ood_evaluator.run_ood_sweep(
            test_images_dir=args.test_images,
            test_annotations_path=resolved_anns,
            output_csv=args.save_csv,
        )

    # 6. 시각화 실험 보고서 작성 (Report)
    if args.mode in ["report", "all"]:
        print("\n[단계 4] 시각화 그래프 및 Markdown 실험 보고서 자동 생성...")
        tracker = ExperimentTracker()
        reporter = ReportGenerator(tracker)
        report_file = reporter.build_markdown_report()
        print(f"📄 시각화 리포트 갱신 완료: {report_file}")

    print("\n✨ 파이프라인 프로세스가 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    main()