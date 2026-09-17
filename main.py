import argparse
import sys
from utils.data_prep import YOLODataPrep
from utils.experiment_manager import ExperimentTracker, ReportGenerator
from train import run_training
from infer import run_inference


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Object Detection Master Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="prep",
        choices=["prep", "train", "infer", "report", "all"],
        help="Pipeline mode: prep (data check), train (train model), infer (run infer), report (gen report), all (full pipeline)",
    )
    # 데이터 관련 설정
    parser.add_argument("--data", type=str, default="/Volumes/Macintosh SUB/Dataset/yolo_data_v2/data.yaml", help="Path to data.yaml")

    # 학습 설정
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Ultralytics YOLO model name or path (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (default: 10)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project output path")
    parser.add_argument("--name", type=str, default="train_run", help="Run name")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.15, help="Mixup augmentation probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Flip left-right probability")

    # 추론 설정
    parser.add_argument("--weights", type=str, default="runs/detect/train_run/weights/best.pt", help="Weights file for inference")
    parser.add_argument("--source", type=str, default="/Volumes/Macintosh SUB/Dataset/yolo_data/val/images", help="Inference source path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--device", type=str, default="mps", help="Computation device (mps, cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="runs/predict/val_results", help="Inference image output directory")
    parser.add_argument("--save-csv", type=str, default="runs/predict/predictions.csv", help="Inference output CSV path")

    args = parser.parse_args()

    print("=" * 70)
    print(f"🎯 [YOLO Master Pipeline 실행] 선택된 모드: {args.mode.upper()}")
    print("=" * 70)

    # 1. 데이터 검증 (Prep)
    if args.mode in ["prep", "all"]:
        print("\n[단계 1/3] 데이터셋 무결성 검증 및 라벨 통계 분석...")
        prep = YOLODataPrep(data_yaml_path=args.data)
        prep.validate_dataset()
        prep.get_class_distribution(split="train")

    # 2. 모델 학습 (Train)
    if args.mode in ["train", "all"]:
        print("\n[단계 2/3] Ultralytics YOLO 모델 학습 개시...")
        run_training(args)

    # 3. 모델 추론 (Infer)
    if args.mode in ["infer", "all"]:
        print("\n[단계 3/3] 최적 모델 기반 추론 및 예측 결과 내보내기...")
        import os
        if not os.path.exists(args.weights) and args.mode == "all":
            fallback_weights = os.path.join(args.project, args.name, "weights", "best.pt")
            if os.path.exists(fallback_weights):
                args.weights = fallback_weights
            else:
                args.weights = args.model

        run_inference(args)

    # 4. 시각화 실험 보고서 작성 (Report)
    if args.mode in ["report", "all"]:
        print("\n[단계 4/4] 시각화 그래프 및 Markdown 실험 보고서 자동 생성 중...")
        tracker = ExperimentTracker()
        reporter = ReportGenerator(tracker)
        report_file = reporter.build_markdown_report()
        print(f"📄 시각화 리포트 생성 완료: {report_file}")

    print("\n✨ 파이프라인 프로세스가 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    main()