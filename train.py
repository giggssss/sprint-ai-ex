import argparse
import os
import torch
from ultralytics import YOLO


def run_training(args):
    """Ultralytics YOLO 모델 학습을 수행하고 실험 매니저에 자동 로그 및 리포트를 갱신합니다."""
    print("=" * 60)
    print(f"🚀 YOLO 모델 학습 시작: {args.model}")
    print(f"  • 데이터셋 경로 : {args.data}")
    print(f"  • Epochs        : {args.epochs}")
    print(f"  • Batch Size    : {args.batch}")
    print(f"  • Image Size    : {args.imgsz}")
    print(f"  • Device        : {args.device}")
    print("=" * 60)

    # 1. 모델 로드 (Pretrained Weights)
    model = YOLO(args.model)

    # 2. 학습 실행
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        device=args.device,
        project=args.project,
        name=args.name,
        mosaic=args.mosaic,
        mixup=args.mixup,
        fliplr=args.fliplr,
        save=True,
        plots=True,
        exist_ok=True,
    )

    print("\n✅ 학습이 성공적으로 완료되었습니다!")
    run_dir = os.path.join(args.project, args.name)
    print(f"📁 결과 저장 경로: {run_dir}")

    # 3. 실험 로깅 및 자동 Markdown 보고서 갱신
    try:
        from utils.experiment_manager import ExperimentTracker, ReportGenerator
        tracker = ExperimentTracker()

        val_metrics = {
            "mAP50": round(float(getattr(results.box, "map50", 0.0)), 4),
            "mAP50-95": round(float(getattr(results.box, "map", 0.0)), 4),
            "Precision": round(float(getattr(results.box, "mp", 0.0)), 4),
            "Recall": round(float(getattr(results.box, "mr", 0.0)), 4),
        }

        tracker.log_run(
            run_name=args.name,
            model_name=args.model,
            params=vars(args),
            metrics=val_metrics,
            run_dir=run_dir,
        )

        reporter = ReportGenerator(tracker)
        report_path = reporter.build_markdown_report()
        print(f"📊 최신 실험 보고서 갱신 완료: {report_path}")
    except Exception as e:
        print(f"⚠️ 실험 자동 로깅 중 알림: {e}")

    return results


def main():
    default_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Ultralytics YOLO Object Detection Training")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Ultralytics model name or path (default: yolo11n.pt)")
    parser.add_argument("--data", type=str, default="/Volumes/Macintosh SUB/Dataset/yolo_data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--device", type=str, default=default_device, help=f"Computation device (default: {default_device})")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project output directory")
    parser.add_argument("--name", type=str, default="train_yolo", help="Training run name")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.15, help="Mixup augmentation probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Flip left-right probability")

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
