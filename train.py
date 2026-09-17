import os
import argparse
import shutil
import torch
import yaml
from ultralytics import YOLO


def run_training(args):
    """Ultralytics YOLO 모델 학습을 수행하고 실험 매니저에 자동 로그 및 리포트를 갱신합니다."""
    print("=" * 60)
    print(f"🚀 YOLO 모델 학습 준비: {args.model}")
    print(f"  • 데이터셋 경로 : {args.data}")
    print(f"  • Final Epochs  : {args.epochs}")
    print(f"  • Batch Size    : {args.batch}")
    print(f"  • Image Size    : {args.imgsz}")
    print(f"  • Device        : {args.device}")
    
    hyperparameters = {}
    
    # [1] 하이퍼파라미터 튜닝 로직
    if args.tune:
        print("=" * 60)
        print(f"🔬 하이퍼파라미터 튜닝 모드 활성화 (Epochs: {args.tune_epochs}, Iters: {args.tune_iters})")
        print("=" * 60)
        tune_model = YOLO(args.model)
        
        # tune 실행 (runs/detect/tune 에 결과가 저장됨)
        tune_model.tune(
            data=args.data,
            epochs=args.tune_epochs,
            iterations=args.tune_iters,
            optimizer='AdamW',
            plots=False,
            save=False,
            val=False,
            device=args.device,
            amp=False
        )
        
        # tune 결과 불러오기
        tune_dir = os.path.join(args.project, "tune")
        best_hp_path = os.path.join(tune_dir, "best_hyperparameters.yaml")
        
        if os.path.exists(best_hp_path):
            print(f"\n🌟 최적의 하이퍼파라미터를 찾았습니다: {best_hp_path}")
            with open(best_hp_path, 'r', encoding='utf-8') as f:
                hyperparameters = yaml.safe_load(f)
            print(hyperparameters)
        else:
            print(f"⚠️ 튜닝 결과를 찾을 수 없습니다 ({best_hp_path}). 기본 파라미터로 진행합니다.")

    print("=" * 60)
    print("🎯 최종 학습(Main Training) 시작")
    print("=" * 60)
    
    # 2. 모델 로드 (Pretrained Weights) - 새로 인스턴스화
    model = YOLO(args.model)

    # 3. 기본 인자 설정
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "amp": False,
        "deterministic": False,
        "save": True,
        "plots": True,
        "exist_ok": True,
    }
    
    # 4. 하이퍼파라미터 병합 (튜닝 결과가 있다면 덮어쓰기)
    if not args.tune:
        # 튜닝이 아닐 때만 명령줄 기본 인자 사용
        train_kwargs.update({
            "lr0": args.lr0,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "fliplr": args.fliplr,
        })
    else:
        # 튜닝 결과 병합
        train_kwargs.update(hyperparameters)

    # 5. 학습 실행
    results = model.train(**train_kwargs)

    print("\n✅ 학습이 성공적으로 완료되었습니다!")
    run_dir = os.path.join(args.project, args.name)
    best_pt_path = os.path.join(run_dir, "weights", "best.pt")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    if os.path.exists(best_pt_path):
        target_pt = os.path.join(models_dir, "best.pt")
        shutil.copy2(best_pt_path, target_pt)
        print(f"📁 가중치(best.pt) 파일 복사 완료: {target_pt}")
    else:
        print(f"⚠️ 학습된 가중치를 찾을 수 없습니다: {best_pt_path}")
    print(f"📁 결과 저장 경로: {run_dir}")

    # 6. 실험 로깅 및 자동 Markdown 보고서 갱신
    try:
        from utils.experiment_manager import ExperimentTracker, ReportGenerator
        tracker = ExperimentTracker()

        val_metrics = {
            "mAP50": round(float(getattr(results.box, "map50", 0.0)), 4),
            "mAP50-95": round(float(getattr(results.box, "map", 0.0)), 4),
            "Precision": round(float(getattr(results.box, "mp", 0.0)), 4),
            "Recall": round(float(getattr(results.box, "mr", 0.0)), 4),
        }

        # 튜닝된 파라미터가 있다면 로깅에 추가
        log_params = vars(args).copy()
        if hyperparameters:
            log_params.update(hyperparameters)

        tracker.log_run(
            run_name=args.name,
            model_name=args.model,
            params=log_params,
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

    parser = argparse.ArgumentParser(description="Ultralytics YOLO Object Detection Training & Tuning")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Ultralytics model name or path")
    parser.add_argument("--data", type=str, default="/Volumes/Macintosh SUB/Dataset/yolo_data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of final training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--device", type=str, default=default_device, help=f"Computation device (default: {default_device})")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project output directory")
    parser.add_argument("--name", type=str, default="train_yolo", help="Training run name")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.15, help="Mixup augmentation probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Flip left-right probability")
    
    # 튜닝 관련 파라미터 추가
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning before final training")
    parser.add_argument("--tune_epochs", type=int, default=5, help="Number of epochs per tuning iteration (default: 5)")
    parser.add_argument("--tune_iters", type=int, default=10, help="Number of tuning iterations/trials (default: 10)")

    args = parser.parse_args()
    
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS is requested but not available. Falling back to CPU.")
        args.device = 'cpu'
        
    run_training(args)


if __name__ == "__main__":
    main()
