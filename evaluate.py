"""
대회 모델 평가 및 OOD 필터링 통합 실행 스크립트 (CLI)
사용법:
    1. 표준 테스트셋 평가 및 predictions.csv 생성:
       python evaluate.py --mode test
    2. k-NN Feature Bank 기반 OOD 거절 필터링 및 최적화:
       python evaluate.py --mode ood
    3. Val 셋 기반 최적 F1 Confidence 임계값 탐색:
       python evaluate.py --mode threshold
    4. 기존 예측 CSV 파일 단독 정량 평가:
       python evaluate.py --mode eval_only --csv predictions.csv
"""
import argparse
import sys
import os

import config
from utils.evaluator import YOLOEvaluator, OODEvaluator


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Evaluation & OOD Rejection Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "ood", "threshold", "eval_only"],
        help="Evaluation mode: test (standard inference + eval), ood (k-NN feature bank filtering), threshold (find optimal conf), eval_only (evaluate CSV)",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default=config.DATA_YAML_DEFAULT, help="Path to data.yaml")
    parser.add_argument("--test-images", type=str, default=config.TEST_IMAGES_DEFAULT, help="Path to test images directory")
    parser.add_argument("--test-anns", type=str, default=None, help="Path to test annotations JSON (COCO format)")
    parser.add_argument("--csv", type=str, default=config.DEFAULT_PREDICTIONS_CSV, help="Output or input predictions CSV path")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (default: read from optimal_conf.txt or 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size (default: 960)")
    parser.add_argument("--device", type=str, default=config.DEVICE, help=f"Computation device (default: {config.DEVICE})")

    args = parser.parse_args()

    weights = config.resolve_weights(args.weights)
    anns = args.test_anns or config.resolve_test_annotations()

    print("=" * 65)
    print(f"🎯 [평가 파이프라인 개시] 모드: {args.mode.upper()}")
    print(f"  • 가중치: {weights}")
    print(f"  • 데이터 설정: {args.data}")
    print(f"  • 디바이스: {args.device}")
    print("=" * 65)

    evaluator = YOLOEvaluator(
        weights_path=weights,
        data_yaml_path=args.data,
        device=args.device,
    )

    if args.mode == "test":
        evaluator.predict_test_set(
            test_images_dir=args.test_images,
            test_annotations_path=anns,
            output_csv=args.csv,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
        )
        evaluator.evaluate_coco(predictions_csv=args.csv, annotations_path=anns)

    elif args.mode == "ood":
        ood_evaluator = OODEvaluator(
            weights_path=weights,
            data_yaml_path=args.data,
            device=args.device,
        )
        ood_evaluator.run_ood_sweep(
            test_images_dir=args.test_images,
            test_annotations_path=anns,
            output_csv=args.csv,
        )

    elif args.mode == "threshold":
        evaluator.find_optimal_threshold(split="val")

    elif args.mode == "eval_only":
        evaluator.evaluate_coco(predictions_csv=args.csv, annotations_path=anns)

    print("\n✨ 평가 태스크가 완료되었습니다.")


if __name__ == "__main__":
    main()
