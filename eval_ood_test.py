"""
k-NN Feature Bank 기반 OOD 거절 및 mAP@[0.75:0.95] 최적화 스크립트 (하위 호환 래퍼)
통합된 utils.evaluator.OODEvaluator 모듈을 호출합니다.
"""
import config
from utils.evaluator import OODEvaluator, YOLOEvaluator


def main():
    weights = config.resolve_weights()
    anns = config.resolve_test_annotations()

    print(f"🚀 [OOD Filtering Pipeline] Weights: {weights}")
    print(f"  • Annotations: {anns}")

    ood_evaluator = OODEvaluator(weights_path=weights)
    best_map, best_thresh, best_csv = ood_evaluator.run_ood_sweep(
        test_annotations_path=anns,
        output_csv=config.DEFAULT_PREDICTIONS_CSV,
    )
    print(f"\n✅ 완료: Best mAP@[0.75:0.95] = {best_map:.4f} (Threshold: {best_thresh:.2f}) -> {best_csv}")


if __name__ == "__main__":
    main()
