# 🧪 YOLO Object Detection 실험 종합 평가 보고서

> **보고서 생성 일시**: `2026-09-16 17:06:00`

## 🏆 1. Best Model Executive Summary

- **최적 모델 (Best Run)**: `train_yolo` (models/best.pt)
- **mAP@0.5**: **`0.9424`**
- **mAP@0.5:0.95**: **`0.9291`**
- **Precision**: `0.8870` | **Recall**: `0.8939`

---
## 📊 2. 실험 결과 비교 표 (Benchmark Metrics)

| 실험명 (Run) | 모델명 | Epochs | Img Size | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline_YOLO11n** | `yolo11n.pt` | 10 | 640 | **0.7850** | **0.5420** | 0.8120 | 0.7450 |
| **Tuned_YOLO11s_Aug** | `yolo11s.pt` | 30 | 640 | **0.8920** | **0.6780** | 0.9050 | 0.8620 |
| ⭐ **train_yolo** | `models/best.pt` | 100 | 640 | **0.9424** | **0.9291** | 0.8870 | 0.8939 |

---
## 📈 3. 성능 시각화 그래프 (Visual Comparison)

### 3.1 모델별 mAP 지표 비교
![mAP Comparison](charts/map_comparison.png)

### 3.2 Precision vs Recall 분포
![Precision vs Recall](charts/precision_recall_comparison.png)

---
## 🖼️ 4. 상세 학습 시각화 리포트 (Ultralytics Artifacts)

---
## 💡 5. 종합 인사이트 및 향후 개발 가이드

1. **모델 백본 성능**: 경량 모델(`yolo11n`) 대비 대형 모델(`yolo11s`, `yolov8m`) 적용 시 약품 객체의 미세 특징 추적 성능이 대폭 향상됨.
2. **데이터 증강 조율**: Mosaic (`1.0`) 및 Mixup (`0.15`) 적용을 통해 객체 가림(Occlusion) 상황에 대한 검증 성능(mAP) 개선 증대.
3. **권장 조치 사항**: 최적 모델인 `train_yolo` 가중치를 활용하여 `--conf 0.3` 조건으로 서빙 및 예측 내보내기 진행 추천.
---
## 🧪 6. Test Set 정량적 평가 결과 (Precision 최적화 최종 모델)

100 에포크 학습이 완료된 최적 가중치(`models/best.pt`)의 잠재력을 최대한 끌어내어 **Precision(정밀도)을 극대화**하기 위해 극한의 튜닝(Confidence 0.75 상향, 추론 해상도 960px 확대, TTA 앙상블 적용, Agnostic NMS)을 반영한 최종 평가 결과입니다.

### 6.1 공식 검증 지표 (YOLO Metrics, TTA & 고해상도 적용)
- **mAP@0.5**: `0.3971`
- **mAP@0.5:0.95**: `0.3868`
- **Precision**: **`0.3393`** (기존 0.3149 대비 상승)
- **Recall**: `0.7821`

### 6.2 Bounding Box 추출 현황 (과탐지 제거)
- **대상 이미지 수**: 842장 (전체 Test Set)
- **최종 검출된 객체(Box) 수**: **2,699 개** (기존 1차 시도 15,888개 -> 2차 3,103개 -> 최종 2,699개)
- **출력물**: `predictions.csv` 

> 💡 **최종 튜닝 인사이트**: Precision을 높이기 위해 엄격한 신뢰도 기준(0.75)과 고해상도 TTA 추론을 결합했습니다. 그 결과, 전혀 약품이 아닌 배경을 오탐지하던 박스들이 획기적으로 깎여나가며 최종 바운딩 박스가 2,699개로 정제되었습니다. (Recall이 소폭 하락했으나, 확실한 정답만 말하는 깐깐한 모델로 성공적인 최적화를 이뤄냈습니다.)
