# 💊 Sprint AI Object Detection (Pill Classification & OOD Rejection)

본 프로젝트는 불균형 데이터셋 및 17개 미학습 클래스(OOD)가 혼재된 테스트 환경에서 알약 객체를 고정밀로 탐지하고 분류하기 위한 **Ultralytics YOLO 기반 객체 탐지 및 OOD 거절 파이프라인**입니다.

> 🎯 **공식 대회 평가 지표**: `mAP@[0.75:0.95]` (Strict IoU thresholds 0.75 ~ 0.95)

---

## 📁 프로젝트 구조

```
sprint-basic-project/
├── config.py                 # 중앙 집중 설정 모듈 (경로, 디바이스, 기본 파라미터)
├── main.py                   # 마스터 파이프라인 CLI (prep, train, infer, eval, ood, report, all)
├── train.py                  # YOLO 모델 학습 및 하이퍼파라미터 튜닝
├── evaluate.py               # 모델 평가, k-NN OOD 거절 필터링 및 임계값 최적화
├── infer.py                  # 단일/배치 이미지 추론 및 바운딩 박스 CSV 추출
├── viewer_app.py             # Streamlit 기반 추론 결과 및 정답(GT) 시각화 분석 앱
├── simclr_yolo.py            # Self-Supervised Learning (SimCLR) 사전학습
├── class_mapping.json        # YOLO 클래스 ID (0~55) -> 대회 카테고리 ID 매핑
├── optimal_conf.txt          # 최적 Confidence 임계값 저장 파일
├── predictions.csv           # 최종 대회 제출용 예측 결과 CSV
├── experiments/              # 실험 기록, 시각화 차트 및 분석 보고서
│   ├── charts/
│   ├── reports/
│   ├── experiment_history.json
│   └── experiment_report.md
└── utils/
    ├── __init__.py           # 공통 모듈 익스포트
    ├── data_prep.py          # Stratified Split 데이터셋 구축, 라벨 변환 및 검증
    ├── evaluator.py          # COCO mAP@[0.75:0.95] 평가 및 k-NN Feature Bank OOD 엔진
    ├── experiment_manager.py # 실험 로깅 및 Markdown 보고서 자동 생성
    └── data_downloader.py    # Kaggle 데이터 다운로더
```

---

## 🚀 빠른 시작 (Usage)

### 1. 마스터 파이프라인 실행 (`main.py`)
전체 워크플로우를 단일 명령어로 제어할 수 있습니다:

```bash
# 1) 데이터셋 무결성 및 클래스 분포 검증
python main.py --mode prep

# 2) 모델 학습 (기본 30 Epochs)
python main.py --mode train --model yolo11n.pt --epochs 30 --batch 16

# 3) 테스트셋 정량 평가 (mAP@[0.75:0.95] 산출 및 predictions.csv 생성)
python main.py --mode eval

# 4) k-NN Feature Bank 기반 OOD 거절 필터링 스윕
python main.py --mode ood

# 5) 실험 시각화 보고서 갱신
python main.py --mode report
```

---

### 2. 정량 평가 및 OOD 거절 파이프라인 (`evaluate.py`)
대회 공식 기준인 `mAP@[0.75:0.95]`를 산출하고 미학습 클래스 오탐지를 제거합니다:

```bash
# 표준 테스트셋 평가 및 predictions.csv 생성
python evaluate.py --mode test

# k-NN Feature Bank 임베딩 코사인 유사도 기반 OOD 거절 스윕
python evaluate.py --mode ood

# Validation 셋 F1 점수 기준 최적 Confidence 임계값 탐색
python evaluate.py --mode threshold

# 기존 CSV 파일 단독 정량 평가
python evaluate.py --mode eval_only --csv predictions.csv
```

---

### 3. 하이퍼파라미터 튜닝 및 학습 (`train.py`)

```bash
# 기본 학습
python train.py --model yolo11s.pt --epochs 50 --batch 16 --imgsz 640

# 유전 알고리즘 기반 하이퍼파라미터 자동 튜닝 후 학습
python train.py --model yolo11s.pt --tune --tune_epochs 5 --tune_iters 10
```

---

### 4. 인터랙티브 분석 뷰어 (`viewer_app.py`)
모델 예측과 실제 정답(GT)을 바운딩 박스별로 1:1 교차 검증할 수 있는 대시보드입니다:

```bash
streamlit run viewer_app.py
```
- **주요 기능**:
  - `TP (Correct)`, `FP (Misclassified)`, `FP (Background)`, `FN (Missed)` 상태별 시각적 구분
  - 전체 원본 이미지 상의 **🟩 실제 정답 (GT) 박스** 및 **🟥 모델 예측 (Pred) 박스** 독립 토글 지원
  - IoU 임계값 및 Confidence 임계값 동적 필터링