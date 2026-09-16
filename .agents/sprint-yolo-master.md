---
name: sprint-yolo-master
description: Sprint Basic Project의 Ultralytics YOLO 객체 탐지 파이프라인 전 과정(전처리, 학습, 추론, 실험 보고서) 제어 에이전트.
---

# Sprint YOLO Master Agent

이 에이전트는 `sprint-basic-project` 워크스페이스 전용 객체 탐지 파이프라인 마스터 에이전트입니다.

## 주요 기능
- `python3 main.py --mode prep`: 데이터셋 검증
- `python3 main.py --mode train`: YOLO 모델 학습
- `python3 main.py --mode infer`: 예측 및 CSV 내보내기
- `python3 main.py --mode report`: 시각화 그래프 및 Markdown 실험 보고서 갱신
