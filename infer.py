import argparse
import os
import glob
import pandas as pd
import torch
from ultralytics import YOLO

import config


def run_inference(args):
    """학습된 YOLO 모델을 사용하여 이미지 추론을 실행하고 CSV 파일로 내보냅니다."""
    weights_path = config.resolve_weights(args.weights)
    print("=" * 60)
    print(f"🔍 YOLO 추론 시작: {weights_path}")
    print(f"  • 입력 소스       : {args.source}")
    print(f"  • Conf Threshold : {args.conf}")
    print(f"  • IoU Threshold  : {args.iou}")
    print("=" * 60)

    # 1. 가중치 존재 확인
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights_path}")

    model = YOLO(weights_path)

    # 2. 소스 파일 탐색
    if os.path.isdir(args.source):
        image_paths = sorted(
            glob.glob(os.path.join(args.source, "*.[jJ][pP][gG]"))
            + glob.glob(os.path.join(args.source, "*.[pP][nN][gG]"))
            + glob.glob(os.path.join(args.source, "*.[jJ][pP][eE][gG]"))
        )
    else:
        image_paths = [args.source]

    print(f"📷 총 {len(image_paths)}개 이미지 추론 진행 중...")

    # 3. Predict 실행 및 시각화 저장
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=os.path.dirname(args.output_dir),
        name=os.path.basename(args.output_dir),
        exist_ok=True,
    )

    # 4. Bounding Box 결과 구조화 및 CSV 수집
    records = []
    for r in results:
        image_name = os.path.basename(r.path)
        boxes = r.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = r.names[cls_id]
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()  # [xmin, ymin, xmax, ymax]

                records.append({
                    "image_name": image_name,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "xmin": round(xyxy[0], 2),
                    "ymin": round(xyxy[1], 2),
                    "xmax": round(xyxy[2], 2),
                    "ymax": round(xyxy[3], 2),
                })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
    df.to_csv(args.save_csv, index=False, encoding="utf-8-sig")

    print("\n✅ 추론 완료 및 결과 내보내기 성공!")
    print(f"📊 예측 결과 CSV 저장 위치 : {args.save_csv}")
    print(f"🖼️ 시각화 이미지 저장 위치 : {args.output_dir}")
    print(f"📈 총 탐지된 바운딩 박스 개수: {len(df)}개")

    return df


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Object Detection Inference")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained model weights (.pt)")
    parser.add_argument("--source", type=str, default=config.VAL_IMAGES_DEFAULT, help="Image file or directory path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold (default: 0.6)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--device", type=str, default=config.DEVICE, help=f"Computation device (default: {config.DEVICE})")
    parser.add_argument("--output-dir", type=str, default="runs/predict/val_results", help="Directory to save output images")
    parser.add_argument("--save-csv", type=str, default="runs/predict/predictions.csv", help="CSV file path to save detection results")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
