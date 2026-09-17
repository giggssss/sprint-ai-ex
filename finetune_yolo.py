from ultralytics import YOLO

def main():
    print("Loading SimCLR-pretrained YOLO model (Scratch YOLO11s)...")
    # Load the model with SSL-pretrained backbone
    model = YOLO('models/simclr_scratch_yolo11s.pt')
    
    print("Starting Fine-tuning on annotated Train set...")
    # Fine-tune the model. We don't freeze the backbone so it can adapt to the detection task,
    # but the rich representations learned via SSL will provide a massive head-start.
    model.train(
        data='/Volumes/Macintosh SUB/Dataset/yolo_data/data.yaml',
        epochs=100, # 100 epochs since head is untrained
        batch=16,
        imgsz=640,
        project='runs/detect',
        name='train_yolo11s_scratch_ssl',
        device='mps' # Use Apple Silicon
    )
    print("Fine-tuning complete. Best model saved in runs/detect/train_yolo_finetune_ssl/weights/best.pt")

if __name__ == '__main__':
    main()
