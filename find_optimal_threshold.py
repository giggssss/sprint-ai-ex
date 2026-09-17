import numpy as np
from ultralytics import YOLO

def main():
    model = YOLO("models/best.pt")
    # Run validation on the val set (uses data.yaml internally by default if model was trained with it)
    print("Running validation to extract Precision-Recall curve data...")
    metrics = model.val(data="/Volumes/Macintosh SUB/Dataset/yolo_data/data.yaml", split="val", plots=False, verbose=False)
    
    # metrics.box.f1 is an array of shape (num_classes, num_thresholds)
    # The thresholds are evenly spaced between 0 and 1, usually 1000 points.
    
    f1_curve = np.array(metrics.box.f1)
    
    if f1_curve.ndim == 2:
        mean_f1 = f1_curve.mean(axis=0)
    else:
        mean_f1 = f1_curve
        
    points = mean_f1.shape[0]
    conf_thresholds = np.linspace(0.0, 1.0, points)
    
    optimal_idx = np.argmax(mean_f1)
    optimal_conf = conf_thresholds[optimal_idx]
    optimal_f1 = mean_f1[optimal_idx]
    
    print(f"Optimal Confidence Threshold: {optimal_conf:.4f}")
    print(f"Max F1 Score (mean across classes): {optimal_f1:.4f}")
    
    # Save the optimal conf to a file so we can read it or use it
    with open("optimal_conf.txt", "w") as f:
        f.write(str(optimal_conf))

if __name__ == "__main__":
    main()
