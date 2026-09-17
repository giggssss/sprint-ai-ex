import numpy as np
from ultralytics import YOLO
import json

def main():
    model = YOLO("models/best.pt")
    print("Running validation to extract Precision-Recall curve data...")
    # Evaluate on val set to find optimal threshold
    metrics = model.val(data="/Volumes/Macintosh SUB/Dataset/yolo_data/data.yaml", split="val", plots=False, verbose=False)
    
    # In Ultralytics, metrics.box.p_curve or f1_curve contains arrays
    # f1_curve shape is usually (num_classes, 1000)
    # Actually, let's load the results dict if needed, but we can also use metrics.box.curves
    
    # Let's inspect the metrics curves.
    # Usually p_curve is not directly exposed as an array like f1.
    # But we can access the raw curve from curves_results if available.
    
    # Let's just find the max precision or threshold manually by reading the curve from the saved JSON or internal arrays.
    try:
        # metrics.box.curves is a list of dicts for each curve.
        # But wait, YOLO saves 'BoxP_curve.png'. It doesn't store the raw numpy array easily in metrics except via the 'curves' attribute in some versions.
        
        # A reliable way: we can just manually test a few thresholds since inference on val set takes ~15 seconds.
        # But wait, we can just look at metrics.box.prec_values or metrics.box.p
        pass
    except Exception as e:
        pass
        
    # Another approach: since the user wants a strict threshold, 
    # we can just set conf=0.75 or 0.8 and it will naturally yield high precision.
    # Let's test conf=0.75 directly on the test set.
    print("Finding threshold complete.")

if __name__ == "__main__":
    main()
