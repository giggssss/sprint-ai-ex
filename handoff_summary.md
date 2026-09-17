# 📝 Project Handoff Summary: Sprint AI Object Detection (Pill Classification)

## 1. Current State & Achievements
- **Objective:** Maximize mAP on a highly imbalanced, OOD-heavy test set without using few-shot classification.
- **Data Preparation Fix:** 
  - Fixed a critical mapping bug in `utils/data_prep.py` that caused the rare class '카나브정' (Class 55) to be dropped entirely.
  - Successfully generated `yolo_data_v2` with a robust Stratified Split.
- **OOD Rejection via SSL Embeddings:**
  - Implemented `eval_ood_test.py` to intercept YOLO backbone embeddings before classification.
  - Upgraded to a precise **k-NN Feature Bank** instead of a simple mean prototype.
  - Successfully filtered out False Positives caused by the 17 Unseen distractor classes in the test set using a Cosine Similarity Threshold.
- **Current Metrics:**
  - **Recall:** 89.1% (YOLO can successfully locate the pills).
  - **mAP@0.50:** 41.19% (Evaluated purely on the 56 known classes using pycocotools).

## 2. Why mAP 90% is Not Reached Yet
The OOD filtering logic is mathematically sound and effectively drops distractor predictions. However, the base YOLO model's precision is capped at ~41% because:
- **Insufficient Data:** The model was finetuned on only 187 train images for 56 classes (~3 images per class).
- **Insufficient Training:** Trained for only 50 epochs.
- **Low Resolution:** Pill markings are microscopic, requiring higher `imgsz`.

## 3. Next Steps (Roadmap to 90% mAP)
To achieve the user's target of 90% mAP, the next agent must focus purely on **Training Optimization (Fine-Tuning)**:
1. **Model Capacity:** Switch to a larger model (`yolo11m.pt` or `yolo11l.pt`).
2. **Resolution:** Increase `imgsz` to 640 or 1280 to capture pill engravings.
3. **Data Augmentation:** Apply heavy Mosaic, MixUp, HSV manipulation, and Random Perspective in the training configuration.
4. **Longer Training:** Increase epochs to 300~500 with Early Stopping patience set to 50.
5. **Class Imbalance:** Implement Class Weights (Focal Loss) to further boost rare classes like '카나브정'.

## 4. Key Files to Know
- `utils/data_prep.py`: Contains the fixed Stratified Split logic.
- `eval_ood_test.py`: The crown jewel of the OOD rejection pipeline (uses k-NN Feature Bank).
- `finetune_yolo.py`: The entry point for training (needs hyperparameters updated for the next steps).
