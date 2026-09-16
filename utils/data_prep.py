import os
import glob
import yaml
from collections import Counter


class YOLODataPrep:
    """YOLO 데이터셋 검증 및 data.yaml 관리 클래스."""

    def __init__(self, data_yaml_path: str = "/Volumes/Macintosh SUB/Dataset/yolo_data/data.yaml"):
        self.data_yaml_path = data_yaml_path
        self.dataset_dir = os.path.dirname(data_yaml_path)
        self.config = {}

    def load_yaml(self) -> dict:
        """data.yaml 파일을 로드합니다."""
        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"data.yaml 파일을 찾을 수 없습니다: {self.data_yaml_path}")

        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        return self.config

    def validate_dataset(self) -> dict:
        """train 및 val 디렉터리의 이미지/라벨 개수 및 파일 매핑을 검증합니다."""
        self.load_yaml()
        stats = {}

        for split in ["train", "val"]:
            img_dir = os.path.join(self.dataset_dir, split, "images")
            lbl_dir = os.path.join(self.dataset_dir, split, "labels")

            images = glob.glob(os.path.join(img_dir, "*.[jJ][pP][gG]")) + \
                     glob.glob(os.path.join(img_dir, "*.[pP][nN][gG]")) + \
                     glob.glob(os.path.join(img_dir, "*.[jJ][pP][eE][gG]"))
            labels = glob.glob(os.path.join(lbl_dir, "*.txt"))

            img_basenames = {os.path.splitext(os.path.basename(p))[0] for p in images}
            lbl_basenames = {os.path.splitext(os.path.basename(p))[0] for p in labels}

            unmatched_imgs = img_basenames - lbl_basenames
            unmatched_lbls = lbl_basenames - img_basenames

            stats[split] = {
                "num_images": len(images),
                "num_labels": len(labels),
                "unmatched_images": len(unmatched_imgs),
                "unmatched_labels": len(unmatched_lbls),
            }

            print(f"--- [{split.upper()} Split] ---")
            print(f"  • 이미지 개수: {len(images)}")
            print(f"  • 라벨 개수  : {len(labels)}")
            if unmatched_imgs:
                print(f"  ⚠️ 라벨 없는 이미지: {len(unmatched_imgs)}개")
            if unmatched_lbls:
                print(f"  ⚠️ 이미지 없는 라벨: {len(unmatched_lbls)}개")

        print("\n✅ 데이터셋 무결성 검사가 완료되었습니다.")
        return stats

    def get_class_distribution(self, split: str = "train") -> Counter:
        """라벨 파일들을 탐색하여 클래스별 바운딩 박스 개수 분포를 계산합니다."""
        lbl_dir = os.path.join(self.dataset_dir, split, "labels")
        label_files = glob.glob(os.path.join(lbl_dir, "*.txt"))

        class_counts = Counter()
        for lbl_file in label_files:
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

        names = self.config.get("names", [])
        print(f"\n--- [{split.upper()} Class Distribution (Top 10)] ---")
        for class_id, count in class_counts.most_common(10):
            class_name = names[class_id] if class_id < len(names) else f"Class {class_id}"
            print(f"  • [{class_id:02d}] {class_name}: {count}개")

        return class_counts


if __name__ == "__main__":
    prep = YOLODataPrep()
    prep.validate_dataset()
    prep.get_class_distribution("train")
