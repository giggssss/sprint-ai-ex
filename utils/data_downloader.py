import os
import shutil
import kagglehub


class KaggleDataDownloader:
    """Kaggle Competition 데이터를 kagglehub를 이용하여 다운로드하고
    지정된 로컬 디렉터리에 저장/관리하는 클래스입니다.
    """

    DEFAULT_TARGET_DIR = "/Volumes/Macintosh SUB/Dataset/sprint_basic_project_data"
    DEFAULT_COMPETITION = "ai14-level-project"

    def __init__(
        self,
        competition_name: str = DEFAULT_COMPETITION,
        target_dir: str = DEFAULT_TARGET_DIR,
    ):
        """KaggleDataDownloader 초기화 메서드.

        Args:
            competition_name (str): Kaggle Competition 이름
            target_dir (str): 데이터 저장 목표 디렉터리 경로
        """
        self.competition_name = competition_name
        self.target_dir = target_dir

    def download(self, force_copy: bool = False) -> str:
        """Kaggle Competition 데이터를 다운로드하고 지정된 디렉터리로 복사합니다.

        Args:
            force_copy (bool): 목표 디렉터리에 파일이 존재하더라도 덮어쓸지 여부

        Returns:
            str: 데이터가 위치한 최종 디렉터리 경로
        """
        print(f"[{self.competition_name}] 데이터 다운로드를 시작합니다...")

        # 1. kagglehub 경쟁 데이터 다운로드 (캐시 위치 반환)
        cache_path = kagglehub.competition_download(self.competition_name)
        print(f"Path to competition files (cache): {cache_path}")

        # 2. 목표 디렉터리 존재 여부 확인 및 생성
        os.makedirs(self.target_dir, exist_ok=True)

        # 3. 목표 디렉터리로 파일 복사
        if os.path.exists(cache_path):
            cache_items = os.listdir(cache_path)
            target_items = os.listdir(self.target_dir)

            if not target_items or force_copy:
                print(f"데이터를 목표 디렉터리로 복사 중... -> {self.target_dir}")
                for item in cache_items:
                    src_path = os.path.join(cache_path, item)
                    dst_path = os.path.join(self.target_dir, item)

                    if os.path.isdir(src_path):
                        if os.path.exists(dst_path):
                            shutil.rmtree(dst_path)
                        shutil.copytree(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)
                print(f"성공적으로 복사되었습니다: {self.target_dir}")
            else:
                print(f"목표 디렉터리에 데이터가 이미 존재합니다: {self.target_dir}")

        return self.target_dir


if __name__ == "__main__":
    downloader = KaggleDataDownloader()
    save_path = downloader.download()
    print(f"최종 데이터 위치: {save_path}")
