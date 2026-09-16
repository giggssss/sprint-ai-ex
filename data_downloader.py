from utils.data_downloader import KaggleDataDownloader

__all__ = ["KaggleDataDownloader"]

if __name__ == "__main__":
    downloader = KaggleDataDownloader()
    path = downloader.download()
    print("Path to competition files:", path)
