import kagglehub
import os

os.environ["KAGGLEHUB_CACHE"] = "data"


def download_viton_zalando_hd():
    path = kagglehub.dataset_download("ahemateja19bec1025/viton-zalando-hd")
    print(f"Dataset path: {path}")


if __name__ == "__main__":
    download_viton_zalando_hd()
