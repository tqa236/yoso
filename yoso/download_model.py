import os
import requests


def main():
    # Create the required folder structure
    folders = ["ckpt/densepose", "ckpt/humanparsing", "ckpt/openpose/ckpts"]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # URLs of the files to be downloaded
    files = {
        "ckpt/humanparsing/parsing_atr.onnx": "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx",
        "ckpt/humanparsing/parsing_lip.onnx": "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx",
        "ckpt/densepose/model_final_162be9.pkl": "https://huggingface.co/yisol/IDM-VTON/resolve/main/densepose/model_final_162be9.pkl",
        "ckpt/openpose/ckpts/body_pose_model.pth": "https://huggingface.co/yisol/IDM-VTON/resolve/main/openpose/ckpts/body_pose_model.pth",
    }

    # Function to download files
    def download_file(url, path):
        response = requests.get(url, stream=True)
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded: {path}")

    # Download each file
    for path, url in files.items():
        download_file(url, path)

    print("Download and folder setup completed successfully.")


if __name__ == "__main__":
    main()
