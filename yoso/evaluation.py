from PIL import Image
import json
from yoso.demo.app import start_tryon
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_cie76
from torch.utils.data import Dataset, DataLoader
import os
import re


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory containing the images.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.root_dir = root_dir
        self.datapack = self._load_datapack()
        self.transform = transform

    def _load_datapack(self):
        """Group files into datapack by numeric prefix (e.g., 01, 02)."""
        files = sorted(os.listdir(self.root_dir))
        datapack = {}

        # Regex to extract numeric prefix (e.g., "01" from "01_corrected-on-model.jpg")
        pattern = re.compile(r"^(\d+)_")

        for file in files:
            match = pattern.match(file)
            if not match:
                continue  # Skip files without numeric prefix
            prefix = match.group(1)
            if prefix not in datapack:
                datapack[prefix] = {
                    "corrected": None,
                    "on_model": None,
                    "still_life": None,
                }

            if "corrected-on-model" in file:
                datapack[prefix]["corrected"] = os.path.join(self.root_dir, file)
            elif "on-model" in file:
                datapack[prefix]["on_model"] = os.path.join(self.root_dir, file)
            elif "still-life" in file:
                datapack[prefix]["still_life"] = os.path.join(self.root_dir, file)
            datapack[prefix]["output"] = datapack[prefix]["corrected"].replace(
                "corrected-on-model", "output"
            )
            datapack[prefix]["generated_mask"] = datapack[prefix]["corrected"].replace(
                "corrected-on-model", "generated-mask"
            )
        datapack = {k: v for k, v in datapack.items() if all(v.values())}
        return list(datapack.values())

    def __len__(self):
        return len(self.datapack)

    def __getitem__(self, idx):
        return self.datapack[idx]


def load_images(image1_path, image2_path):
    """Load images using PIL and convert to numpy arrays (RGB format)."""
    img1 = np.array(Image.open(image1_path).convert("RGB").resize((768, 1024)))
    img2 = np.array(Image.open(image2_path).convert("RGB").resize((768, 1024)))
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")
    return img1, img2


def calculate_mse(img1, img2):
    """Compute Mean Squared Error (lower = better)."""
    err = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    return err


def calculate_psnr(mse):
    """Compute PSNR (higher = better)."""
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """Compute SSIM (higher = better)."""
    gray1 = np.mean(img1, axis=2).astype(np.uint8)  # Convert RGB to grayscale
    gray2 = np.mean(img2, axis=2).astype(np.uint8)
    return ssim(gray1, gray2, data_range=255)


def calculate_delta_e(img1, img2):
    """Compute average Delta E in CIELAB space (lower = better)."""
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    return np.mean(deltaE_cie76(lab1, lab2))


def compare_images(image1_path, image2_path):
    """Compare two images using all metrics."""
    img1, img2 = load_images(image1_path, image2_path)

    mse = calculate_mse(img1, img2)
    psnr = calculate_psnr(mse)
    ssim_val = calculate_ssim(img1, img2)
    delta_e = calculate_delta_e(img1, img2)

    return {"MSE": mse, "PSNR (dB)": psnr, "SSIM": ssim_val, "Delta E (CIE76)": delta_e}


def ci():
    img = Image.open("data/ci/cloth/00000_00.jpg")
    imgs = {"background": img}
    garment_img = Image.open("data/ci/cloth/00000_00.jpg")
    prompt = "a t shirt"
    is_check = True
    is_checked_crop = False
    denoise_steps = 30
    seed = 42
    image, mask_gray = start_tryon(
        imgs,
        garment_img,
        prompt,
        is_check,
        is_checked_crop,
        denoise_steps,
        seed,
    )
    Path("data/output").mkdir(parents=True, exist_ok=True)
    image.save("data/output/image.png")
    mask_gray.save("data/output/mask.png")


def generate_image():
    img = Image.open("data/ci/cloth/00000_00.jpg")
    imgs = {"background": img}
    garment_img = Image.open("data/ci/cloth/00000_00.jpg")
    prompt = "a t shirt"
    is_check = True
    is_checked_crop = False
    denoise_steps = 30
    seed = 42
    image, mask_gray = start_tryon(
        imgs,
        garment_img,
        prompt,
        is_check,
        is_checked_crop,
        denoise_steps,
        seed,
    )
    Path("data/output").mkdir(parents=True, exist_ok=True)
    image.save("data/output/image.png")
    mask_gray.save("data/output/mask.png")


if __name__ == "__main__":
    prompt = "a garment"
    is_check = True
    is_checked_crop = False
    denoise_steps = 30
    seed = 42
    results = {}
    dataset = ImageDataset(root_dir="data/samples")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i in dataloader:
        corrected_path = i["corrected"][0]
        on_model_path = i["on_model"][0]
        still_life_path = i["still_life"][0]
        output_path = i["output"][0]
        generated_mask_path = i["generated_mask"][0]
        print(on_model_path)
        output, generated_mask = start_tryon(
            {"background": Image.open(on_model_path)},
            Image.open(still_life_path),
            prompt,
            is_check,
            is_checked_crop,
            denoise_steps,
            seed,
        )
        output.save(output_path)
        generated_mask.save(generated_mask_path)
        result = compare_images(corrected_path, output_path)
        print(f"Results for {output_path}: {result}")
        results[output_path] = result
        break  # Remove this line to process all images
    results_path = "data/output/evaluation_results.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
