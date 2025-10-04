from PIL import Image
import json
from yoso.demo.app import start_tryon
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset, DataLoader
import os
import re
from datetime import datetime
import colour
from scipy.stats import entropy, wasserstein_distance
import lpips
import torch


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


def calculate_delta_e_metrics(img1, img2):
    lab1 = colour.convert(img1 / 255.0, "RGB", "CIE Lab", target_illuminant="D65")
    lab2 = colour.convert(img2 / 255.0, "RGB", "CIE Lab", target_illuminant="D65")
    lab1_flat = lab1.reshape(-1, 3)
    lab2_flat = lab2.reshape(-1, 3)
    delta_e76 = colour.difference.delta_E(lab1_flat, lab2_flat, method="CIE 1976")
    delta_e94 = colour.difference.delta_E(lab1_flat, lab2_flat, method="CIE 1994")
    delta_e00 = colour.difference.delta_E(lab1_flat, lab2_flat, method="CIE 2000")
    return {
        "Delta E (CIE76)": np.mean(delta_e76),
        "Delta E (CIE94)": np.mean(delta_e94),
        "Delta E (CIE2000)": np.mean(delta_e00),
    }


def calculate_histogram_metrics(img1, img2, bins=32):
    lab1 = colour.convert(
        img1 / 255.0, "RGB", "CIE Lab", target_illuminant="D65"
    ).reshape(-1, 3)
    lab2 = colour.convert(
        img2 / 255.0, "RGB", "CIE Lab", target_illuminant="D65"
    ).reshape(-1, 3)
    metrics = {}
    for i, ch in enumerate(["L", "a", "b"]):
        h1, _ = np.histogram(
            lab1[:, i],
            bins=bins,
            range=(lab1[:, i].min(), lab1[:, i].max()),
            density=True,
        )
        h2, _ = np.histogram(
            lab2[:, i],
            bins=bins,
            range=(lab2[:, i].min(), lab2[:, i].max()),
            density=True,
        )
        h1 += 1e-8
        h2 += 1e-8
        h1 /= np.sum(h1)
        h2 /= np.sum(h2)
        metrics[f"{ch}_EMD"] = wasserstein_distance(h1, h2)
        metrics[f"{ch}_KL"] = entropy(h1, h2)
    return metrics


def compute_lpips(img1, img2, net="vgg"):
    """Compute LPIPS (Learned Perceptual Image Patch Similarity)."""
    loss_fn = lpips.LPIPS(net=net)
    t1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    t2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    d = loss_fn(t1, t2)
    return {"LPIPS": float(d.item())}


def calculate_lpips(img1, img2, net="vgg"):
    loss_fn = lpips.LPIPS(net=net)
    t1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    t2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    d = loss_fn(t1, t2)
    return float(d.item())


def compare_images(image1_path, image2_path):
    img1, img2 = load_images(image1_path, image2_path)

    mse = calculate_mse(img1, img2)
    psnr = calculate_psnr(mse)
    ssim_val = calculate_ssim(img1, img2)

    delta_e_metrics = calculate_delta_e_metrics(img1, img2)

    hist_metrics = calculate_histogram_metrics(img1, img2)

    lpips_val = calculate_lpips(img1, img2)

    results = {
        "MSE": mse,
        "PSNR (dB)": psnr,
        "SSIM": ssim_val,
        "LPIPS": lpips_val,
        **delta_e_metrics,
        **hist_metrics,
    }

    return results


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
        # break  # Remove this line to process all images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data/output", exist_ok=True)

    # Update filename and JSON
    results_path = f"data/output/evaluation_results_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
