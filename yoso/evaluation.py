import os
import numpy as np
from pathlib import Path
from typing import Callable
from PIL import Image
import json
from yoso.demo.app import start_tryon

# =============================================================================
# Image Loading Functions
# =============================================================================


def load_images_from_dir(
    dir_path: str, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
) -> list[tuple[str, np.ndarray]]:
    """
    Load all images from a directory.

    Args:
        dir_path: Path to directory containing images
        extensions: tuple of valid image extensions

    Returns:
        list of (filename, image_array) tuples
    """
    path = Path(dir_path)
    images = []

    for ext in extensions:
        for img_path in sorted(path.glob(f"*{ext}")):
            try:
                img = Image.open(img_path)
                img_array = np.array(img)
                images.append((img_path.name, img_array))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    return images


def load_image_sets(
    set1_path: str,
    set2_path: str,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> tuple[list, list]:
    """
    Load images from two directories.

    Args:
        set1_path: Path to first image set
        set2_path: Path to second image set
        extensions: tuple of valid image extensions

    Returns:
        tuple of two lists containing (filename, image_array) pairs
    """
    set1 = load_images_from_dir(set1_path, extensions)
    set2 = load_images_from_dir(set2_path, extensions)

    print(f"Loaded {len(set1)} images from set 1")
    print(f"Loaded {len(set2)} images from set 2")

    return set1, set2


# =============================================================================
# Image Processing Functions
# =============================================================================


def apply_function_to_images(
    images: list[tuple[str, np.ndarray]], func: Callable[[np.ndarray], np.ndarray]
) -> list[tuple[str, np.ndarray]]:
    """
    Apply a function to each image in a list.

    Args:
        images: list of (filename, image_array) tuples
        func: Function that takes an image array and returns a processed image array

    Returns:
        list of (filename, processed_image) tuples
    """
    processed = []
    for filename, img in images:
        try:
            result = func(img)
            processed.append((filename, result))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return processed


# =============================================================================
# Metric Calculation Functions
# =============================================================================


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Mean Squared Error between two images."""
    return float(np.mean((img1.astype(float) - img2.astype(float)) ** 2))


def calculate_psnr(
    img1: np.ndarray, img2: np.ndarray, max_pixel: float = 255.0
) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(max_pixel / np.sqrt(mse)))


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(img1.astype(float) - img2.astype(float))))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate simplified SSIM (Structural Similarity Index)."""
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
    if len(img2.shape) == 3:
        img2 = np.mean(img2, axis=2)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.std(img1)
    sigma2 = np.std(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2)
    )

    return float(ssim)


def get_metric_function(metric_name: str) -> Callable:
    """Get metric calculation function by name."""
    metric_functions = {
        "mse": calculate_mse,
        "psnr": calculate_psnr,
        "mae": calculate_mae,
        "ssim": calculate_ssim,
    }
    return metric_functions.get(metric_name)


# =============================================================================
# Evaluation Functions
# =============================================================================


def match_image_pairs(
    set1: list[tuple[str, np.ndarray]], set2: list[tuple[str, np.ndarray]]
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Match images from two sets by filename.

    Args:
        set1: First set of images
        set2: Second set of images

    Returns:
        list of (filename, img1, img2) tuples for matched pairs
    """
    set1_dict = {name: img for name, img in set1}
    set2_dict = {name: img for name, img in set2}

    common_names = set(set1_dict.keys()) & set(set2_dict.keys())

    pairs = []
    for name in sorted(common_names):
        img1 = set1_dict[name]
        img2 = set2_dict[name]

        if img1.shape != img2.shape:
            print(f"Warning: {name} has different shapes, skipping")
            continue

        pairs.append((name, img1, img2))

    return pairs


def calculate_metrics_for_pair(
    img1: np.ndarray, img2: np.ndarray, metrics: list[str]
) -> dict[str, float]:
    """
    Calculate multiple metrics for a single image pair.

    Args:
        img1: First image
        img2: Second image
        metrics: list of metric names to calculate

    Returns:
        Dictionary mapping metric names to values
    """
    results = {}
    for metric_name in metrics:
        metric_func = get_metric_function(metric_name)
        if metric_func:
            results[metric_name] = metric_func(img1, img2)
    return results


def evaluate_image_pairs(
    pairs: list[tuple[str, np.ndarray, np.ndarray]], metrics: list[str]
) -> dict[str, dict[str, float]]:
    """
    Evaluate all image pairs with specified metrics.

    Args:
        pairs: list of (filename, img1, img2) tuples
        metrics: list of metric names

    Returns:
        Dictionary mapping filenames to metric dictionaries
    """
    results = {}
    for filename, img1, img2 in pairs:
        results[filename] = calculate_metrics_for_pair(img1, img2, metrics)
    return results


# =============================================================================
# Aggregation Functions
# =============================================================================


def aggregate_metrics(
    results: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Aggregate metrics across all images.

    Args:
        results: Dictionary of per-image metric results

    Returns:
        Dictionary of aggregated statistics for each metric
    """
    if not results:
        return {}

    metrics = list(next(iter(results.values())).keys())
    aggregated = {}

    for metric in metrics:
        values = [results[img][metric] for img in results if metric in results[img]]
        if values:
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

    return aggregated


# =============================================================================
# Output Functions
# =============================================================================


def save_results(results: dict, output_path: str) -> None:
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def print_summary(results: dict) -> None:
    """Print a summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Number of image pairs evaluated: {results['num_images']}")
    print("\nAggregated Metrics:")
    print("-" * 60)

    for metric, stats in results["aggregated"].items():
        print(f"\n{metric.upper()}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name:8s}: {value:.4f}")


# =============================================================================
# Main Pipeline Function
# =============================================================================


def run_evaluation_pipeline(
    set1_path: str,
    set2_path: str,
    processing_func: Callable[[np.ndarray], np.ndarray] = None,
    metrics: list[str] = ["mse", "psnr", "mae", "ssim"],
) -> dict:
    """
    Run the complete evaluation pipeline.

    Args:
        set1_path: Path to first image set
        set2_path: Path to second image set
        processing_func: Optional function to apply to images before evaluation
        metrics: list of metrics to calculate

    Returns:
        Dictionary containing per-image and aggregated results
    """
    # Load images
    set1, set2 = load_image_sets(set1_path, set2_path)

    # Apply processing function if provided
    if processing_func:
        print("Applying processing function to set 1...")
        set1 = apply_function_to_images(set1, processing_func)
        print("Applying processing function to set 2...")
        set2 = apply_function_to_images(set2, processing_func)

    # Match pairs
    pairs = match_image_pairs(set1, set2)

    # Evaluate
    print("Calculating metrics...")
    per_image_results = evaluate_image_pairs(pairs, metrics)

    # Aggregate
    aggregated_results = aggregate_metrics(per_image_results)

    return {
        "per_image": per_image_results,
        "aggregated": aggregated_results,
        "num_images": len(per_image_results),
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    pass
    # # Example processing function
    # def normalize_image(img: np.ndarray) -> np.ndarray:
    #     """Normalize image to 0-255 range."""
    #     img = img.astype(float)
    #     img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    #     return (img * 255).astype(np.uint8)

    # # Run evaluation
    # results = run_evaluation_pipeline(
    #     set1_path="./images/set1",
    #     set2_path="./images/set2",
    #     processing_func=normalize_image,  # Optional: None for no processing
    #     metrics=["mse", "psnr", "mae", "ssim"],
    # )

    # # Display and save results
    # print_summary(results)
    # save_results(results, "evaluation_results.json")
