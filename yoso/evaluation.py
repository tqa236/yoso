from PIL import Image
from yoso.demo.app import start_tryon
from pathlib import Path


if __name__ == "__main__":
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
