from PIL import Image
from yoso.demo.app import start_tryon


if __name__ == "__main__":
    img = Image.open("data/samples/01_on-model.jpg")
    imgs = {"background": img}
    garment_img = Image.open("data/samples/01_still-life.jpg")
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
    image.save("data/output/image.png")
    mask_gray.save("data/output/mask.png")
