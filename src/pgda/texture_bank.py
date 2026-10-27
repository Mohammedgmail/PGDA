import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from diffusers import StableDiffusionPipeline

# Configuration
TEXTURE_DIR = "assets/texture_bank"

NUM_IMAGES_PER_PROMPT = 1000

# Material prompts mapping
PROMPTS = {
    "Heavy Rust": "A flat, high-resolution, seamless texture of heavy rusty iron, top-down view.",
    "Wet Ice": "A flat, high-resolution, seamless texture of smooth wet ice, top-down view.",
    "Coarse Sandpaper": "A flat, high-resolution, seamless texture of coarse black sandpaper, top-down view."
}


def generate_textures():
    os.makedirs(TEXTURE_DIR, exist_ok=True)

    # Hardware acceleration check for Apple M3 Max
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Stable Diffusion pipeline on device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)

    # Enable attention slicing to optimize unified memory usage on Apple Silicon
    pipe.enable_attention_slicing()

    for category, prompt in PROMPTS.items():
        category_dir = os.path.join(TEXTURE_DIR, category.replace(" ", "_"))
        os.makedirs(category_dir, exist_ok=True)

        print(f"Generating textures for category: '{category}'...")

        for i in range(NUM_IMAGES_PER_PROMPT):
            # Generate image with a baseline of 25 inference steps for speed vs quality
            image = pipe(prompt, num_inference_steps=25).images[0]

            # Save image
            filename = os.path.join(category_dir, f"texture_{i:04d}.png")
            image.save(filename)
            print(f"  Saved {filename}")

    print(f"\nTexture generation complete. Assets stored in: {TEXTURE_DIR}")


if __name__ == "__main__":
    generate_textures()