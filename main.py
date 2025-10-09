import os
import argparse
from config import METHODS
from pipeline import StableDiffusionGenerator
from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser(description="A program for creating greenback images using diverse techniques")
    parser.add_argument("--method", type=str, choices=METHODS, default="gbp", help="Choose the generation technique (e.g., gbp, tkg)")
    parser.add_argument("--device", type=int, default=0, help="Index of the CUDA GPU to be used")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for random number generation to ensure reproducibility")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cx", type=float, default=0, help="Center x coordinate within [-1, 1]")
    parser.add_argument("--cy", type=float, default=0, help="Center y coordinate within [-1, 1]")
    parser.add_argument("--sd", type=float, default=.5, help="Standard deviation")
    parser.add_argument("--shift_ratio", type=float, default=.11, help="Standard deviation")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    device = f"cuda:{args.device}"
    generator = StableDiffusionGenerator(method=args.method, device=device, seed=args.seed)

    base_prompts = [
        "young woman with virtual reality glasses sitting in armchair",
        "yellow lemon and slice",
        "gray cat british shorthair",
        "vintage golden trumpet making music concept",
        "set of many business people"
    ]
    active_prompts = ', realistic, photo-realistic, 4K, high resolution, high quality'
    negative_prompts = 'background, character, cartoon, anime, text, fail, low resolution'

    prompts, generated_images, masks = generator.generate_images(
        base_prompts=base_prompts,
        active_prompts=active_prompts,
        negative_prompts=negative_prompts,
        steps=args.steps,
        center_x=args.cx,
        center_y=args.cy,
        standard_dev=args.sd,
        shift_ratio=args.shift_ratio,
    )

    output_dir = os.path.join(args.output_dir, "sdxl", args.method, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)

    for prompt, img in zip(prompts, generated_images):
        filename = f"{prompt.replace(' ', '_')[:10]}.png"
        img.save(os.path.join(output_dir, filename))
    save_image(masks, "mask.jpg")


if __name__ == "__main__":
    main()
