import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from controller import (
    CHISController,
    apply_fss_noise,
    build_texture_bank,
    encode_with_vae,
    generate_structure_guide,
    process_masks,
    segment_reference_mask,
)
from model_zoo import UNI2FeatureExtractor, load_pixcell


def load_binary_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    return (mask > 0).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CHIS on one reference image and one target mask."
    )
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--target-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("chis_result.png"))
    parser.add_argument(
        "--pseudo-mask-output", type=Path, default=Path("reference_pseudo_mask.png")
    )
    parser.add_argument(
        "--guide-output", type=Path, default=Path("structure_guide.png")
    )
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--inject-step", type=int, default=40)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    return parser.parse_args()


def main():
    args = parse_args()

    uni_model = UNI2FeatureExtractor()
    pipeline, vae = load_pixcell()

    reference_image = np.array(Image.open(args.reference_image).convert("RGB"))
    reference_pseudo_mask = segment_reference_mask(reference_image)
    Image.fromarray((reference_pseudo_mask * 255).astype(np.uint8)).save(
        args.pseudo_mask_output
    )

    target_mask = load_binary_mask(args.target_mask)
    dr_mask, dt_mask, sigma = process_masks(reference_pseudo_mask, target_mask)

    structure_guide = generate_structure_guide(
        reference_image,
        reference_pseudo_mask,
        target_mask,
        n_bins=100,
    )
    Image.fromarray(structure_guide).save(args.guide_output)

    nucleus_bank, bg_bank = build_texture_bank(vae, reference_image, dr_mask)
    uni_embeds = uni_model.extract_uni_embeddings(reference_image)
    negative_uni_embeds = pipeline.get_unconditional_embedding(uni_embeds.shape[0]).to(
        uni_embeds.device,
        dtype=uni_embeds.dtype,
    )

    guide_latents = encode_with_vae(vae, structure_guide)
    fss_generator = torch.Generator(device=guide_latents.device).manual_seed(args.seed)
    initial_latents = apply_fss_noise(guide_latents, generator=fss_generator)

    controller = CHISController(
        scheduler=pipeline.scheduler,
        sigma=sigma,
        nucleus_bank=nucleus_bank,
        bg_bank=bg_bank,
        soft_mask=dt_mask,
        guide_latents=guide_latents,
        seed=args.seed,
        inject_step=args.inject_step,
    )

    result = pipeline(
        num_inference_steps=args.num_inference_steps,
        uni_embeds=uni_embeds,
        negative_uni_embeds=negative_uni_embeds,
        latents=initial_latents,
        callback=controller,
        callback_steps=1,
        guidance_scale=args.guidance_scale,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(args.output)


def _entrypoint():
    main()


if __name__ == "__main__":
    _entrypoint()
