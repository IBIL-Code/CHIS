import torch
from diffusers import AutoencoderKL, DiffusionPipeline

PIXCELL_MODEL_ID = "StonyBrook-CVLab/PixCell-256"
PIXCELL_PIPELINE_ID = "StonyBrook-CVLab/PixCell-pipeline"
SD3_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_sd3_vae(torch_dtype: torch.dtype) -> AutoencoderKL:
    return AutoencoderKL.from_pretrained(
        SD3_MODEL_ID, subfolder="vae", torch_dtype=torch_dtype
    )


def load_pixcell_pipeline(
    device: str | torch.device | None = None, torch_dtype: torch.dtype = torch.float16
) -> tuple[DiffusionPipeline, AutoencoderKL]:
    sd3_vae = _load_sd3_vae(torch_dtype=torch_dtype)
    pipeline = DiffusionPipeline.from_pretrained(
        PIXCELL_MODEL_ID,
        vae=sd3_vae,
        custom_pipeline=PIXCELL_PIPELINE_ID,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    pipeline.to(_resolve_device(device))
    return (pipeline, sd3_vae)


def load_pixcell(
    device: str | torch.device | None = None, torch_dtype: torch.dtype = torch.float16
) -> tuple[DiffusionPipeline, AutoencoderKL]:
    return load_pixcell_pipeline(device=device, torch_dtype=torch_dtype)
