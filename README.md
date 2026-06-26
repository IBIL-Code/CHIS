

<p align="center">
  <img src="fig/logo.png" width="400">
</p>

# [ MICCAI 2026 ] **Controllable Histopathology Image Synthesis with Training-free Structural Initialization and Textural Modulation**.
## Start Here

### Environment

CHIS uses a lightweight [uv](https://docs.astral.sh/uv/) environment. Python, PyTorch, Diffusers, Transformers, timm, OpenCV, and the remaining runtime dependencies are pinned through `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

The default environment targets Python 3.12 and CUDA-enabled PyTorch. In our tests, single-image inference runs on a consumer NVIDIA RTX 5060 Ti GPU, making CHIS practical without server-grade hardware.

CHIS loads pretrained models from Hugging Face at runtime. The first run will download the required weights into your local Hugging Face cache.

## Pretrained Weights

The model loaders are in `model_zoo/`:

```text
model_zoo/
  pixcell.py   # PixCell diffusion pipeline
  uni.py       # UNI2-h feature extractor
```

The following pretrained models are loaded automatically:

| Component | Hugging Face source | Access |
| --- | --- | --- |
| PixCell weights | [StonyBrook-CVLab/PixCell-256](https://huggingface.co/StonyBrook-CVLab/PixCell-256) | Public model page |
| PixCell custom pipeline | [StonyBrook-CVLab/PixCell-pipeline](https://huggingface.co/StonyBrook-CVLab/PixCell-pipeline) | Public model page |
| SD3 VAE | [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | Requires accepting the Stability AI license/access terms |
| UNI2-h feature extractor | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | Requires requesting/accepting access with a Hugging Face account |

Before running the code, open the gated model pages above and complete the required access steps for **SD3.5 Large** and **UNI2-h**. PixCell also depends on these models at runtime, so the loaders will fail if either gated dependency is not available to your Hugging Face account.

After access is approved, log in from the command line. Hugging Face now recommends the `hf` CLI; with uv, you can run it directly as:

```bash
uvx hf auth login
```

If you already have the standalone `hf` CLI installed, `hf auth login` is equivalent. If the weights are already cached locally, the loaders will reuse them. If they are not cached, the loaders require internet access for the first run.

You can control where Hugging Face stores downloaded weights with:

```bash
export HF_HOME=/path/to/huggingface_cache
```

## Usage

Run CHIS on one reference image and one target mask:

```bash
uv run python image_generation.py \
  --reference-image path/to/reference.png \
  --target-mask path/to/target_mask.png \
  --output chis_result.png
```

The reference pseudo mask is segmented automatically from the reference image. Target masks may be stored as `0/1` or `0/255`; they are normalized and binarized with `--mask-threshold 0.5` by default. Intermediate files are not saved by default; pass `--save-intermediates` to write `reference_pseudo_mask.png` and `structure_guide.png` next to the output image.

```python
from model_zoo import UNI2FeatureExtractor, load_pixcell_pipeline

pipeline, vae = load_pixcell_pipeline()
feature_extractor = UNI2FeatureExtractor()
```


