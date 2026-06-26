import torch
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class UNI2FeatureExtractor:
    def __init__(self, device: str | torch.device | None = None):
        self.device = torch.device(
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-05,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )
        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )
        self.model.eval()
        self.model.to(self.device)

    def extract_embeddings(self, image) -> torch.Tensor:
        image = Image.fromarray(image)
        model_input = self.transform(image).unsqueeze(0)
        with torch.inference_mode():
            embedding = self.model(model_input.to(self.device))
        return embedding.unsqueeze(1).half()

    def extract_uni_embeddings(self, image) -> torch.Tensor:
        return self.extract_embeddings(image)


UniModelLoader = UNI2FeatureExtractor
