import os
from .clip_encoder import CLIPVisionTower
from .imagebind import ImageBindWrapper
from .open_clip_encoder import OpenCLIPVisionTower
from .hf_vision import HFVisionTower
from .siglip_encoder import SigLipVisionTower
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2

import yaml
from .whale.init_model import init_model
# from .eva_clip.eva_clip_encoder import EvaClipVisionTower
# from .dev_eva_clip.eva_vit import EvaViTWrapper


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, "s2", False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf:"):
        return HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower in ["imagebind_huge"]:
        return ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("open_clip_hub"):
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # elif "internal-eva" in vision_tower.lower() or "eva02" in vision_tower.lower():
    #     return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # elif vision_tower in ["EVA-CLIP-8B", "EVA-CLIP-8B-plus"]:
    #     return EvaViTWrapper(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")


def build_audio_encoder(audio_encoder_config, **kwargs):
    with open(audio_encoder_config.mm_audio_encoder + "/train.yaml", "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    configs["cmvn_file"] = audio_encoder_config.mm_audio_encoder + "/global_cmvn"

    #    configs['model_conf']['freeze_encoder'] = audio_encoder_config.freeze_audio_encoder
    #    configs['model_conf']['freeze_adpter'] = audio_encoder_config.freeze_audio_encoder_adapter
    configs["model_conf"]["freeze_encoder"] = getattr(
        audio_encoder_config, "freeze_audio_encoder", True
    )
    configs["model_conf"]["freeze_adpter"] = getattr(
        audio_encoder_config, "freeze_audio_encoder_adapter", True
    )

    return init_model(configs)
