import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
HYCOCLIP_ROOT = REPO_ROOT / "hycoclip"
if str(HYCOCLIP_ROOT) not in sys.path:
    sys.path.append(str(HYCOCLIP_ROOT))

class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_model(args, backbone_name="resnet18_cub", full_model=False):
    if backbone_name.lower().startswith("hycoclip"):
        from hycoclip.hycoclip.config import LazyConfig, LazyFactory
        from hycoclip.hycoclip.utils.checkpointing import CheckpointManager

        config_path = getattr(args, "hycoclip_config", "hycoclip/configs/train_hycoclip_vit_b.py")
        checkpoint_path = getattr(args, "hycoclip_checkpoint", "hycoclip/hycoclip_vit_b.pth")
        device = torch.device(args.device)
        cfg = LazyConfig.load(config_path)
        model = LazyFactory.build_model(cfg, device=device)
        CheckpointManager(model=model).load(checkpoint_path)
        model.eval()
        backbone = model
        preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    elif "clip" in backbone_name:
        from hycoclip.hycoclip.config import LazyConfig, LazyFactory
        from hycoclip.hycoclip.utils.checkpointing import CheckpointManager

        config_path = getattr(args, "hycoclip_config", "hycoclip/configs/train_clip_vit_b.py")
        checkpoint_path = getattr(args, "hycoclip_checkpoint", "hycoclip/clip_vit_b.pth")
        device = torch.device(args.device)
        cfg = LazyConfig.load(config_path)
        model = LazyFactory.build_model(cfg, device=device)
        CheckpointManager(model=model).load(checkpoint_path)
        model.eval()
        backbone = model
        preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
    
    elif "meru" in backbone_name:
        from hycoclip.hycoclip.config import LazyConfig, LazyFactory
        from hycoclip.hycoclip.utils.checkpointing import CheckpointManager

        config_path = getattr(args, "hycoclip_config", "hycoclip/configs/train_meru_vit_b.py")
        checkpoint_path = getattr(args, "hycoclip_checkpoint", "hycoclip/meru_vit_b.pth")
        device = torch.device(args.device)
        cfg = LazyConfig.load(config_path)
        model = LazyFactory.build_model(cfg, device=device)
        CheckpointManager(model=model).load(checkpoint_path)
        model.eval()
        backbone = model
        preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    elif backbone_name == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, pretrained=True, root=args.out_dir)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
            ])
    
    elif backbone_name.lower() == "ham10000_inception":
        from .derma_models import get_derma_model
        model, backbone, model_top = get_derma_model(args, backbone_name.lower())
        preprocess = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])
    

    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone, preprocess
    else:
        return backbone, preprocess


