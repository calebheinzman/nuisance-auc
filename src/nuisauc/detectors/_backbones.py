"""Frozen image encoders that return spatial tokens (torch required).

Imported lazily by `nuisauc.detectors.vlm.encode`; importing this module
imports torch and torchvision.

    extract_fm_spatial  foundation models (DINOv2, Phikon, PLIP), token grid
                        average-pooled to out_grid x out_grid
    resnet_tokens       ResNet-18 with an edge-aware stem, layer3 tokens
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tvt

log = logging.getLogger(__name__)

# ImageNet statistics; DINOv2, Phikon and PLIP all normalise this way.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FM_SPECS = {
    # name: (loader_kind, hf_or_timm_id, input_size, patch_size)
    "dinov2_vits14": ("timm", "vit_small_patch14_dinov2.lvd142m", 224, 14),
    "dinov2_vitb14": ("timm", "vit_base_patch14_dinov2.lvd142m", 224, 14),
    "phikon": ("hf_vit", "owkin/phikon", 224, 16),
    "phikon_v2": ("hf_vit", "owkin/phikon-v2", 224, 14),
    "plip": ("hf_clip", "vinid/plip", 224, 32),
}


class ImageDataset(Dataset):
    """Items -> transformed image tensors."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.transform(self.items[i]["image"])


def build_transform(size: int):
    """Bicubic resize to size x size, tensor, ImageNet normalisation."""
    return tvt.Compose([
        tvt.Resize((size, size), interpolation=tvt.InterpolationMode.BICUBIC),
        tvt.ToTensor(),
        tvt.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(name: str, device: str, size_override: int | None = None):
    """Load a frozen foundation model: (model, forward_tokens, size, patch)."""
    kind, model_id, size, patch = FM_SPECS[name]
    if size_override is not None:
        if size_override % patch:
            raise ValueError(f"{name}: input size {size_override} is not a "
                             f"multiple of the patch size {patch}")
        size = size_override
    if kind == "timm":
        import timm
        # DINOv2 timm checkpoints default to img_size=518; interpolate the
        # position embeddings down to our input size instead.
        model = timm.create_model(model_id, pretrained=True, num_classes=0,
                                  img_size=size)

        def forward_tokens(x):
            # (B, n_prefix + N, C): drop the CLS and any register tokens.
            t = model.forward_features(x)
            return t[:, getattr(model, "num_prefix_tokens", 1):, :]
    elif kind == "hf_vit":
        from transformers import AutoModel
        try:
            model = AutoModel.from_pretrained(model_id, add_pooling_layer=False)
        except TypeError:  # arch without a pooler argument (e.g. phikon-v2)
            model = AutoModel.from_pretrained(model_id)

        def forward_tokens(x):
            return model(pixel_values=x,
                         interpolate_pos_encoding=True).last_hidden_state[:, 1:, :]
    elif kind == "hf_clip":
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained(model_id)

        def forward_tokens(x):
            return model(pixel_values=x).last_hidden_state[:, 1:, :]
    else:
        raise ValueError(kind)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, forward_tokens, size, patch


def extract_fm_spatial(name: str, items: list, batch_size: int = 32,
                       device: str = "cuda", out_grid: int = 8,
                       dtype=np.float16, verbose: bool = True,
                       input_size: int | None = None) -> np.ndarray:
    """Frozen spatial tokens as (N, out_grid**2, C)."""
    if name not in FM_SPECS:
        raise ValueError(f"unknown foundation model {name!r}; "
                         f"choose from {sorted(FM_SPECS)}")
    model, forward_tokens, size, patch = load_model(name, device, input_size)
    grid = size // patch
    loader = DataLoader(ImageDataset(items, build_transform(size)),
                        batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=True)
    chunks = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) \
        if device == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    with torch.no_grad(), autocast:
        for bi, x in enumerate(loader):
            t = forward_tokens(x.to(device, non_blocking=True))
            b, n, c = t.shape
            if n != grid * grid:
                raise RuntimeError(
                    f"{name}: expected {grid*grid} patch tokens, got {n}")
            t = t.transpose(1, 2).reshape(b, c, grid, grid).float()
            if grid != out_grid:
                t = F.adaptive_avg_pool2d(t, out_grid)
            chunks.append(t.flatten(2).transpose(1, 2).cpu().numpy().astype(dtype))
            if verbose and (bi + 1) % 50 == 0:
                log.info("%s: %d/%d", name, (bi + 1) * batch_size, len(items))
    del model
    torch.cuda.empty_cache()
    return np.concatenate(chunks)


# ------------------------------------------------------------ ResNet-18
class EdgeAwareStem(nn.Module):
    """3x3-conv stem without max pooling (preserves thin structures)."""

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18MorphSafe(nn.Module):
    """ResNet-18 with the edge-aware stem; layers 1-4 from ImageNet weights.

    The stem has no ImageNet counterpart and keeps PyTorch's default random
    initialisation, so its tokens depend on the torch RNG state (legacy quirk).
    """

    def __init__(self, pretrained=True, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.stem = EdgeAwareStem(3, 64)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_weights(pretrained)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self, pretrained):
        if pretrained:
            try:
                import torchvision.models as models
                resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                self._copy_layer_weights(self.layer1, resnet18.layer1)
                self._copy_layer_weights(self.layer2, resnet18.layer2)
                self._copy_layer_weights(self.layer3, resnet18.layer3)
                self._copy_layer_weights(self.layer4, resnet18.layer4)
                log.info("Loaded ImageNet-pretrained weights for ResNet-18 layers")
            except Exception as e:
                log.warning("Could not load pretrained weights: %s", e)
                self._init_kaiming()
        else:
            self._init_kaiming()

    def _copy_layer_weights(self, target_layer, source_layer):
        for target_block, source_block in zip(target_layer, source_layer):
            if target_block.conv1.weight.shape == source_block.conv1.weight.shape:
                target_block.conv1.weight.data.copy_(source_block.conv1.weight.data)
            if hasattr(source_block, 'bn1'):
                target_block.bn1.weight.data.copy_(source_block.bn1.weight.data)
                target_block.bn1.bias.data.copy_(source_block.bn1.bias.data)
                target_block.bn1.running_mean.copy_(source_block.bn1.running_mean)
                target_block.bn1.running_var.copy_(source_block.bn1.running_var)
            if target_block.conv2.weight.shape == source_block.conv2.weight.shape:
                target_block.conv2.weight.data.copy_(source_block.conv2.weight.data)
            if hasattr(source_block, 'bn2'):
                target_block.bn2.weight.data.copy_(source_block.bn2.weight.data)
                target_block.bn2.bias.data.copy_(source_block.bn2.bias.data)
                target_block.bn2.running_mean.copy_(source_block.bn2.running_mean)
                target_block.bn2.running_var.copy_(source_block.bn2.running_var)
            if target_block.downsample is not None and source_block.downsample is not None:
                if target_block.downsample[0].weight.shape == source_block.downsample[0].weight.shape:
                    target_block.downsample[0].weight.data.copy_(source_block.downsample[0].weight.data)
                    target_block.downsample[1].weight.data.copy_(source_block.downsample[1].weight.data)
                    target_block.downsample[1].bias.data.copy_(source_block.downsample[1].bias.data)
                    target_block.downsample[1].running_mean.copy_(source_block.downsample[1].running_mean)
                    target_block.downsample[1].running_var.copy_(source_block.downsample[1].running_var)

    def _init_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def load_resnet18_morphsafe(ckpt_path: str | None = None,
                            pretrained: bool = True) -> ResNet18MorphSafe:
    """ResNet18MorphSafe, optionally overwritten by a checkpoint's state dict."""
    model = ResNet18MorphSafe(pretrained=pretrained, embed_dim=512)
    if ckpt_path is not None and Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        log.info("Loaded checkpoint from %s", ckpt_path)
    return model


def eval_transforms(size: int = 81):
    """Bicubic resize, centre crop, tensor, ImageNet normalisation."""
    return tvt.Compose([
        tvt.Resize(size, interpolation=tvt.InterpolationMode.BICUBIC),
        tvt.CenterCrop(size),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TargetItemDataset(Dataset):
    """Items -> (transformed RGB tensor, label)."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        d = self.items[i]
        img = d["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        return self.transform(img.convert("RGB")), int(
            d.get("_resolved_label", d.get("label", 0)))


def resnet_tokens(ckpt: str | None, items: list, size: int, batch_size: int,
                  device: str, layer: str = "layer3") -> np.ndarray:
    """(n, h*w, c) float32 tokens from one ResNet18MorphSafe layer."""
    bb = load_resnet18_morphsafe(ckpt_path=ckpt)
    loader = DataLoader(TargetItemDataset(items, eval_transforms(size=size)),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    acts = {}
    h = getattr(bb, layer).register_forward_hook(
        lambda _m, _i, out: acts.__setitem__("v", out))
    bb.eval().to(device)
    # Fill a preallocated (n, h*w, c) array; the transpose is done per batch.
    out = None
    pos = 0
    with torch.no_grad():
        for images, _ in loader:
            bb(images.to(device, non_blocking=True))
            v = acts["v"]
            b, c, hh, ww = v.shape
            if out is None:
                out = np.empty((len(items), hh * ww, c), dtype=np.float32)
            out[pos:pos + b] = (v.permute(0, 2, 3, 1)
                                 .reshape(b, hh * ww, c)
                                 .float().cpu().numpy())
            pos += b
    h.remove()
    del bb
    torch.cuda.empty_cache()
    return out[:pos]
