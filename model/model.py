import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from transformers import (
    MobileViTV2Config,
    MobileViTV2ForSemanticSegmentation,
    SegformerForSemanticSegmentation,
    SegformerConfig,
    ASTModel,
    ASTConfig,
)
from unet import UNet
from dataset import KEYS
from utils import get_n_params


def get_mobile_model():
    # Initializing a mobilevitv2-small style configuration
    num_channels = 2
    image_size = 128
    configuration = MobileViTV2Config(
        num_channels=num_channels,
        image_size=image_size,
        num_labels=len(KEYS),
        output_stride=8,
    )

    from lovely_tensors import monkey_patch, set_config

    monkey_patch()

    model = MobileViTV2ForSemanticSegmentation(configuration)

    a = torch.rand(3, 2, 128, 128)
    output = model(a)
    print(output)
    return model


#
# def get_model(variant):
#     # Initializing a mobilevitv2-small style configuration
#     num_channels = 2
#     image_size = 128
#
#     variants = {
#         "b0": {"depths": [2, 2, 2, 2], "hidden_sizes": [32, 64, 160, 256]},
#         "b1": {"depths": [2, 2, 2, 2], "hidden_sizes": [64, 128, 320, 512]},
#         "b2": {"depths": [3, 4, 6, 3], "hidden_sizes": [64, 128, 320, 512]},
#         "b3": {"depths": [3, 4, 18, 3], "hidden_sizes": [64, 128, 320, 512]},
#         "b4": {"depths": [3, 8, 27, 3], "hidden_sizes": [64, 128, 320, 512]},
#         "b5": {"depths": [3, 6, 40, 3], "hidden_sizes": [64, 128, 320, 512]},
#     }
#
#     configuration = SegformerConfig(
#         num_channels=num_channels,
#         depths=variants[variant]["depths"],
#         hidden_sizes=variants[variant]["hidden_sizes"],
#         num_labels=len(KEYS),
#     )
#
#     model = SegformerForSemanticSegmentation(configuration)
#     return model


class SpectrogramTransformer(nn.Module):
    def __init__(self, variant, num_classes, input_length):
        super().__init__()

        assert variant in ["tiny", "small", "base"]
        hidden_sizes = {
            "tiny": 192,
            "small": 384,
            "base": 768,
        }
        num_attention_heads = {
            "tiny": 3,
            "small": 6,
            "base": 12,
        }

        hidden_size = hidden_sizes[variant]
        num_attention_heads = num_attention_heads[variant]
        config = ASTConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_mel_bins=128,
            max_length=input_length,
        )
        self.num_classes = num_classes

        self.model = ASTModel(config)
        self.pooler = nn.AdaptiveAvgPool1d(input_length)
        self.adapter = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        time = x.shape[-1]
        # takes in (batch, num_channels, n_mel, time)
        # average over num_channels
        x = reduce(x, "b c m t -> b m t", "mean")
        x = rearrange(x, "b m t -> b t m")
        # (b, time', hidden_size)
        x = self.model(x).last_hidden_state
        # (b, hidden_size, time')
        x = rearrange(x, "b t c -> b c t")
        # (b, hidden_size, time)
        x = self.pooler(x)
        # (b, time, hidden_size)
        x = rearrange(x, "b c t -> b t c")
        # (b, time, num_classes)
        x = self.adapter(x)
        x = rearrange(x, "b t c -> b c t")
        assert x.shape[1] == self.num_classes
        assert x.shape[2] == time

        return x


def get_model(variant, input_length):
    num_classes = len(KEYS)
    return SpectrogramTransformer(variant, num_classes, input_length)


class SpectrogramUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.model = UNet(n_channels, n_classes * 2)

    def forward(self, x):
        # B, channels = 2, mel, time
        x = self.model(x)
        seg, sdf = torch.split(x, self.n_classes, dim=1)
        x = reduce(x, "b c m t -> b c t", "mean")

        seg, sdf = torch.split(x, self.n_classes, dim=1)
        assert seg.shape == sdf.shape

        return seg, sdf


def get_unet():
    num_channels = 2
    num_labels = len(KEYS)
    return SpectrogramUNet(num_channels, num_labels)


if __name__ == "__main__":
    # get_mobile_model()
    model = SpectrogramTransformer(30, 256)
    get_n_params(model)
    a = torch.rand(3, 2, 128, 256)
    b = model(a)
