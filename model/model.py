from transformers import (
    MobileViTV2Config,
    MobileViTV2ForSemanticSegmentation,
    SegformerForSemanticSegmentation,
    SegformerConfig,
)
from dataset import KEYS

# def get_model():
#     # Initializing a mobilevitv2-small style configuration
#     num_channels = 2
#     image_size = 128
#     configuration = MobileViTV2Config(num_channels = num_channels, image_size = image_size, num_labels = len(KEYS), output_stride=1)
#
#     model = MobileViTV2ForSemanticSegmentation(configuration)
#     return model


def get_model():
    # Initializing a mobilevitv2-small style configuration
    num_channels = 2
    image_size = 128

    variants = {
        "b0": {"depths": [2, 2, 2, 2], "hidden_sizes": [32, 64, 160, 256]},
        "b1": {"depths": [2, 2, 2, 2], "hidden_sizes": [64, 128, 320, 512]},
        "b2": {"depths": [3, 4, 6, 3], "hidden_sizes": [64, 128, 320, 512]},
        "b3": {"depths": [3, 4, 18, 3], "hidden_sizes": [64, 128, 320, 512]},
        "b4": {"depths": [3, 8, 27, 3], "hidden_sizes": [64, 128, 320, 512]},
        "b5": {"depths": [3, 6, 40, 3], "hidden_sizes": [64, 128, 320, 512]},
    }

    variant = "b0"

    configuration = SegformerConfig(
        num_channels=num_channels,
        depths=variants[variant]["depths"],
        hidden_sizes=variants[variant]["hidden_sizes"],
        num_labels=len(KEYS),
    )

    model = SegformerForSemanticSegmentation(configuration)
    return model
