import torch
import torch.nn as nn

from .clip import clip

CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 1024,
    "ViT-B/16": 768,
}


class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, truncate_layer=None):
        super().__init__()

        self.model, self.preprocess = clip.load(
            name, device="cpu"
        )  # self.preprecess will not be used during training, which is handled in Dataset class
        self.vit = self.model.visual

        self.out_channels = CHANNELS[name]
        self.num_layers = 24 if (name == "ViT-L/14") else 12

    def forward(self, x, return_feature=False):
        # features = self.model.encode_image(x) # x: NCHW features: N, 768(D)
        x = self.vit.conv1(x)  # shape = [*, width, grid, grid]
        N, C, H, W = x.shape
        x = x.reshape(N, C, -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.vit.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.vit.positional_embedding.to(x.dtype)
        x = self.vit.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        cls_features = []
        for i, layer in enumerate(self.vit.transformer.resblocks.children()):
            # if i == self.truncate_layer:
            #     break
            x = layer(x)
            cls_features.append(x[0])
        cls_features = torch.stack(cls_features, dim=0)  # 24, 128, 1024

        return cls_features
