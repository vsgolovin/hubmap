import pytorch_lightning as pl
import torch
from torch import nn, optim, Tensor
from torch.nn.functional import interpolate
from torchvision import models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import Normalize
from torchvision.utils import make_grid


def get_maskrcnn(pretrained: bool = True, trainable_backbone_layers: int = 3,
                 num_classes: int = 2, predictor_hidden_size: int = 256):
    # load model
    weights = "COCO_V1" if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    # freeze backbone layers
    assert 0 <= trainable_backbone_layers <= 5
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"
                       ][:trainable_backbone_layers]
    for name, parameter in model.backbone.body.named_parameters():
        if not any(name.startswith(prefix) for prefix in layers_to_train):
            parameter.requires_grad_(False)

    # replace heads
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       predictor_hidden_size,
                                                       num_classes)
    return model


class ResNetFeatures(nn.Module):
    def __init__(self, num_layers: int = 50, pretrained: bool = True):
        super().__init__()
        # check argument values
        nl_lst = [18, 34, 50, 101, 152]
        assert num_layers in nl_lst, f"num_layers should be one of {nl_lst}"
        weights = "DEFAULT" if pretrained else None

        # create model
        rn = models.get_model(f"resnet{num_layers}", weights=weights)
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

        # number of trainable layers (between 0 and 5)
        self.n_trainable_layers = 5

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def set_n_trainable_layers(self, n: int):
        assert 0 <= n <= 5
        to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:n]
        if n == 5:
            to_train.append("bn1")
        for name, parameter in self.named_parameters():
            in_list = any(name.startswith(layer) for layer in to_train)
            parameter.requires_grad_(in_list)
        self.n_trainable_layers = n


class ResNet50AutoEncoder(nn.Module):
    def __init__(self, pretrained: bool = True,
                 trainable_backbone_layers: int = 3,
                 dropout: float | None = None):
        super().__init__()
        self.encoder = ResNetFeatures(50, pretrained=pretrained)
        self.encoder.set_n_trainable_layers(trainable_backbone_layers)
        # optionally drop random final feature map channels
        assert dropout is None or 0.0 <= dropout < 1.0
        self.dropout = nn.Dropout2d(p=dropout) if dropout else nn.Identity()
        self.decoder = nn.ModuleList([
            self._dec_block(2048, 1024),
            self._dec_block(1024, 512),
            self._dec_block(512, 256),
            self._dec_block(256, 128),
            self._dec_block(128, 64),
            nn.Conv2d(64, 3, 1, 1)
        ])

    def forward(self, x: Tensor) -> Tensor:
        "Input shold be in [0, 1] range"
        x = self.encoder(x)
        x = self.dropout(x)  # identity by default
        for module in self.decoder:
            x = module(x)
        return x

    @staticmethod
    def _dec_block(c_in: int, c_out: int) -> nn.Module:
        return nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(c_in // 4, c_in // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_in // 4),
            nn.GELU(),
            nn.Conv2d(c_in // 4, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU()
        )


class MaskedAutoEncoder(pl.LightningModule):
    def __init__(self, masking_ratio: float, mask_patch_size: int = 32,
                 learning_rate: float = 1e3, weight_decay: float = 0.0,
                 pretrained: bool = True,
                 trainable_backbone_layers: int = 3):
        super().__init__()
        self.lr = learning_rate
        self.wd = weight_decay
        self.mratio = masking_ratio
        self.mask_xy = mask_patch_size
        self.save_hyperparameters()

        # network architecture
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        self.encoder = ResNetFeatures(50, pretrained=pretrained)
        self.set_trainable_backbone_layers(trainable_backbone_layers)
        self.mask_token = nn.Parameter(torch.zeros((1, 2048, 1, 1)))
        self.decoder = nn.ModuleList([
            self._dec_block(2048, 1024),
            self._dec_block(1024, 512),
            self._dec_block(512, 256),
            self._dec_block(256, 128),
            self._dec_block(128, 64),
            nn.Conv2d(64, 3, 1, 1)
        ])
        self.mask = None  # placeholder to share mask during forward pass

    @staticmethod
    def _dec_block(c_in: int, c_out: int) -> nn.Module:
        return nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(c_in // 4, c_in // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_in // 4),
            nn.GELU(),
            nn.Conv2d(c_in // 4, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU()
        )

    def _create_mask(self, b: int, h: int, w: int) -> Tensor:
        # randomly select masking regions
        assert h % self.mask_xy == 0 and w % self.mask_xy == 0
        hm, wm = h // self.mask_xy, w // self.mask_xy
        # number of masked squares
        n = int(round(self.mratio * hm * wm))
        # indices of masked squares for every image
        inds = torch.rand((b, hm * wm)).argsort(1)[:, :n]
        # convert indices to boolean mask
        mask = torch.ones((b, hm * wm), dtype=torch.float32)
        mask[torch.arange(b).repeat_interleave(n), inds.ravel()] = 0.0
        # reshape and upscale mask
        mask = mask.reshape((b, 1, hm, wm))
        mask = interpolate(mask, size=(h, w), mode="nearest")
        self.mask = mask.repeat((1, 3, 1, 1)).to(self.device)

    def set_trainable_backbone_layers(self, n: int):
        self.encoder.set_n_trainable_layers(n)

    def _encode(self, x: Tensor) -> Tensor:
        x = self.normalize(x)
        x = self.encoder(x)
        mtoken = self.mask_token.repeat((x.size(0), 1, x.size(2), x.size(3)))
        mask = self.mask[:, 0:1, ::self.mask_xy, ::self.mask_xy]
        return x * mask + mtoken * (1 - mask)

    def _decode(self, x: Tensor) -> Tensor:
        for module in self.decoder.children():
            x = module(x)
        return x

    def forward(self, x: Tensor):
        b, _, h, w = x.shape
        self._create_mask(b, h, w)  # creates self.mask
        y = self._encode(x)
        return self._decode(y)

    def training_step(self, batch, batch_idx):
        input = batch
        pred = self.forward(input)
        mask = 1 - self.mask
        loss = (mask * (pred - input)**2).sum() / mask.sum()
        self.log("train_loss", loss, prog_bar=True)
        self.mask = None  # delete just in case
        return loss

    def validation_step(self, batch, batch_idx):
        # compute loss
        input = batch
        pred = self.forward(input)
        mask = 1 - self.mask
        loss = (mask * (pred - input)**2).sum() / mask.sum()
        self.log("val_loss", loss, prog_bar=True)
        # log images
        if batch_idx == 0:
            img_stack = torch.cat(
                [input, input * self.mask, pred.clip(0, 1),
                 (input * self.mask + pred * (1 - self.mask)).clip(0, 1)],
                dim=0
            )
            grid = make_grid(img_stack, nrow=len(input))
            self.logger.experiment.add_image("outputs", grid,
                                             self.current_epoch)
        # delete mask
        self.mask = None

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
