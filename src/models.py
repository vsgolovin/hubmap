import pytorch_lightning as pl
import torch
from torch import nn, optim, Tensor
from torch.nn.functional import interpolate
from torchvision import models
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import Normalize
from torchvision.utils import make_grid


def get_maskrcnn(pretrained: bool = True, trainable_backbone_layers: int = 3,
                 num_classes: int = 2, v2: bool = False,
                 predictor_hidden_size: int = 256) -> detection.FasterRCNN:
    # load model
    weights = "DEFAULT" if pretrained else None
    if v2:
        constructor = detection.maskrcnn_resnet50_fpn_v2
    else:
        constructor = detection.maskrcnn_resnet50_fpn
    model = constructor(weights=weights, weights_backbone=weights)

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
    def __init__(self, pretrained: bool = True, num_layers: int = 50):
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
        self.encoder = ResNetFeatures(pretrained=pretrained, num_layers=50)
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


class PSDecoderBlock(nn.Module):
    "Lightweight decoder block with PixelShuffle upsampling"
    def __init__(self, in_channels: int, out_channels: int,
                 skip_connection: bool = True, upscale_factor: int = 2):
        super().__init__()
        self.upsample = nn.PixelShuffle(upscale_factor=upscale_factor)
        conv_channels = in_channels // upscale_factor**2
        if skip_connection:
            conv_channels *= 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.GELU(),
            nn.Conv2d(conv_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.skip_connection = skip_connection  # boolean flag

    def forward(self, x: Tensor, skip_con: Tensor | None = None) -> Tensor:
        x = self.upsample(x)
        if skip_con is not None:
            x = torch.cat([x, skip_con], dim=1)
        return self.conv_block(x)


class ResUNet(nn.Module):
    "U-Net with ResNet backbone"
    def __init__(self, pretrained: bool = True, resnet_layers: int = 50,
                 out_channels: int = 1, trainable_backbone_layers: int = 3):
        super().__init__()
        assert resnet_layers >= 50, "ResNet18/34 not supported"
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        self.encoder = ResNetFeatures(pretrained=pretrained,
                                      num_layers=resnet_layers)
        self.set_trainable_backbone_layers(trainable_backbone_layers)
        self.sc_convs = nn.ModuleDict({
            "relu": self._skip_con(64, 64),
            "layer1": self._skip_con(256, 128),
            "layer2": self._skip_con(512, 256),
            "layer3": self._skip_con(1024, 512),
        })
        self.decoder = nn.ModuleList([
            PSDecoderBlock(2048, 1024),
            PSDecoderBlock(1024, 512),
            PSDecoderBlock(512, 256),
            PSDecoderBlock(256, 128),
            PSDecoderBlock(128, 32, skip_connection=False)
        ])
        self.head = nn.Conv2d(32, out_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        # encode input and collect intermediate feature maps
        x = self.normalize(x)
        skip_connections = [None]
        for name, module in self.encoder.named_children():
            x = module(x)
            if name in self.sc_convs.keys():
                skip_connections.append(self.sc_convs[name](x))
        # decode collected feature maps
        for module in self.decoder:
            sc = skip_connections.pop()
            x = module(x, sc)
        # final transformation
        return self.head(x)

    @staticmethod
    def _skip_con(c_in: int, c_out: int):
        "Skip connection from encoder to decoder"
        if c_in == c_out:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),
            nn.GELU()
        )

    def set_trainable_backbone_layers(self, n: int):
        self.encoder.set_n_trainable_layers(n)


class UpsampleDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, 1, 1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: Tensor, skip_con: Tensor | None) -> Tensor:
        x = self.upsample(x)
        if skip_con is not None:
            x = torch.cat([x, skip_con], dim=1)
        return self.conv_block(x)


class EfficientUNet(nn.Module):
    def __init__(self, pretrained: bool = True, out_channels: int = 1,
                 trainable_backbone_layers: int = 0):
        super().__init__()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        weights = "DEFAULT" if pretrained else None
        self.encoder = models.efficientnet_b5(weights=weights).features[:-1]
        self.set_trainable_backbone_layers(trainable_backbone_layers)
        self.decoder = nn.ModuleList([
            UpsampleDecoderBlock(512, 176, 256),
            UpsampleDecoderBlock(256, 64, 128),
            UpsampleDecoderBlock(128, 40, 64),
            UpsampleDecoderBlock(64, 24, 64),
            UpsampleDecoderBlock(64, 0, 32)
        ])
        self.head = nn.Conv2d(32, out_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.normalize(x)
        features = [None]
        for i, module in enumerate(self.encoder.children()):
            x = module(x)
            if i in [1, 2, 3, 5]:
                features.append(x)
        for module in self.decoder.children():
            x = module(x, features.pop())
        return self.head(x)

    def set_trainable_backbone_layers(self, n: int):
        assert 0 <= n <= 8
        for i, module in enumerate(self.encoder.children()):
            module.requires_grad_(i > (7 - n))


class DenoisingAutoEncoder(pl.LightningModule):
    def __init__(self, lr: float, weight_decay: float = 0.0,
                 pretrained: bool = True, resnet_layers: int = 50,
                 trainable_backbone_layers: int = 3, noise_eps: float = 0.4):
        super().__init__()
        self.lr = lr
        self.wd = weight_decay
        self.model = ResUNet(
            pretrained=pretrained,
            resnet_layers=resnet_layers,
            out_channels=1,
            trainable_backbone_layers=trainable_backbone_layers
        )
        self.noise_eps = noise_eps
        self.generator = None
        self.save_hyperparameters()

    def forward(self, x):
        "Input should already be noisy"
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr,
                          weight_decay=self.wd)

    def training_step(self, batch, batch_idx):
        # train model to predict added noise
        noise = torch.randn(
            size=(batch.size(0), 1, batch.size(2), batch.size(3)),
            device=self.device
        ) * self.noise_eps
        inp = batch + noise
        out = self(inp)
        loss = ((out - noise)**2).mean()
        self.log("train/loss", loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(2147483647)

    def validation_step(self, batch, batch_idx):
        # same as training, except same noise at every epoch
        noise = torch.randn(
            size=(batch.size(0), 1, batch.size(2), batch.size(3)),
            generator=self.generator,
            device=self.device
        ) * self.noise_eps
        inp = batch + noise
        out = self(inp)
        loss = ((out - noise)**2).mean()
        self.log("val/loss", loss, prog_bar=True)
        # save first batch of results
        if batch_idx == 0:
            img_stack = torch.cat(
                [batch, inp.clip(0, 1), (inp - out).clip(0, 1)],
                dim=0
            )
            grid = make_grid(img_stack, nrow=len(batch))
            self.logger.experiment.add_image("denoising_results", grid,
                                             self.current_epoch)
