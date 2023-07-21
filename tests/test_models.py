import pytest
import torch
from src.models import ResUNet, PSDecoderBlock


@pytest.mark.parametrize("c_in,c_out", [(64, 32), (64, 16), (128, 128)])
@pytest.mark.parametrize("sc", [True, False])
@pytest.mark.parametrize("scale", [2, 4])
def test_decoder_block(c_in: int, c_out: int, sc: bool, scale: int):
    dec = PSDecoderBlock(c_in, c_out, sc, upscale_factor=scale)
    inp = torch.randn((1, c_in, 7, 7))
    if sc:
        skip_connection = torch.randn(
            (1, c_in // scale**2, 7 * scale, 7 * scale))
    else:
        skip_connection = None
    with torch.no_grad():
        out = dec(inp, skip_connection)
    assert out.shape == torch.Size([1, c_out, 7 * scale, 7 * scale])


@pytest.mark.parametrize("rn_layers", (50, 101, 152))
@pytest.mark.parametrize("c_out", (1, 3))
@pytest.mark.parametrize("b", (1, 8))
def test_resunet(rn_layers: int, c_out: int, b: int):
    inp = torch.rand((b, 3, 224, 224))
    unet = ResUNet(pretrained=False, resnet_layers=rn_layers,
                   out_channels=c_out)
    with torch.no_grad():
        out = unet(inp)
    assert out.shape == torch.Size([b, c_out, 224, 224])
