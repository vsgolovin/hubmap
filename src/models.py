from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_maskrcnn(pretrained: bool = True, trainable_backbone_layers: int = 3,
                 num_classes: int = 2, predictor_hidden_size: int = 256):
    # load model
    weights = "COCO_V1" if pretrained else None
    assert 0 <= trainable_backbone_layers <= 5
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers
    )
    # replace heads
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       predictor_hidden_size,
                                                       num_classes)
    return model
