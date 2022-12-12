import torch
from torch import nn
from torchvision.models import AlexNet


class myFCN(nn.Module):
    """
    A fully-convolutional network with a feature extractor modeled after AlexNet.
    Original 'self.features' found at 
        https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py.
    """
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )  # Output size H / 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )  # Output (H-1)/2
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Shortcut upsampling connections
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(384, 1, kernel_size=1)
        )
        self.up4 = nn.Sequential(

            nn.Conv2d(384, 1, kernel_size=1)
        )

        # Final Concatenation part
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, num_classes, 1),
        ]
        self.head = nn.Sequential(
            layers
        )

    def forward(self, x: torch.Tensor) -> torch.tensor:
        x2 = self.double_conv(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.head(x5)

        # Shortcuts
        up3 = self.up3(x3)
        up4 = self.up4(x4)

        # Consolidate shortcuts and FCN head output
        out = x6.sum(up3).sum(up4)
        return out

# class FCN(_SimpleSegmentationModel):
#     """
#     Implements FCN model from
#     `"Fully Convolutional Networks for Semantic Segmentation"
#     <https://arxiv.org/abs/1411.4038>`_.

#     Args:
#         backbone (nn.Module): the network used to compute the features for the model.
#             The backbone should return an OrderedDict[Tensor], with the key being
#             "out" for the last feature map used, and "aux" if an auxiliary classifier
#             is used.
#         classifier (nn.Module): module that takes the "out" element returned from
#             the backbone and returns a dense prediction.
#         aux_classifier (nn.Module, optional): auxiliary classifier used during training
#     """

#     pass


# class FCNHead(nn.Sequential):
#     def __init__(self, in_channels: int, channels: int) -> None:
#         inter_channels = in_channels // 4
#         layers = [
#             nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Conv2d(inter_channels, channels, 1),
#         ]

#         super().__init__(*layers)


# _COMMON_META = {
#     "categories": _VOC_CATEGORIES,
#     "min_size": (1, 1),
#     "_docs": """
#         These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC
#         dataset.
#     """,
# }


# class FCN_ResNet50_Weights(WeightsEnum):
#     COCO_WITH_VOC_LABELS_V1 = Weights(
#         url="https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
#         transforms=partial(SemanticSegmentation, resize_size=520),
#         meta={
#             **_COMMON_META,
#             "num_params": 35322218,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet50",
#             "_metrics": {
#                 "COCO-val2017-VOC-labels": {
#                     "miou": 60.5,
#                     "pixel_acc": 91.4,
#                 }
#             },
#             "_ops": 152.717,
#             "_weight_size": 135.009,
#         },
#     )
#     DEFAULT = COCO_WITH_VOC_LABELS_V1


# class FCN_ResNet101_Weights(WeightsEnum):
#     COCO_WITH_VOC_LABELS_V1 = Weights(
#         url="https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth",
#         transforms=partial(SemanticSegmentation, resize_size=520),
#         meta={
#             **_COMMON_META,
#             "num_params": 54314346,
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet101",
#             "_metrics": {
#                 "COCO-val2017-VOC-labels": {
#                     "miou": 63.7,
#                     "pixel_acc": 91.9,
#                 }
#             },
#             "_ops": 232.738,
#             "_weight_size": 207.711,
#         },
#     )
#     DEFAULT = COCO_WITH_VOC_LABELS_V1


# def _fcn_resnet(
#     backbone: ResNet,
#     num_classes: int,
#     aux: Optional[bool],
# ) -> FCN:
#     return_layers = {"layer4": "out"}
#     if aux:
#         return_layers["layer3"] = "aux"
#     backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

#     aux_classifier = FCNHead(1024, num_classes) if aux else None
#     classifier = FCNHead(2048, num_classes)
#     return FCN(backbone, classifier, aux_classifier)


# @register_model()
# @handle_legacy_interface(
#     weights=("pretrained", FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
#     weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
# )
# def fcn_resnet50(
#     *,
#     weights: Optional[FCN_ResNet50_Weights] = None,
#     progress: bool = True,
#     num_classes: Optional[int] = None,
#     aux_loss: Optional[bool] = None,
#     weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
#     **kwargs: Any,
# ) -> FCN:
#     """Fully-Convolutional Network model with a ResNet-50 backbone from the `Fully Convolutional
#     Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper.

#     .. betastatus:: segmentation module

#     Args:
#         weights (:class:`~torchvision.models.segmentation.FCN_ResNet50_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.segmentation.FCN_ResNet50_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         num_classes (int, optional): number of output classes of the model (including the background).
#         aux_loss (bool, optional): If True, it uses an auxiliary loss.
#         weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained
#             weights for the backbone.
#         **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
#             for more details about this class.

#     .. autoclass:: torchvision.models.segmentation.FCN_ResNet50_Weights
#         :members:
#     """

#     weights = FCN_ResNet50_Weights.verify(weights)
#     weights_backbone = ResNet50_Weights.verify(weights_backbone)

#     if weights is not None:
#         weights_backbone = None
#         num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
#         aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
#     elif num_classes is None:
#         num_classes = 21

#     backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
#     model = _fcn_resnet(backbone, num_classes, aux_loss)

#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress))

#     return model
