import pdb
from typing import Any, Callable, List, Optional, Type, Union
from PCONV_operator import PseudoContextV2, PseudoPadV2, PseudoFillV2, SphereSlice, SphereUslice
import torch
import torch.nn as nn
from torch import Tensor
from .mask_bnorm import MaskedBatchNorm2d


class PseudoParam:

    def __init__(self, ctx: PseudoContextV2, npart: int, device_ids) -> None:
        self.ctx = ctx
        self.npart = npart
        self.device_ids = device_ids


class MaskManager:

    def __init__(self, param: PseudoParam, strides=[2, 2, 2, 2, 2]) -> None:
        self.mask_dict = {}
        self.ratio_dict = {}
        self.old_w, self.old_n, self.device = -1, -1, None
        self.strides = strides
        self.fill_ops = [PseudoFillV2(0, param.npart, param.ctx, device=param.device_ids) for _ in strides]

    def empty_dict(self):
        key_list = list(self.mask_dict.keys())
        for pkey in key_list:
            self.mask_dict.pop(pkey)
            self.ratio_dict.pop(pkey)

    def update(self, x: torch.Tensor):
        n, _, h, w = x.shape
        if w == self.old_w and self.device == x.device and n == self.old_n: return
        if (not w == self.old_w) or (not n == self.old_n):
            self.empty_dict()
            th, tw = h, w
            for stride, op in zip(self.strides, self.fill_ops):
                th = th // stride
                tw = tw // stride
                dt = torch.ones((n, 1, th, tw), dtype=torch.float32, device=x.device)
                pkey = '{}_0'.format(tw)
                self.mask_dict[pkey] = op(dt)
                self.ratio_dict[pkey] = n * th * tw / torch.sum(self.mask_dict[pkey]).item()
                self.mask_dict[pkey].detach()
        if not x.device == self.device:
            for pkey in self.mask_dict.keys():
                self.mask_dict[pkey] = self.mask_dict[pkey].to(x.device)

        self.old_w, self.device = w, x.device

    def add_new_item(self, x: torch.Tensor, pad: int):
        w = x.shape[-1]
        nkey = '{}_{}'.format(w, pad)
        if nkey in self.mask_dict.keys(): return
        old_mask = self.mask_dict['{}_0'.format(w)]
        h = old_mask.shape[-2]
        new_mask = old_mask[:, :, :1, :].repeat(1, 1, h + 2 * pad, 1)
        self.mask_dict[nkey] = new_mask
        self.ratio_dict[nkey] = float(torch.numel(new_mask)) / torch.sum(new_mask).item()


class BatchNorm2DPseudo(MaskedBatchNorm2d):

    def __init__(self, mask_mng: MaskManager, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True, pad=0):
        super(BatchNorm2DPseudo, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        # self.bn = MaskedBatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.mask_mng = mask_mng
        self.pad = pad

    def forward(self, x):
        w = x.shape[-1]
        if self.pad > 0: self.mask_mng.add_new_item(x, self.pad)
        mask = self.mask_mng.mask_dict['{}_{}'.format(w, self.pad)]
        tx = super(BatchNorm2DPseudo, self).forward(x, mask)
        return tx * mask


class PseudoConv2D(nn.Module):

    def __init__(self, param: PseudoParam, in_planes, out_planes, kernel_size=3, pad=1, stride=1, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_planes, in_planes, kernel_size, kernel_size)))
        self.bias = nn.Parameter(torch.zeros((out_planes,))) if bias else None
        self.stride = stride
        self.pad = PseudoPadV2(pad, param.npart, param.ctx, device=param.device_ids)
        self.trim = PseudoFillV2(0, param.npart, param.ctx, device=param.device_ids)
        nn.init.kaiming_uniform_(self.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        tx = self.pad(x)
        tx = nn.functional.conv2d(tx, self.weight, self.bias, self.stride)
        return self.trim(tx)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            param: PseudoParam,
            mask_mng: MaskManager,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            base_width: int = 64,
    ) -> None:
        super().__init__()
        if base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")

        self.conv1 = PseudoConv2D(param, inplanes, planes, stride=stride)
        self.bn1 = BatchNorm2DPseudo(mask_mng, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = PseudoConv2D(param, planes, planes)
        self.bn2 = BatchNorm2DPseudo(mask_mng, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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


class ResNet(nn.Module):
    def __init__(
            self,
            npart, device_ids,
            block: BasicBlock,
            layers: List[int],
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64
    ) -> None:
        super().__init__()

        self._norm_layer = BatchNorm2DPseudo
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.slice = SphereSlice(npart, pad=0, opt=False, device=device_ids)
        self.uslice = SphereUslice(npart, pad=0, opt=False, device=device_ids)
        self.ctx = PseudoContextV2(npart, False, device=device_ids)
        self.param = PseudoParam(self.ctx, npart, device_ids)
        self.mask_mng = MaskManager(self.param)
        self.pad1 = PseudoPadV2(3, npart, self.ctx, device=device_ids)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.pool_pad = PseudoPadV2(1, npart, self.ctx, device=device_ids)
        self.bn1 = BatchNorm2DPseudo(self.mask_mng, self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(self.param, self.mask_mng, block, 64, layers[0])
        self.layer2 = self._make_layer(self.param, self.mask_mng, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.param, self.mask_mng, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.param, self.mask_mng, block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
            self,
            param: PseudoParam,
            mng: MaskManager,
            block: BasicBlock,
            planes: int,
            blocks: int,
            stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(mng, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                param, mng, self.inplanes, planes, stride, downsample, self.base_width
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    param,
                    mng,
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        tx = self.slice(x)
        self.mask_mng.update(tx)
        x = self.pad1(tx)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool_pad(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.uslice(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        npart, device_ids,
        block: BasicBlock,
        layers: List[int],
        **kwargs: Any,
) -> ResNet:
    model = ResNet(npart, device_ids, block, layers, **kwargs)
    return model


def resnet18(npart=16, device_ids=0) -> ResNet:
    return _resnet(npart, device_ids, BasicBlock, [2, 2, 2, 2])


def resnet34(npart=16, device_ids=0) -> ResNet:
    return _resnet(npart, device_ids, BasicBlock, [3, 4, 6, 3])


def load_pretrained_model(model_path, ndict):
    weight = torch.load(model_path)
    for pkey in weight.keys():
        nkey = pkey  # 'backbone.' + pkey
        if nkey in ndict.keys():
            ndict[nkey] = weight[pkey]
    return ndict


class ResNet18(nn.Module):
    def __init__(self, pretrained_path=None):
        super(ResNet18, self).__init__()

        self.backbone = resnet18()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.quality = self.quality_regression(512, 128, 1)

        params = load_pretrained_model(pretrained_path, self.backbone.state_dict())
        self.backbone.load_state_dict(params)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, x):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.backbone(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # x: batch * frames x 1

        x = self.quality(x)
        # b x frames
        x = x.view(x_size[0], x_size[1])

        x = torch.mean(x, dim=1)
        x = torch.flatten(x)

        return x



class ResNet34(nn.Module):
    def __init__(self, pretrained_path=None):
        super(ResNet34, self).__init__()

        self.backbone = resnet34()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.quality = self.quality_regression(512, 128, 1)

        params = load_pretrained_model(pretrained_path, self.backbone.state_dict())
        self.backbone.load_state_dict(params)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, x):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.backbone(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # x: batch * frames x 1

        x = self.quality(x)
        # b x frames
        x = x.view(x_size[0], x_size[1])

        x = torch.mean(x, dim=1)
        x = torch.flatten(x)

        return x



if __name__ == '__main__':
    device_id = 0
    device = f'cuda:{device_id}'
    model = resnet18(device_ids=device_id)
    params = load_pretrained_model("https://download.pytorch.org/models/resnet18-f37072fd.pth", model.state_dict())
    model.load_state_dict(params)
    model = model.to(device)
    data = torch.rand((1, 3, 512, 1024), dtype=torch.float32, device=device)
    y = model(data)
    pdb.set_trace()
    loss = torch.sum(y ** 2) / 2
    loss.backward()
    pass