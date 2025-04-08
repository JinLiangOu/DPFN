import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY


# from mamba_ssm.modules.mamba2 import Mamba2 as ma


class BSConvU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias=True):
        super(BSConvU, self).__init__()

        self.pw = nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            bias=False)

        self.dw = nn.Conv2d(out_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            dilation,
                            groups=out_channels,
                            bias=bias)

    def forward(self, x):
        x = self.pw(x)
        x = self.dw(x)
        return x


class RepBSConvU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias=True,
                 deploy=False):
        super(RepBSConvU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.deploy = deploy

        self.pw = nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            bias=False)

        if deploy:
            self.dw = nn.Conv2d(out_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                dilation,
                                groups=out_channels,
                                bias=bias)

        else:
            self.dw1k = nn.Conv2d(out_channels,
                                  out_channels,
                                  (1, kernel_size),
                                  stride,
                                  (0, padding),
                                  dilation,
                                  groups=out_channels,
                                  bias=bias)
            self.dwk1 = nn.Conv2d(out_channels,
                                  out_channels,
                                  (kernel_size, 1),
                                  stride,
                                  (padding, 0),
                                  dilation,
                                  groups=out_channels,
                                  bias=bias)
            self.dwk = nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size,
                                 stride,
                                 padding,
                                 dilation,
                                 groups=out_channels,
                                 bias=bias)

    def forward(self, x):
        x = self.pw(x)
        if self.deploy:
            x = self.dw(x)
        else:
            x1 = self.dw1k(x)
            x2 = self.dwk1(x)
            x3 = self.dwk(x)
            x = x1 + x2 + x3
        return x

    def reparameter(self):
        _pad = (self.kernel_size - 1) // 2
        weight1 = F.pad(
            self.dw1k.weight.data,
            (_pad,
             _pad,
             0,
             0))
        weight2 = F.pad(
            self.dwk1.weight.data,
            (0,
             0,
             _pad,
             _pad))
        weight3 = self.dwk.weight.data
        bias1 = self.dw1k.bias.data
        bias2 = self.dwk1.bias.data
        bias3 = self.dwk.bias.data
        weight = weight1 + weight2 + weight3
        bias = bias1 + bias2 + bias3
        return weight, bias

    def switch_deploy(self):
        self.deploy = True
        self.dw = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            groups=self.out_channels,
            bias=self.bias)
        self.dw.weight.data, self.dw.bias.data = self.reparameter()
        self.__delattr__('dw1k')
        self.__delattr__('dwk1')
        self.__delattr__('dwk')


class HyperDistillationRef(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(HyperDistillationRef, self).__init__()
        main_channels = out_channels // 2
        hyper_channels = out_channels // 8
        h1_channels = hyper_channels
        h2_channels = hyper_channels * 2
        h3_channels = hyper_channels * 3

        self.layer_uc1 = RepBSConvU(in_channels,
                                    main_channels,
                                    3,
                                    1,
                                    1,
                                    1)
        self.layer_uc2 = BSConvU(main_channels,
                                 main_channels,
                                 3,
                                 1,
                                 3,
                                 3)
        self.layer_uc3 = RepBSConvU(main_channels,
                                    main_channels,
                                    5,
                                    1,
                                    2,
                                    1)
        self.layer_uc4 = BSConvU(main_channels,
                                 main_channels,
                                 5,
                                 1,
                                 6,
                                 3)

        self.layer_ref1 = nn.Conv2d(in_channels,
                                    h1_channels,
                                    1,
                                    1,
                                    0,
                                    1)
        self.layer_ref2 = nn.Conv2d(main_channels,
                                    h2_channels,
                                    1,
                                    1,
                                    0,
                                    1)
        self.layer_ref3 = nn.Conv2d(main_channels,
                                    h3_channels,
                                    1,
                                    1,
                                    0,
                                    1)

        self.ref = nn.Conv2d(hyper_channels * 10,
                             out_channels,
                             1,
                             1,
                             0,
                             1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        hyper_x1 = self.act(self.layer_ref1(x))
        x1 = self.act(self.layer_uc1(x))
        hyper_x2 = self.act(self.layer_ref2(x1))
        x2 = self.act(self.layer_uc2(x1))
        hyper_x3 = self.act(self.layer_ref3(x2))
        x3 = self.act(self.layer_uc3(x2))
        x4 = self.act(self.layer_uc4(x3))
        x = self.act(self.ref(torch.concat([hyper_x1, hyper_x2, hyper_x3, x4], dim=1))) + shortcut
        return x


class SimpleDistillationRef(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(SimpleDistillationRef, self).__init__()
        main_channels = out_channels // 2
        hyper_channels = out_channels // 8
        h1_channels = hyper_channels
        h2_channels = hyper_channels * 2
        h3_channels = hyper_channels * 3

        self.layer_uc1 = BSConvU(in_channels,
                                    main_channels,
                                    3,
                                    1,
                                    1,
                                    1)
        self.layer_uc2 = BSConvU(main_channels,
                                 main_channels,
                                 3,
                                 1,
                                 1,
                                 1)
        self.layer_uc3 = BSConvU(main_channels,
                                    main_channels,
                                    5,
                                    1,
                                    2,
                                    1)
        self.layer_uc4 = BSConvU(main_channels,
                                 main_channels,
                                 5,
                                 1,
                                 2,
                                 1)

        self.layer_ref1 = nn.Conv2d(in_channels,
                                    h1_channels,
                                    1,
                                    1,
                                    0,
                                    1)
        self.layer_ref2 = nn.Conv2d(main_channels,
                                    h2_channels,
                                    1,
                                    1,
                                    0,
                                    1)
        self.layer_ref3 = nn.Conv2d(main_channels,
                                    h3_channels,
                                    1,
                                    1,
                                    0,
                                    1)

        self.ref = nn.Conv2d(hyper_channels * 10,
                             out_channels,
                             1,
                             1,
                             0,
                             1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        hyper_x1 = self.act(self.layer_ref1(x))
        x1 = self.act(self.layer_uc1(x))
        hyper_x2 = self.act(self.layer_ref2(x1))
        x2 = self.act(self.layer_uc2(x1))
        hyper_x3 = self.act(self.layer_ref3(x2))
        x3 = self.act(self.layer_uc3(x2))
        x4 = self.act(self.layer_uc4(x3))
        x = self.act(self.ref(torch.concat([hyper_x1, hyper_x2, hyper_x3, x4], dim=1))) + shortcut
        return x


class MultilayerPerceptron(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(MultilayerPerceptron, self).__init__()
        mid_channels = in_channels // 16
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      1,
                      1,
                      0,
                      1),
            nn.GELU(),
            nn.Conv2d(mid_channels,
                      out_channels,
                      1,
                      1,
                      0,
                      1)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class DualChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DualChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = MultilayerPerceptron(in_channels, out_channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        shortcut = x
        x = self.max_pool(x) + self.avg_pool(x)
        x = self.mlp(x)
        x = self.act(x)
        return x * shortcut


class DualSpaitalAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 simple = False):
        super(DualSpaitalAttention, self).__init__()
        mid_channels = in_channels // 8
        self.conv1 = nn.Conv2d(in_channels,
                               mid_channels,
                               1,
                               1,
                               0,
                               1)

        self.conv2 = RepBSConvU(mid_channels,
                                mid_channels,
                                3,
                                1,
                                1,
                                1, 
                                deploy=simple)
        self.conv3 = BSConvU(mid_channels,
                                mid_channels,
                                3,
                                1,
                                3,
                                3)

        self.conv4 = RepBSConvU(mid_channels,
                                mid_channels,
                                5,
                                1,
                                2,
                                1, 
                                deploy=simple)
        self.conv5 = BSConvU(mid_channels,
                                mid_channels,
                                5,
                                1,
                                6,
                                3)

        self.ref = nn.Conv2d(mid_channels * 2,
                             out_channels,
                             1,
                             1,
                             0,
                             1)
        self.act = nn.GELU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x1 = self.act(self.conv2(x))
        x1 = self.act(self.conv3(x1))

        x2 = self.act(self.conv4(x))
        x2 = self.act(self.conv5(x2))

        x = self.ref(torch.concat([x1, x2], dim=1))
        x = self.sig(x)
        x = x * shortcut
        return x


class LightWiseSimple(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(LightWiseSimple, self).__init__()
        self.sdr = SimpleDistillationRef(in_channels,
                                        out_channels)
        self.dsa = DualSpaitalAttention(out_channels,
                                        out_channels,
                                        simple = True)
        self.ref = nn.Conv2d(out_channels,
                             out_channels,
                             1,
                             1,
                             0,
                             1)

    def forward(self, x):
        shortcut = x
        x = self.sdr(x)
        x = self.dsa(x) + x
        x = self.ref(x + shortcut)
        return x

class LightWiseBasic(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(LightWiseBasic, self).__init__()
        self.hdr = HyperDistillationRef(in_channels,
                                        out_channels)
        self.dsa = DualSpaitalAttention(out_channels,
                                        out_channels)
        self.dca = DualChannelAttention(out_channels,
                                        out_channels)
        self.ref = nn.Conv2d(out_channels,
                             out_channels,
                             1,
                             1,
                             0,
                             1)

    def forward(self, x):
        shortcut = x
        x = self.hdr(x)
        x = self.dsa(x) + x
        x = self.dca(x) + x
        x = self.ref(x + shortcut)
        return x


@ARCH_REGISTRY.register()
class REPDPFN(nn.Module):
    def __init__(self,
                 in_dim=3,
                 main_channels=64,
                 hyper_channels=24,
                 out_dim=3,
                 upscale=4,
                 upsampler='pixelshuffledirect'):
        super(REPDCFN, self).__init__()
        self.mc = main_channels
        self.hc = hyper_channels

        self.embed = RepBSConvU(in_dim * 4,
                                main_channels + hyper_channels,
                                3,
                                1,
                                1,
                                1)

        self.M1 = LightWiseBasic(in_channels=main_channels,
                                 out_channels=main_channels)
        self.M2 = LightWiseBasic(in_channels=main_channels,
                                 out_channels=main_channels)
        self.M3 = LightWiseBasic(in_channels=main_channels,
                                 out_channels=main_channels)
        self.M4 = LightWiseBasic(in_channels=main_channels,
                                 out_channels=main_channels)
        self.M5 = LightWiseBasic(in_channels=main_channels,
                                 out_channels=main_channels)
        self.M6 = LightWiseBasic(in_channels=main_channels,
                                 out_channels=main_channels)
        self.M7 = LightWiseBasic(in_channels=main_channels,
                                 out_channels=main_channels)
        self.M8 = LightWiseBasic(in_channels=main_channels,
                                 out_channels=main_channels)
        self.M9 = LightWiseBasic(in_channels=main_channels,
                                 out_channels=main_channels)

        self.H1 = LightWiseSimple(in_channels=hyper_channels,
                                 out_channels=hyper_channels)
        self.H2 = LightWiseSimple(in_channels=hyper_channels,
                                 out_channels=hyper_channels)
        self.H3 = LightWiseSimple(in_channels=hyper_channels,
                                 out_channels=hyper_channels)
        self.H4 = LightWiseSimple(in_channels=hyper_channels,
                                 out_channels=hyper_channels)

        self.hyper_fusion_1 = nn.Conv2d(hyper_channels,
                                        main_channels,
                                        1,
                                        1,
                                        0,
                                        1)
        self.hyper_fusion_2 = nn.Conv2d(hyper_channels,
                                        main_channels,
                                        1,
                                        1,
                                        0,
                                        1)
        self.hyper_fusion_3 = nn.Conv2d(hyper_channels,
                                        main_channels,
                                        1,
                                        1,
                                        0,
                                        1)
        self.hyper_fusion_4 = nn.Conv2d(hyper_channels,
                                        main_channels,
                                        1,
                                        1,
                                        0,
                                        1)

        self.ref = nn.Sequential(
            nn.Conv2d(main_channels * 9 + hyper_channels * 4,
                      main_channels,
                      1,
                      1,
                      0,
                      1),
            nn.GELU(),
            RepBSConvU(main_channels,
                       main_channels,
                       3,
                       1,
                       1,
                       1)
        )
        # self.ref = BSConvU(channels, channels, 3, 1, 1, 1)

        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale,
                                                           num_feat=main_channels,
                                                           num_out_ch=out_dim)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=main_channels,
                                                          num_feat=main_channels,
                                                          num_out_ch=out_dim)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=main_channels,
                                                    num_feat=main_channels,
                                                    num_out_ch=out_dim)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=main_channels,
                                              unf=24,
                                              out_nc=out_dim)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        out_main, out_hyper = self.embed(torch.concat([input,
                                                       input,
                                                       input,
                                                       input],
                                                      dim=1)).split([self.mc, self.hc], dim=1)

        # out_MF = self.F1(out_fea)

        out_H1 = self.H1(out_hyper)
        out_H2 = self.H2(out_H1)
        out_H3 = self.H3(out_H2)
        out_H4 = self.H4(out_H3)

        out_M1 = self.M1(out_main)
        out_M2 = self.M2(out_M1 + self.hyper_fusion_1(out_H1))
        out_M3 = self.M3(out_M2)
        out_M4 = self.M4(out_M3 + self.hyper_fusion_2(out_H2))
        out_M5 = self.M5(out_M4)
        out_M6 = self.M6(out_M5 + self.hyper_fusion_3(out_H3))
        out_M7 = self.M7(out_M6)
        out_M8 = self.M8(out_M7 + self.hyper_fusion_4(out_H4))
        out_M9 = self.M9(out_M8)
        # out_M10 = self.M10(out_M9)

        out_fea = self.ref(
            torch.concat([out_M1,
                          out_H1,
                          out_M2,
                          out_M3,
                          out_H2,
                          out_M4,
                          out_M5,
                          out_H3,
                          out_M6,
                          out_M7,
                          out_H4,
                          out_M8,
                          out_M9],
                         dim=1)) + out_main
        output = self.upsampler(out_fea)
        return output