import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from torchvision import transforms as T
from pathlib import Path
import math
from torch.nn import init
from utils.path_hyperparameter import ph
from PIL import Image



class PH_Placeholder:
    def __init__(self):
        # self.log_path = './logs_ablation_multi_dpfa/'  # 日志路径保持不变，便于追踪
        self.log_path = './GZCDD/'
        self.patch_size = 256  # 默认图像处理尺寸


ph = PH_Placeholder()


class Conv_BN_ReLU(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                      padding=kernel // 2, bias=False, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


def kernel_size(in_channel):

    k = int((math.log2(in_channel) + 1) // 2)
    if k % 2 == 0:
        return k + 1
    else:
        return k

class DepthwiseSeparableConv(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        # 深度卷积（Depthwise Convolution）
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        # 逐点卷积（Pointwise Convolution）
        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class GMA(nn.Module):


    def __init__(self, in_channel, gamma=2, reduction=4):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction


        self.group_conv = nn.Sequential(
            nn.Conv2d(
                in_channel,
                in_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=gamma,
                dilation=1
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channel,
                in_channel,
                kernel_size=3,
                stride=1,
                padding=2,
                groups=gamma,
                dilation=2
            )
        )

        # 通道缩减与恢复
        mid_channel = in_channel // reduction

        # 轻量级混合注意力
        self.attention = nn.Sequential(
            # 通道缩减
            nn.Conv2d(in_channel, mid_channel, 1),
            nn.GELU(),

            # 深度可分离空间注意力
            nn.Conv2d(mid_channel, mid_channel, 3, padding=1, groups=mid_channel),
            nn.Conv2d(mid_channel, 1, 1),  # 空间注意力图
            nn.Sigmoid(),

            # 通道注意力融合
            nn.Conv2d(1, in_channel, 1),
            nn.Sigmoid()
        )

        # 残差连接缩放因子
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, log=False, module_name=None, img_name=None):
        """
        参数:
            x (Tensor): 输入特征图 [B, C, H, W]
        """
        identity = x

        # 多尺度特征提取
        group_features = self.group_conv(x)

        # 混合注意力计算
        attention_map = self.attention(group_features)

        # 特征增强 (残差连接)
        out = identity + self.res_scale * (group_features * attention_map)

        # 日志记录
        if log:
            log_feature(
                [attention_map],
                module_name=module_name,
                feature_name_list=['attention_map'],
                img_name=img_name,
                module_output=True
            )

        return out


class SCFA(nn.Module):


    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 动态卷积核大小（随通道数自适应）
        self.k = kernel_size(in_channel)

        # 共用通道注意力卷积
        self.channel_conv = nn.Conv1d(4, 2, kernel_size=self.k, padding=self.k // 2)

        # 共用空间注意力卷积
        self.spatial_conv = nn.Conv2d(4, 2, kernel_size=7, padding=3)

        # 1x1卷积用于 concat 后通道还原
        self.fusion_conv = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1)

        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2, log=None, module_name=None, img_name=None):
        # ===== 通道注意力 =====
        t1_avg = self.avg_pool(t1)
        t1_max = self.max_pool(t1)
        t2_avg = self.avg_pool(t2)
        t2_max = self.max_pool(t2)

        channel_pool = torch.cat([t1_avg, t1_max, t2_avg, t2_max], dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        shared_channel_att = self.channel_conv(channel_pool)  # b,2,c
        channel_att = F.softmax(shared_channel_att, dim=1)  # b,2,c
        w1 = channel_att[:, 0].unsqueeze(-1).unsqueeze(-1)  # b,c,1,1
        w2 = channel_att[:, 1].unsqueeze(-1).unsqueeze(-1)  # b,c,1,1
        t1 = t1 * w1
        t2 = t2 * w2

        # ===== 空间注意力 =====
        t1_avg_spatial = torch.mean(t1, dim=1, keepdim=True)
        t1_max_spatial = torch.max(t1, dim=1, keepdim=True)[0]
        t2_avg_spatial = torch.mean(t2, dim=1, keepdim=True)
        t2_max_spatial = torch.max(t2, dim=1, keepdim=True)[0]

        spatial_pool = torch.cat([t1_avg_spatial, t1_max_spatial, t2_avg_spatial, t2_max_spatial], dim=1)  # b,4,h,w
        shared_spatial_att = self.spatial_conv(spatial_pool)  # b,2,h,w
        spatial_att = F.softmax(shared_spatial_att, dim=1)  # b,2,h,w
        s1 = spatial_att[:, 0:1, :, :]  # b,1,h,w
        s2 = spatial_att[:, 1:2, :, :]  # b,1,h,w
        t1 = t1 * s1
        t2 = t2 * s2


        fuse = torch.cat([t1, t2], dim=1)           # b,2c,h,w
        fuse = self.fusion_conv(fuse)               # b,c,h,w


        fuse = fuse + t1 + t2

        if log:
            log_list = [t1, t2, s1, s2, fuse]
            feature_name_list = ['t1', 't2',
                                 't1_spatial_attention', 't2_spatial_attention', 'fuse']
            log_feature(log_list=log_list, module_name=module_name,
                        feature_name_list=feature_name_list,
                        img_name=img_name, module_output=True)

        return fuse

class DepthwiseSeparableConvBlock(nn.Module):

    def __init__(self, in_d, out_d, k=3, s=1, p=1, bias=False):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_d, in_d, k, s, p, groups=in_d, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_d)
        self.relu1 = nn.ReLU(inplace=True)


        self.pointwise_conv = nn.Conv2d(in_d, out_d, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_d)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise_conv(x)))
        x = self.relu2(self.bn2(self.pointwise_conv(x)))
        return x


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class Encoder_Block(nn.Module):


    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel == in_channel * 2, 'Encoder_Block: out_channel should be double of in_channel'

        # 增加一个下采样前的卷积层
        self.pre_down_conv = Conv_BN_ReLU(in_channel=in_channel, out_channel=in_channel, kernel=3, stride=1)

        # 下采样卷积层
        self.down_conv = Conv_BN_ReLU(in_channel=in_channel, out_channel=out_channel, kernel=3, stride=2)
        # self.down_conv = DepthwiseSeparableConvBlock(in_d=in_channel, out_d=out_channel, k=3, s=2, p=1)

        # 引入 EMA 注意力模块，增强下采样后的特征
        self.ema = GMA(in_channel=out_channel)
        # self.ema = EMA(in_channel=out_channel, reduction=16, gamma=4)

        # 最后的特征提取卷积
        self.post_down_conv = Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1)

    def forward(self, x, log=False, module_name=None, img_name=None):
        x = self.pre_down_conv(x)
        x = self.down_conv(x)

        x_res = x.clone()  # 为 EMA 模块保留残差连接

        x_att = self.ema(x, log=log, module_name=module_name, img_name=img_name)

        output = x_res + x_att
        output = self.post_down_conv(output)

        return output


class Decoder_Block(nn.Module):
    """Basic block in decoder."""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=in_channel + out_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, de, en):
        de = self.up(de)
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)

        return output

def channel_split(x):
    """Half segment one feature on channel dimension into two features, mixture them on channel dimension,
    and split them into two features."""
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]

def log_feature(log_list, module_name, feature_name_list, img_name, module_output=True, labels=None):
    """
    Log output feature of module and model (backwards-compatible).

    Parameters:
        log_list (list[Tensor]): [B, C, H, W]
        module_name (str): 模块名（作为子目录）
        feature_name_list (list[str]): 与 log_list 对应的名字
        img_name (list[str] or str): 每个 batch 的图片名或单个名字
        module_output (bool): True=中间特征可视化；False=模型输出可视化/对比
        labels (Tensor, optional): [B,1,H0,W0]，用于 TP/FP/TN/FN 对比（0/1 或 0/255 均可）
    """
    # 延迟导入，避免外部环境无这几个库时报错
    import os
    from pathlib import Path
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    try:
        import cv2
    except Exception:
        cv2 = None

    def norm_names(name_in, b):
        # list/tuple -> 按原样；str/其他 -> 复制 b 份
        if isinstance(name_in, (list, tuple)):
            names = [str(n) for n in name_in]
            if len(names) < b:
                names = names + [f"img_{i}" for i in range(len(names), b)]
            else:
                names = names[:b]
        else:
            names = [str(name_in)] * b
        return names

    def get_base_dir():
        try:
            base = os.path.join(ph.log_path, module_name)  # 复用你原来的全局 ph
        except Exception:
            base = os.path.join("runs", "vis", module_name)
        Path(base).mkdir(parents=True, exist_ok=True)
        return base

    def get_target_size(h, w):
        try:
            P = int(ph.patch_size)
            return (P, P)
        except Exception:
            return (h, w)

    def mm01(arr):
        a_min = float(arr.min())
        a_max = float(arr.max())
        if a_max > a_min:
            arr = (arr - a_min) / (a_max - a_min)
        else:
            arr = arr * 0.0
        return np.clip(arr, 0.0, 1.0)

    def colorize_jet(prob01, reverse=True):
        # prob01: [H,W] in [0,1]
        img8 = (prob01 * 255.0).astype(np.uint8)
        if reverse:
            img8 = 255 - img8  # 近似 matplotlib 的 "jet_r"
        if cv2 is not None:
            return cv2.applyColorMap(img8, cv2.COLORMAP_JET)  # BGR
        # fallback: 3通道灰度
        return np.stack([img8, img8, img8], axis=-1)

    bsize = None
    base_dir = get_base_dir()

    for k, log in enumerate(log_list):
        if not isinstance(log, torch.Tensor) or log.dim() != 4:
            continue

        feat_name = feature_name_list[k]
        log = log.detach()
        b, c, h, w = log.size()
        bsize = b
        names = norm_names(img_name, b)
        tgt_h, tgt_w = get_target_size(h, w)

        if module_output:
            # 中间特征可视化：均值 -> resize -> per-image min-max -> JET 伪彩
            vis = torch.mean(log, dim=1, keepdim=True)  # [B,1,h,w]
            vis = F.interpolate(vis, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
            vis_np = vis.squeeze(1).cpu().numpy()  # [B,H,W]
            out_dir = os.path.join(base_dir, feat_name)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            eq_dir = os.path.join(base_dir, f"{feat_name}_equalize")
            Path(eq_dir).mkdir(parents=True, exist_ok=True)

            for i in range(b):
                norm01 = mm01(vis_np[i])
                color = colorize_jet(norm01, reverse=True)  # JET_R 风格
                if cv2 is not None:
                    cv2.imwrite(os.path.join(out_dir, f"{names[i]}.jpg"), color)  # BGR
                    # 直方图均衡版（先对灰度再上色）
                    gray8 = (norm01 * 255).astype(np.uint8)
                    try:
                        eq = cv2.equalizeHist(gray8)
                        eq_color = cv2.applyColorMap(255 - eq, cv2.COLORMAP_JET)
                        cv2.imwrite(os.path.join(eq_dir, f"{names[i]}.jpg"), eq_color)
                    except Exception:
                        pass
                else:
                    Image.fromarray(color[:, :, ::-1]).save(os.path.join(out_dir, f"{names[i]}.jpg"))

        else:
            # 模型输出：保存热力图 + 二值图 + (可选)TP/FP/TN/FN 对比
            preds_dir = base_dir
            Path(preds_dir).mkdir(parents=True, exist_ok=True)

            prob = torch.sigmoid(log)  # [B,C,h,w]
            # 规整到单通道：如果多通道，取最大响应
            if prob.size(1) > 1:
                prob1 = prob.max(dim=1, keepdim=False)[0]  # [B,h,w]
            else:
                prob1 = prob[:, 0]  # [B,h,w]

            # 上采样到目标大小
            prob_up = F.interpolate(prob1.unsqueeze(1), size=(tgt_h, tgt_w), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()  # [B,H,W]
            bin_up = (prob_up >= 0.5).astype(np.uint8)  # [B,H,W]

            # 预备标签（可选）：统一成 0/1
            gt_up = None
            if labels is not None and isinstance(labels, torch.Tensor):
                lab = labels.detach()
                if lab.dim() == 4:
                    lab = lab[:, 0]
                lab_up = F.interpolate(lab.unsqueeze(1).float(), size=(tgt_h, tgt_w), mode='nearest').squeeze(1).cpu().numpy()
                # 兼容 0/1 或 0/255
                if lab_up.max() > 1.0:
                    gt_up = (lab_up >= 128).astype(np.uint8)
                else:
                    gt_up = (lab_up > 0.5).astype(np.uint8)

            for i in range(b):
                # 1) 彩色热力图（JET_R 风格）
                heat01 = np.clip(prob_up[i], 0.0, 1.0)
                heat_color = colorize_jet(heat01, reverse=True)
                heat_name = f"{names[i]}_{feat_name}_heat.png"
                if cv2 is not None:
                    cv2.imwrite(os.path.join(preds_dir, heat_name), heat_color)
                else:
                    Image.fromarray(heat_color[:, :, ::-1]).save(os.path.join(preds_dir, heat_name))

                # 2) 二值预测图（0/255）
                pr = (bin_up[i] * 255).astype(np.uint8)
                pred_name = f"{names[i]}_{feat_name}_pred.png"
                Image.fromarray(pr).save(os.path.join(preds_dir, pred_name))

                # 3) 仅对 change_out 生成 TP/FP/TN/FN 对比图
                if gt_up is not None and feat_name == "change_out":
                    gt = gt_up[i]
                    pr01 = bin_up[i]
                    H, W = gt.shape
                    cmp_img = np.zeros((H, W, 3), dtype=np.uint8)
                    tp = (pr01 == 1) & (gt == 1)
                    fp = (pr01 == 1) & (gt == 0)
                    tn = (pr01 == 0) & (gt == 0)
                    fn = (pr01 == 0) & (gt == 1)
                    # 颜色：TP白，FP红，TN黑，FN绿
                    cmp_img[tp] = [255, 255, 255]
                    cmp_img[fp] = [255, 0, 0]
                    cmp_img[tn] = [0, 0, 0]
                    cmp_img[fn] = [0, 255, 0]
                    compare_name = f"{names[i]}_{feat_name}_compare.png"
                    Image.fromarray(cmp_img).save(os.path.join(preds_dir, compare_name))

    if bsize is not None:
        try:
            print(f"✅ log_feature: saved visualizations for {bsize} images under {os.path.join(ph.log_path, module_name)}")
        except Exception:
            print(f"✅ log_feature: saved visualizations for {bsize} images under {os.path.join('runs','vis',module_name)}")


class ConvBlock(nn.Module):
    def __init__(self, in_d, out_d, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_d, out_d, k, s, p, bias=False),
            nn.BatchNorm2d(out_d),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class MultiScaleConvBlock(nn.Module):
    """
    多尺度卷积块：使用1x1和3x3卷积并行处理输入，然后拼接。
    """
    def __init__(self, in_d, out_d):
        super().__init__()
        # 1x1 卷积分支，用于捕获通道间信息
        self.conv1x1 = ConvBlock(in_d, out_d // 2, k=1, p=0)
        # 3x3 卷积分支，用于捕获空间信息
        self.conv3x3 = ConvBlock(in_d, out_d // 2, k=3, p=1)

    def forward(self, x):
        feat1x1 = self.conv1x1(x)
        feat3x3 = self.conv3x3(x)
        # 将两个分支的输出拼接
        return torch.cat([feat1x1, feat3x3], dim=1)


class ASFAM(nn.Module):
    def __init__(self, in_d=None):
        super().__init__()
        if in_d is None:
            in_d = [64, 128, 256, 512]
        self.in_d = in_d

        # residual refinement
        self.res_c2 = ConvBlock(in_d[0], in_d[0], k=1, p=0)
        self.res_c3 = ConvBlock(in_d[1], in_d[1], k=1, p=0)
        self.res_c4 = ConvBlock(in_d[2], in_d[2], k=1, p=0)

        # scale2: c2 + c3
        self.conv_scale2_c2 = MultiScaleConvBlock(in_d[0], 32)
        self.conv_scale2_c3 = MultiScaleConvBlock(in_d[1], 32)
        self.conv_s2 = nn.Sequential(
            nn.Conv2d(64, in_d[0], 1, bias=False),
            nn.BatchNorm2d(in_d[0]),
            nn.ReLU(inplace=True),
            ConvBlock(in_d[0], in_d[0])
        )

        # scale3: c3 + c4
        self.conv_scale3_c3 = MultiScaleConvBlock(in_d[1], 64)
        self.conv_scale3_c4 = MultiScaleConvBlock(in_d[2], 64)
        self.conv_s3 = nn.Sequential(
            nn.Conv2d(128, in_d[1], 1, bias=False),
            nn.BatchNorm2d(in_d[1]),
            nn.ReLU(inplace=True),
            ConvBlock(in_d[1], in_d[1])
        )

        # scale4: c4 + c5
        self.conv_scale4_c4 = MultiScaleConvBlock(in_d[2], 128)
        self.conv_scale4_c5 = MultiScaleConvBlock(in_d[3], 128)
        self.conv_s4 = nn.Sequential(
            nn.Conv2d(256, in_d[2], 1, bias=False),
            nn.BatchNorm2d(in_d[2]),
            nn.ReLU(inplace=True),
            ConvBlock(in_d[2], in_d[2])
        )

        # scale5: just refine c5
        self.conv_s5 = nn.Sequential(
            nn.Conv2d(in_d[3], in_d[3], 1, bias=False),
            nn.BatchNorm2d(in_d[3]),
            nn.ReLU(inplace=True),
            ConvBlock(in_d[3], in_d[3], k=1, p=0)
        )

    def forward(self, c2, c3, c4, c5):
        # scale2
        c2_s2 = self.conv_scale2_c2(c2)
        c3_s2 = F.interpolate(self.conv_scale2_c3(c3), scale_factor=2, mode='bilinear', align_corners=False)
        s2 = self.conv_s2(torch.cat([c2_s2, c3_s2], dim=1)) + self.res_c2(c2)

        # scale3
        c3_s3 = self.conv_scale3_c3(c3)
        c4_s3 = F.interpolate(self.conv_scale3_c4(c4), scale_factor=2, mode='bilinear', align_corners=False)
        s3 = self.conv_s3(torch.cat([c3_s3, c4_s3], dim=1)) + self.res_c3(c3)

        # scale4
        c4_s4 = self.conv_scale4_c4(c4)
        c5_s4 = F.interpolate(self.conv_scale4_c5(c5), scale_factor=2, mode='bilinear', align_corners=False)
        s4 = self.conv_s4(torch.cat([c4_s4, c5_s4], dim=1)) + self.res_c4(c4)

        # scale5
        s5 = self.conv_s5(c5)

        return s2, s3, s4, s5


class SimpleCrossFeatureBlock(nn.Module):
    def __init__(self, in_channels):
        super(SimpleCrossFeatureBlock, self).__init__()

        self.conv_interact = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # 拼接两个时相的特征图
        x_fused = torch.cat([x1, x2], dim=1)

        # 通过卷积层学习交互特征
        x_interact = self.conv_interact(x_fused)

        # 将交互特征与原始特征相加，实现增强和残差连接
        x1_out = x1 + x_interact
        x2_out = x2 + x_interact

        return x1_out, x2_out    #最普通的版本

# class SimpleCrossFeatureBlock(nn.Module):
#     """
#     CFR (Contextual Feature Refinement) Module - 极简且参数高效的精确率提升模块。
#     创新点：
#     1. 上下文引导的通道自校准：通过学习全局上下文的通道统计，动态调整交互特征贡献。
#     2. 极简高效的隐式门控：参数极少，非注意力机制，无显式差异计算。
#     3. 核心交互逻辑的稳健强化：在交互特征生成后、注入前进行精炼，保持稳定。
#     """
#
#     def __init__(self, in_channels):
#         super(SimpleCrossFeatureBlock, self).__init__()
#         c = in_channels
#
#         # 1. 主干交互路径：与你原始模块的核心卷积层相同，学习基础交互特征
#         self.conv_interact_main = nn.Sequential(
#             nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.ReLU(inplace=True)
#         )
#
#         # 2. 上下文引导的通道自校准分支：生成通道级的自校准因子
#         self.channel_calibration_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # 全局平均池化，得到 (N, 2C, 1, 1)
#             nn.Conv2d(c * 2, c, kernel_size=1, bias=True),  # 1x1卷积映射到C通道，学习通道重要性
#             nn.Softplus()  # Softplus激活，确保校准因子非负且平滑，允许放大或抑制
#         )
#
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#
#         # 对 channel_calibration_head 中的 1x1 卷积进行初始化
#         if isinstance(self.channel_calibration_head[1], nn.Conv2d):
#             # 对于 Softplus 激活，使用 He 或 Xavier 初始化通常都可以
#             # 考虑初始时对特征影响较小，可以偏向于较小的初始值或使用零偏置
#             init.kaiming_uniform_(self.channel_calibration_head[1].weight, nonlinearity='relu')
#             if self.channel_calibration_head[1].bias is not None:
#                 # 初始偏置可设为0，或略负值让Softplus初始输出接近0，减少初期扰动
#                 init.constant_(self.channel_calibration_head[1].bias, 0)
#
#     def forward(self, x1, x2):
#         # 拼接两个时相的特征图
#         x_fused = torch.cat([x1, x2], dim=1)  # (N, 2C, H, W)
#
#         # 1. 主干交互路径：生成基础交互特征
#         base_interact_feat = self.conv_interact_main(x_fused)  # (N, C, H, W)
#
#         # 2. 上下文引导的通道自校准：生成通道级的校准因子
#         # calibration_factors 的形状为 (N, C, 1, 1)，经过Softplus确保非负
#         calibration_factors = self.channel_calibration_head(x_fused)
#
#         # 3. 将基础交互特征与校准因子进行逐元素乘法（广播到空间维度）
#         # 校准因子会根据全局上下文信息，自适应地调整每个通道的交互特征强度。
#         # 这有助于模型聚焦于更可靠的通道信息，提升精确率。
#         refined_interact_feat = base_interact_feat * calibration_factors  # (N, C, H, W)
#
#         # 4. 将精炼后的交互特征与原始特征相加，实现增强和残差连接
#         x1_out = x1 + refined_interact_feat
#         x2_out = x2 + refined_interact_feat
#
#         return x1_out, x2_out

# class SimpleCrossFeatureBlock(nn.Module):
#     """
#     CFR (Contextual Feature Refinement) Module - 极简且参数高效的精确率提升模块。
#     创新点：
#     1. 上下文引导的通道自校准：通过学习全局上下文的通道统计，动态调整交互特征贡献。
#     2. 极简高效的隐式门控：参数极少，非注意力机制，无显式差异计算。
#     3. 核心交互逻辑的稳健强化：在交互特征生成后、注入前进行精炼，保持稳定。
#     """
#
#     def __init__(self, in_channels):
#         super(SimpleCrossFeatureBlock, self).__init__()
#         c = in_channels
#
#         # 1. 主干交互路径：与你原始模块的核心卷积层相同，学习基础交互特征
#         self.conv_interact_main = nn.Sequential(
#             nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.ReLU(inplace=True)
#         )
#
#         # 2. 上下文引导的通道自校准分支：生成通道级的自校准因子
#         self.channel_calibration_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # 全局平均池化，得到 (N, 2C, 1, 1)
#             nn.Conv2d(c * 2, c, kernel_size=1, bias=True),  # 1x1卷积映射到C通道，学习通道重要性
#             nn.Softplus()  # Softplus激活，确保校准因子非负且平滑，允许放大或抑制
#         )
#
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#
#         # 对 channel_calibration_head 中的 1x1 卷积进行初始化
#         if isinstance(self.channel_calibration_head[1], nn.Conv2d):
#             # 对于 Softplus 激活，使用 He 或 Xavier 初始化通常都可以
#             # 考虑初始时对特征影响较小，可以偏向于较小的初始值或使用零偏置
#             init.kaiming_uniform_(self.channel_calibration_head[1].weight, nonlinearity='relu')
#             if self.channel_calibration_head[1].bias is not None:
#                 # 初始偏置可设为0，或略负值让Softplus初始输出接近0，减少初期扰动
#                 init.constant_(self.channel_calibration_head[1].bias, 0)
#
#     def forward(self, x1, x2):
#         # 拼接两个时相的特征图
#         x_fused = torch.cat([x1, x2], dim=1)  # (N, 2C, H, W)
#
#         # 1. 主干交互路径：生成基础交互特征
#         base_interact_feat = self.conv_interact_main(x_fused)  # (N, C, H, W)
#
#         # 2. 上下文引导的通道自校准：生成通道级的校准因子
#         # calibration_factors 的形状为 (N, C, 1, 1)，经过Softplus确保非负
#         calibration_factors = self.channel_calibration_head(x_fused)
#
#         # 3. 将基础交互特征与校准因子进行逐元素乘法（广播到空间维度）
#         # 校准因子会根据全局上下文信息，自适应地调整每个通道的交互特征强度。
#         # 这有助于模型聚焦于更可靠的通道信息，提升精确率。
#         refined_interact_feat = base_interact_feat * calibration_factors  # (N, C, H, W)
#
#         # 4. 将精炼后的交互特征与原始特征相加，实现增强和残差连接
#         x1_out = x1 + refined_interact_feat
#         x2_out = x2 + refined_interact_feat
#
#         return x1_out, x2_out

# class SimpleCrossFeatureBlock(nn.Module):
#     """
#     ADR (Adaptive Difference Refinement) Block - 极简且参数高效的创新模块。
#     创新点：
#     1. 差异中心自适应调制：交互卷积路径用于生成差异的调制图。
#     2. 隐式上下文门控：调制图对原始差异进行乘法调制，无额外参数。
#     3. 对称守恒差异注入：通过 +/- d_refined/2 保持特征和不变，防止漂移。
#     """
#
#     def __init__(self, in_channels):
#         super(SimpleCrossFeatureBlock, self).__init__()
#         c = in_channels
#
#         # 1. 主卷积路径：用于学习上下文交互特征，并为调制图 M 提供基础信息
#         # 结构与你原始的conv_interact前几层相同
#         self.conv_main_path = nn.Sequential(
#             nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.ReLU(inplace=True)
#         )
#
#         # 2. 调制头：从主路径输出的特征生成差异调制图 M
#         # 仅额外增加一个1x1卷积，参数量极小
#         self.modulation_head = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=1, bias=True),  # 1x1卷积，通道数不变
#             nn.Sigmoid()  # Sigmoid确保调制图 M 的范围在 [0, 1]
#         )
#
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#
#         # 对 modulation_head 中的 1x1 卷积进行初始化，通常使用 Xavier 或 Kaiming uniform
#         if isinstance(self.modulation_head[0], nn.Conv2d):
#             init.xavier_uniform_(self.modulation_head[0].weight)  # Sigmoid激活函数更适合Xavier
#             if self.modulation_head[0].bias is not None:
#                 init.constant_(self.modulation_head[0].bias, 0)
#
#     def forward(self, x1, x2):
#         # 拼接两个时相的特征图
#         x_fused = torch.cat([x1, x2], dim=1)  # (N, 2C, H, W)
#
#         # 1. 通过主卷积路径学习上下文特征
#         context_feat = self.conv_main_path(x_fused)  # (N, C, H, W)
#
#         # 2. 从上下文特征生成差异调制图 M，范围 [0, 1]
#         modulation_map = self.modulation_head(context_feat)  # (N, C, H, W)
#
#         # 3. 计算原始差异信号
#         raw_difference = x2 - x1  # (N, C, H, W)
#
#         # 4. 应用调制图 M 精炼差异信号
#         # 这确保了只有在上下文交互特征认为重要的区域，差异信号才会被加强或保留
#         refined_difference = raw_difference * modulation_map  # (N, C, H, W)
#
#         # 5. 对称守恒地将精炼后的差异注入回原始特征
#         # 确保 x1_out + x2_out = x1 + x2，维持能量守恒
#         x1_out = x1 - 0.5 * refined_difference
#         x2_out = x2 + 0.5 * refined_difference
#
#         return x1_out, x2_out

#
#
# class SimpleCrossFeatureBlock(nn.Module):
#     """
#     RFIM (Robust Filtered Interaction Module) - 精确率优先版本 (无 LayerScale)。
#     - 鲁棒特征预提炼
#     - 上下文感知差异显著性门控（非注意力）
#     - 差异增强型交互特征融合
#     """
#
#     def __init__(self, in_channels):
#         super(SimpleCrossFeatureBlock, self).__init__()
#         c = in_channels
#
#         # 1. 鲁棒特征预提炼 (每个时相独立，共享权重)
#         # 使用两个ConvBNReLU层来对x1和x2进行初始提炼，降低噪声
#         self.feature_refinement = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.ReLU(inplace=True)
#         )
#
#         # 2. 上下文感知差异显著性门控分支 (Context-Aware Difference Saliency Gating Branch)
#         #   输入: 提炼后的差异绝对值 (C channels)
#         #   输出: 单通道显著性门控图 (sigmoid激活)
#         self.lp = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # 低通滤波，提供上下文
#         self.saliency_head = nn.Sequential(
#             nn.Conv2d(c, c // 2, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c // 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c // 2, 1, kernel_size=1, bias=False),  # 输出单通道显著性分数
#             # Sigmoid激活将在forward中应用
#         )
#
#         # 3. 差异增强型交互特征融合 (Difference-Enhanced Interactive Feature Fusion)
#         #   输入: 拼接的提炼特征 (2C) + 原始差异 (C) => 3C
#         #   输出: 最终交互特征 (C)
#         self.main_interact_fusion = nn.Sequential(
#             nn.Conv2d(c * 3, c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.ReLU(inplace=True)
#         )
#
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#
#     def forward(self, x1, x2):
#         # 1. 鲁棒特征预提炼
#         x1_res = self.feature_refinement(x1)
#         x2_res = self.feature_refinement(x2)
#
#         # 原始差异特征 (用于主交互路径)
#         diff_raw = x2_res - x1_res  # (N, C, H, W)
#
#         # 2. 上下文感知差异显著性门控
#         abs_diff_for_saliency = torch.abs(diff_raw)  # 差异的绝对值
#         # 对绝对差异进行低通滤波，获取上下文信息，再通过卷积头生成显著性分数
#         saliency_map_logits = self.saliency_head(self.lp(abs_diff_for_saliency))  # (N, 1, H, W)
#
#         # 应用Sigmoid生成门控值 (0到1之间)
#         saliency_gate = torch.sigmoid(saliency_map_logits)  # (N, 1, H, W)
#
#         # 3. 差异增强型交互特征融合
#         # 将提炼后的x1, x2和原始差异特征拼接
#         fused_input_for_main = torch.cat([x1_res, x2_res, diff_raw], dim=1)  # (N, 3C, H, W)
#
#         # 通过卷积层学习交互特征
#         x_interact_raw = self.main_interact_fusion(fused_input_for_main)  # (N, C, H, W)
#
#         # 应用差异显著性门控，强调高置信度差异区域，抑制低置信度区域，提升精确率
#         x_interact_gated = x_interact_raw * saliency_gate  # (N, C, H, W) * (N, 1, H, W)
#
#         # 将门控后的交互特征与原始特征相加，实现增强和残差连接
#         x1_out = x1 + x_interact_gated
#         x2_out = x2 + x_interact_gated
#
#         return x1_out, x2_out

# class SimpleCrossFeatureBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(SimpleCrossFeatureBlock, self).__init__()
#
#         interact_channels = in_channels * 2  # 拼接后的输入通道数
#         out_channels = in_channels  # 交互模块的输出通道数，与原始特征通道数一致
#
#         # 分支1: 主干3x3卷积路径（与原模块类似，捕捉局部交互）
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(interact_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         # 分支2: 深度可分离5x5卷积，捕捉稍大感受野的交互
#         # 深度卷积部分：通道数不变，只在空间维度上操作
#         # 逐点卷积部分：将深度卷积的输出映射到所需的out_channels
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(interact_channels, interact_channels, kernel_size=5, padding=2, groups=interact_channels,
#                       bias=False),
#             nn.BatchNorm2d(interact_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(interact_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         # 可学习的标量权重，用于融合两个分支的输出
#         # 初始化为1，保证开始时两个分支贡献相等
#         self.fusion_weights_raw = nn.Parameter(torch.ones(2, dtype=torch.float32))
#
#         # LayerScale 参数，按通道控制残差连接的强度
#         # 初始化为0.1，保证训练初期更新幅度较小，增加稳定性
#         self.gamma = nn.Parameter(torch.ones(out_channels) * 0.1)
#
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#         # nn.Parameter 类型的 fusion_weights_raw 和 gamma 已在 __init__ 中初始化，此处无需重复处理
#
#     def forward(self, x1, x2):
#         # 拼接两个时相的特征图，作为交互模块的输入
#         x_fused = torch.cat([x1, x2], dim=1)
#
#         # 两个并行分支分别学习交互特征
#         feat_b1 = self.branch1(x_fused)
#         feat_b2 = self.branch2(x_fused)
#
#         # 可学习的凸组合融合：
#         # 1. 对原始权重应用 softplus 函数，确保权重非负
#         # 2. 对权重进行归一化，使其和为1，实现凸组合
#         fusion_weights = F.softplus(self.fusion_weights_raw)
#         fusion_weights = fusion_weights / (fusion_weights.sum() + 1e-6)  # 加上epsilon防止除0
#
#         # 将两个分支的输出按权重相加融合
#         x_interact_combined = fusion_weights[0] * feat_b1 + fusion_weights[1] * feat_b2
#
#         # 应用 LayerScale：将gamma参数重塑为 (1, C, 1, 1) 进行通道wise的乘法
#         # 控制交互特征的强度，增强训练稳定性
#         scaled_x_interact = self.gamma.view(1, -1, 1, 1) * x_interact_combined
#
#         # 将按比例缩放后的交互特征与原始特征相加，实现增强和残差连接
#         x1_out = x1 + scaled_x_interact
#         x2_out = x2 + scaled_x_interact
#
#         return x1_out, x2_out

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init
#
# class SimpleCrossFeatureBlock(nn.Module):
#     """
#     Recall-oriented Symmetric Interaction (无注意力)
#     - 和差分解: s=(x1+x2)/2, d=x2-x1
#     - d分支: a*d + b*(d - Avg3(d))，保留低频变化并兼顾边缘
#     - 局部相关性调制: 7x7窗口的按通道皮尔逊相关，带下限 m_min
#     - 差分增益: 对d分支施加>1的轻微增益beta，偏向召回
#     - 反对称注入: x1+=alpha*z, x2-=alpha*z
#     """
#     def __init__(self, in_channels, lp_ks=3, corr_ks=7, m_min=0.25, gamma_init=1.25):
#         super().__init__()
#         self.in_channels = in_channels
#         self.m_min = m_min
#
#         # 固定核平均池化（低通 / 局部统计）
#         self.lp = nn.AvgPool2d(kernel_size=lp_ks, stride=1, padding=lp_ks // 2)
#         self.corr_pool = nn.AvgPool2d(kernel_size=corr_ks, stride=1, padding=corr_ks // 2)
#
#         # s/d各自轻量卷积
#         self.s_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#         )
#         self.d_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#         )
#
#         # 融合
#         self.mix = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels),
#         )
#
#         # 相关性调制强度（可学习）
#         self.corr_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
#
#         # a,b>=0: d直通/高通混合权重；alpha>0；beta>0(对d分支的额外增益, 实际使用1+beta)
#         def inv_softplus(v):  # 返回x使得 softplus(x)=v
#             return math.log(math.exp(v) - 1.0)
#
#         self.a_raw = nn.Parameter(torch.tensor(inv_softplus(1.0)))  # a≈1.0
#         self.b_raw = nn.Parameter(torch.tensor(inv_softplus(0.3)))  # b≈0.3
#         self.alpha_raw = nn.Parameter(torch.tensor(inv_softplus(0.1)))  # alpha≈0.1
#         self.beta_raw  = nn.Parameter(torch.tensor(inv_softplus(0.3)))  # beta≈0.3 => 1+beta≈1.3
#
#         self.init_params()
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#
#     def local_channel_corr(self, x1, x2, eps=1e-5):
#         """
#         局部(窗口)按通道皮尔逊相关, 输出(N,C,H,W), 非学习、非注意力
#         """
#         mu1 = self.corr_pool(x1)
#         mu2 = self.corr_pool(x2)
#         x1c = x1 - mu1
#         x2c = x2 - mu2
#         cov = self.corr_pool(x1c * x2c)
#         v1 = torch.sqrt(self.corr_pool(x1c.pow(2)) + eps)
#         v2 = torch.sqrt(self.corr_pool(x2c.pow(2)) + eps)
#         rho = cov / (v1 * v2 + eps)
#         return rho.clamp_(-1.0, 1.0)
#
#     def forward(self, x1, x2):
#         # 1) 和/差分解
#         s = 0.5 * (x1 + x2)
#         d = x2 - x1
#
#         # 2) d分支: 直通与高通的非负混合
#         d_hp = d - self.lp(d)
#         a = F.softplus(self.a_raw)   # >=0
#         b = F.softplus(self.b_raw)   # >=0
#         d_mix = a * d + b * d_hp     # 既保留低频变化，又强调边缘
#
#         # 3) 局部相关性调制 + 下限
#         rho = self.local_channel_corr(x1, x2)                    # (N,C,H,W)
#         m = torch.sigmoid(self.corr_gamma * (1.0 - rho))         # (0,1)
#         if self.m_min > 0:
#             m = self.m_min + (1.0 - self.m_min) * m              # 防止过度抑制
#         d_mix = d_mix * m
#
#         # 4) s/d各自变换，d分支增益(偏召回)
#         s_feat = self.s_conv(self.lp(s))
#         d_feat = self.d_conv(d_mix)
#         beta = 1.0 + F.softplus(self.beta_raw)                   # >1
#         d_feat = beta * d_feat
#
#         # 5) 交互提取（不做tanh饱和）
#         z = self.mix(torch.cat([s_feat, d_feat], dim=1))
#
#         # 6) 反对称残差注入
#         alpha = F.softplus(self.alpha_raw)                        # >0
#         x1_out = x1 + alpha * z
#         x2_out = x2 - alpha * z
#         return x1_out, x2_out


