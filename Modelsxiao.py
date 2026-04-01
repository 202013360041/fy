import torch
import torch.nn as nn
from torch.nn import init

from models.dpcd_partstwo import (Conv_BN_ReLU, Encoder_Block, SCFA, Decoder_Block, log_feature, ASFAM, SimpleCrossFeatureBlock)


class DPCD(nn.Module):
    """ Change detection model

    Input :obj:`t1_img` and :obj:`t2_img`, extract encoder feature by :obj:`en_block1-4`,
    then exchange channel feature of :obj:`t1_feature` and :obj:`t2_feature`, and extract
    encoder feature by :obj:`en_block5`.

    Upsample to get decoder feature by :obj:`de_block1-3`, get :obj:`seg_feature1` and :obj:`seg_feature2`
    by :obj:`seg_out1` and :obj:`seg_out2`.

    Fuse t1 and t2 corresponding feature to get change feature by :obj:`dpfa` and :obj:`change_blcok`.

    Notice that output of module and model could be log in this model.

    Attribute:
        en_block(class): encoder feature extractor.
        channel_exchange(class): exchange t1 and t2 feature.
        de_block(class): decoder feature upsampler and extractor.
        dpfa(class): fuse t1 and t2 feature to get change feature
            using both spatial and channel attention.
        change_block(class): change feature upsampler and extracor.
        seg_out(class): get decoder feature seg out result.
        upsample_x2(class): upsample change feature by 2.
        conv_out_change(class): conv out change feature out result.
    """

    def __init__(self):
        super().__init__()

        channel_list = [32, 64, 128, 256, 512]
        # encoder
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=1))
        self.en_block2 = Encoder_Block(in_channel=channel_list[0], out_channel=channel_list[1])
        self.en_block3 = Encoder_Block(in_channel=channel_list[1], out_channel=channel_list[2])
        self.en_block4 = Encoder_Block(in_channel=channel_list[2], out_channel=channel_list[3])
        self.en_block5 = Encoder_Block(in_channel=channel_list[3], out_channel=channel_list[4])

        # 实例化 MASAM 模块
        self.asfam = ASFAM()

        self.csfb_att2 = SimpleCrossFeatureBlock(in_channels=channel_list[1])
        self.csfb_att3 = SimpleCrossFeatureBlock(in_channels=channel_list[2])
        self.csfb_att4 = SimpleCrossFeatureBlock(in_channels=channel_list[3])
        self.csfb_att5 = SimpleCrossFeatureBlock(in_channels=channel_list[4])

        # decoder
        self.de_block1 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.de_block2 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.de_block3 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        # dpfa
        self.dpfa1 = SCFA(in_channel=channel_list[4])
        self.dpfa2 = SCFA(in_channel=channel_list[3])
        self.dpfa3 = SCFA(in_channel=channel_list[2])
        self.dpfa4 = SCFA(in_channel=channel_list[1])

        # change path
        self.change_block4 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.change_block3 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.change_block2 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        self.seg_out1 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)
        self.seg_out2 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)

        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_change = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

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
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, t1, t2, log=False, img_name=None, label=None):
        # encoder
        t1_1 = self.en_block1(t1)
        t2_1 = self.en_block1(t2)

        if log:
            t1_2 = self.en_block2(t1_1, log=log, module_name='t1_1_en_block2', img_name=img_name)
            t2_2 = self.en_block2(t2_1, log=log, module_name='t2_1_en_block2', img_name=img_name)

            t1_3 = self.en_block3(t1_2, log=log, module_name='t1_2_en_block3', img_name=img_name)
            t2_3 = self.en_block3(t2_2, log=log, module_name='t2_2en_block3', img_name=img_name)

            t1_4 = self.en_block4(t1_3, log=log, module_name='t1_3_en_block4', img_name=img_name)
            t2_4 = self.en_block4(t2_3, log=log, module_name='t2_3_en_block4', img_name=img_name)

            t1_5 = self.en_block5(t1_4, log=log, module_name='t1_4_en_block5', img_name=img_name)
            t2_5 = self.en_block5(t2_4, log=log, module_name='t2_4_en_block5', img_name=img_name)
        else:
            t1_2 = self.en_block2(t1_1)
            t2_2 = self.en_block2(t2_1)

            t1_3 = self.en_block3(t1_2)
            t2_3 = self.en_block3(t2_2)

            t1_4 = self.en_block4(t1_3)
            t2_4 = self.en_block4(t2_3)

            t1_5 = self.en_block5(t1_4)
            t2_5 = self.en_block5(t2_4)

        #对 t1 和 t2 的特征图分别进行处理
        t1_s2, t1_s3, t1_s4, t1_s5 = self.asfam(t1_2, t1_3, t1_4, t1_5)
        t2_s2, t2_s3, t2_s4, t2_s5 = self.asfam(t2_2, t2_3, t2_4, t2_5)

        t1_s2_enhanced, t2_s2_enhanced = self.csfb_att2(t1_s2, t2_s2)
        t1_s3_enhanced, t2_s3_enhanced = self.csfb_att3(t1_s3, t2_s3)
        t1_s4_enhanced, t2_s4_enhanced = self.csfb_att4(t1_s4, t2_s4)
        t1_s5_enhanced, t2_s5_enhanced = self.csfb_att5(t1_s5, t2_s5)


        de1_5 = t1_s5_enhanced
        de2_5 = t2_s5_enhanced

        de1_4 = self.de_block1(de1_5, t1_s4_enhanced)
        de2_4 = self.de_block1(de2_5, t2_s4_enhanced)

        de1_3 = self.de_block2(de1_4, t1_s3_enhanced)
        de2_3 = self.de_block2(de2_4, t2_s3_enhanced)

        de1_2 = self.de_block3(de1_3, t1_s2_enhanced)
        de2_2 = self.de_block3(de2_3, t2_s2_enhanced)

        seg_out1 = self.seg_out1(de1_2)
        seg_out2 = self.seg_out2(de2_2)

        if log:
            change_5 = self.dpfa1(de1_5, de2_5, log=log, module_name='de1_5_de2_5_dpfa1',
                                  img_name=img_name)

            change_4 = self.change_block4(change_5, self.dpfa2(de1_4, de2_4, log=log, module_name='de1_4_de2_4_dpfa2',
                                                               img_name=img_name))

            change_3 = self.change_block3(change_4, self.dpfa3(de1_3, de2_3, log=log, module_name='de1_3_de2_3_dpfa3',
                                                               img_name=img_name))

            change_2 = self.change_block2(change_3, self.dpfa4(de1_2, de2_2, log=log, module_name='de1_2_de2_2_dpfa4',
                                                               img_name=img_name))
        else:
            change_5 = self.dpfa1(de1_5, de2_5)

            change_4 = self.change_block4(change_5, self.dpfa2(de1_4, de2_4))

            change_3 = self.change_block3(change_4, self.dpfa3(de1_3, de2_3))

            change_2 = self.change_block2(change_3, self.dpfa4(de1_2, de2_2))

        change = self.upsample_x2(change_2)
        change_out = self.conv_out_change(change)

        if log:
            log_feature(log_list=[change_out, seg_out1, seg_out2], module_name='model_levir',
                        feature_name_list=['change_out', 'seg_out1', 'seg_out2'],
                        img_name=img_name, module_output=False, labels=label)

        return change_out, seg_out1, seg_out2
