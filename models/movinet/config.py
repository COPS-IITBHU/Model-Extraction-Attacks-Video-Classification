"""
Inspired by
https://github.com/PeizeSun/SparseR-CNN/blob/dff4c43a9526a6d0d2480abc833e78a7c29ddb1a/detectron2/config/defaults.py
"""
from fvcore.common.config import CfgNode as CN


def fill_SE_config(conf, input_channels,
                   out_channels,
                   expanded_channels,
                   kernel_size,
                   stride,
                   padding,
                   padding_avg,
                   ):
    conf.expanded_channels = expanded_channels
    conf.padding_avg = padding_avg
    fill_conv(conf, input_channels,
              out_channels,
              kernel_size,
              stride,
              padding,
              )


def fill_conv(conf, input_channels,
              out_channels,
              kernel_size,
              stride,
              padding, ):
    conf.input_channels = input_channels
    conf.out_channels = out_channels
    conf.kernel_size = kernel_size
    conf.stride = stride
    conf.padding = padding


_C = CN()

_C.MODEL = CN()

###################
#### MoViNetA2 ####
###################

_C.MODEL.MoViNetA2 = CN()
_C.MODEL.MoViNetA2.name = "A2"
_C.MODEL.MoViNetA2.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA2_statedict_v3?raw=true"
_C.MODEL.MoViNetA2.stream_weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA2_stream_statedict_v3?raw=true"

_C.MODEL.MoViNetA2.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA2.conv1, 3, 16, (1, 3, 3), (1, 2, 2), (0, 1, 1))

_C.MODEL.MoViNetA2.blocks = [[CN() for _ in range(3)],
                             [CN() for _ in range(5)],
                             [CN() for _ in range(5)],
                             [CN() for _ in range(6)],
                             [CN() for _ in range(7)]]

# Block2
fill_SE_config(_C.MODEL.MoViNetA2.blocks[0][0], 16, 16, 40, (1, 5, 5), (1, 2, 2), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[0][1], 16, 16, 40, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[0][2], 16, 16, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

# block 3
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][0], 16, 40, 96, (3, 3, 3), (1, 2, 2), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][1], 40, 40, 120, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][2], 40, 40, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][3], 40, 40, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[1][4], 40, 40, 120, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

# block 4
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][0], 40, 72, 240, (5, 3, 3), (1, 2, 2), (2, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][1], 72, 72, 160, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][2], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][3], 72, 72, 192, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[2][4], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

# block 5
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][0], 72, 72, 240, (5, 3, 3), (1, 1, 1), (2, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][1], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][2], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][3], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][4], 72, 72, 144, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[3][5], 72, 72, 240, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

# block 6
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][0], 72, 144, 480, (5, 3, 3), (1, 2, 2), (2, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][1], 144, 144, 384, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][2], 144, 144, 384, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][3], 144, 144, 480, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][4], 144, 144, 480, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][5], 144, 144, 480, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
fill_SE_config(_C.MODEL.MoViNetA2.blocks[4][6], 144, 144, 576, (1, 3, 3), (1, 1, 1), (0, 1, 1), (0, 1, 1))

_C.MODEL.MoViNetA2.conv7 = CN()
fill_conv(_C.MODEL.MoViNetA2.conv7, 144, 640, (1, 1, 1), (1, 1, 1), (0, 0, 0))

_C.MODEL.MoViNetA2.dense9 = CN()
_C.MODEL.MoViNetA2.dense9.hidden_dim = 2048
