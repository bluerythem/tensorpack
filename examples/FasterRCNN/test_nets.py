"""
Just to make sure the build process of pvanet works well
"""

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_number
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu

from PVANet import pvanet, pvanet_scope
from basemodel import (
    image_preprocess, resnet_c4_backbone, resnet_conv5,
    resnet_fpn_backbone)
import config

import util
import tensorflow as tf
slim = tf.contrib.slim

is_training = True

# with slim.arg_scope(pvanet_scope(is_training,data_format='NCHW',axis = 1)):
#     inputs = tf.placeholder(dtype = tf.float32, shape = [1, 3, 896, 1200])
#     net, end_points = pvanet(inputs)
#     for k in sorted(end_points.keys()):
#         print (k, end_points[k].shape)
#     print (net.shape)

# with slim.arg_scope(pvanet_scope(is_training,data_format='NHWC',axis=-1)):
#     inputs = tf.placeholder(dtype = tf.float32, shape = [None, None, None, 3])
#     net, end_points = pvanet(inputs)
#     for k in sorted(end_points.keys()):
#         print (k, end_points[k].shape)
#     print (net.shape)

image=tf.placeholder(tf.float32, (1, 3, 896, 1200), 'image')
c2345 = resnet_fpn_backbone(image, config.RESNET_NUM_BLOCK)

for node in c2345:
    print(node.shape)