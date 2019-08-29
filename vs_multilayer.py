#该代码用于生成结果向量，最后计算loss

from __future__ import division

import numpy as np
import tensorflow as tf

# components
from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import pooling_layer as pool
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu

"""
name : 'CBR'
input_batch : visual_featuremap_ph_train  (4096*3)

"""
def vs_multilayer(input_batch, name, middle_layer_dim=1000, output_layer_dim=21*3, dropout=True, reuse=False):
    with tf.variable_scope(name):
        if reuse==True:
            print name+" reuse variables"
            tf.get_variable_scope().reuse_variables()
        else:
            print name+" doesn't reuse variables"

        
        layer1 = fc_relu('layer1', input_batch, output_dim=middle_layer_dim)  # (4096*3)--->1000
        if dropout:
            layer1 = drop(layer1, 0.5)
        sim_score = fc('layer2', layer1, output_dim=output_layer_dim) # 1000---->21*3
    return sim_score
