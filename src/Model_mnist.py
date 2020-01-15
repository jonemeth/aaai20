import tensorflow as tf
import numpy as np
import math

from ops import *

import cfg


def encode_mnist(x, is_train, drop_keep):
    input_size = int(x.get_shape()[2])
    input_depth = int(x.get_shape()[1])
    use_bn = False
    use_wn = False
    use_ln = False
    use_heNorm=False
    use_gn = False
    bias = (not use_bn) and (not use_ln)

    activation = tf.nn.relu
    
    
    x = conv2d( x, 16, 'conv_2', is_train, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm,  activation=activation, gn=use_gn)
    x = conv2d( x, 32, 'conv_3', is_train, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm,  activation=activation, gn=use_gn)
    
    
    
    zC_mu = conv2d( x, 64, 'conv_4_C_mu', is_train, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm,  activation=activation, gn=use_gn)
    zC_mu = tf.reshape(zC_mu, [-1,  int(zC_mu.get_shape()[1])* int(zC_mu.get_shape()[2])* int(zC_mu.get_shape()[3]) ])
    zC_mu = linear(zC_mu,  128, 'lin_1_C_mu', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=activation, noisy_weights=False, trainable=True, gn=False)
    zC_mu = linear(zC_mu,  cfg.cfg.ZC_dim, 'lin_2_C_mu', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=None, noisy_weights=False, trainable=True, gn=False)


    zS_mu = conv2d( x, 64, 'conv_4_S_mu', is_train, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm,  activation=activation, gn=use_gn)
    zS_mu = tf.reshape(zS_mu, [-1,  int(zS_mu.get_shape()[1])* int(zS_mu.get_shape()[2])* int(zS_mu.get_shape()[3]) ])
    zS_mu = linear(zS_mu,  128, 'lin_1_S_mu', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=activation, noisy_weights=False, trainable=True, gn=False)
    zS_mu = linear(zS_mu,  cfg.cfg.ZS_dim, 'lin_2_S_mu', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=None, noisy_weights=False, trainable=True, gn=False)
    
    
    
    zC_lv = conv2d( x, 64, 'conv_4_C_lv', is_train, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm,  activation=activation, gn=use_gn)
    zC_lv = tf.reshape(zC_lv, [-1,  int(zC_lv.get_shape()[1])* int(zC_lv.get_shape()[2])* int(zC_lv.get_shape()[3]) ])
    zC_lv = linear(zC_lv,  128, 'lin_1_C_lv', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=activation, noisy_weights=False, trainable=True, gn=False)
    zC_lv = linear(zC_lv,  cfg.cfg.ZC_dim, 'lin_2_C_lv', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=None, noisy_weights=False, trainable=True, gn=False)


    zS_lv = conv2d( x, 64, 'conv_4_S_lv', is_train, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm,  activation=activation, gn=use_gn)
    zS_lv = tf.reshape(zS_lv, [-1,  int(zS_lv.get_shape()[1])* int(zS_lv.get_shape()[2])* int(zS_lv.get_shape()[3]) ])
    zS_lv = linear(zS_lv,  128, 'lin_1_S_lv', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=activation, noisy_weights=False, trainable=True, gn=False)
    zS_lv = linear(zS_lv,  cfg.cfg.ZS_dim, 'lin_2_S_lv', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=None, noisy_weights=False, trainable=True, gn=False)


    z_mu = tf.concat( [zC_mu, zS_mu], axis=1)
    z_logvar = tf.concat( [zC_lv, zS_lv], axis=1)
    z_var = None
    z_prec = None

    if cfg.cfg.accumulation == cfg.AccumulationType.GVAE:
      z_var = tf.exp( z_logvar  )
      z_logvar = None
      z_prec = None

    if cfg.cfg.accumulation == cfg.AccumulationType.MLVAE:
      z_logvar = 5.0*tf.nn.tanh( z_logvar )
      z_var = None
      z_prec = None

    return z_mu, z_var, z_logvar, z_prec


def decode_mnist(z, output_size, is_train):
    use_bn = False
    use_wn = False
    use_ln = False
    use_pn = False
    use_heNorm=False
    use_gn = False
    bias = (not use_bn) and (not use_ln)

    activation = tf.nn.relu
    
    x = linear(z,  2*128, 'lin_1', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=activation, noisy_weights=False, trainable=True, gn=False)
    x = linear(x,  output_size//8*output_size//8*64, 'lin_2', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=activation, noisy_weights=False, trainable=True, gn=False)
    x = tf.reshape(x, [-1, 64, output_size//8, output_size//8 ])
    
    x = deconv2d(x, 32, 'deconv_1', is_train, k_h=5, k_w=5, d_h=2, d_w=2, out_shape=[output_size//4, output_size//4], padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=activation, gn=use_gn )
    x = deconv2d(x, 16, 'deconv_2', is_train, k_h=5, k_w=5, d_h=2, d_w=2, out_shape=[output_size//2, output_size//2], padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=activation, gn=use_gn )
    x = deconv2d(x, cfg.cfg.color_channels, 'deconv_3', is_train, k_h=5, k_w=5, d_h=2, d_w=2, out_shape=[output_size, output_size], padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=None, gn=use_gn )

   
    if cfg.cfg.binary:
      x = tf.nn.sigmoid(x)

    return x, 0


def discriminator_encode_mnist(x, is_train, drop_keep):

    input_size = int(x.get_shape()[2])
    input_depth = int(x.get_shape()[1])
    use_bn = False
    use_wn = False
    use_ln = False
    use_heNorm=False
    use_gn = False
    bias = (not use_bn) and (not use_ln)

    activation = tf.nn.relu

    z = conv2d( x, 16, 'conv_2_C_mu', is_train, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm,  activation=activation, gn=use_gn)
    z = conv2d( z, 32, 'conv_3_C_mu', is_train, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm,  activation=activation, gn=use_gn)
    z = conv2d( z, 64, 'conv_4_C_mu', is_train, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm,  activation=activation, gn=use_gn)
    z = tf.reshape(z, [-1,  int(z.get_shape()[1])* int(z.get_shape()[2])* int(z.get_shape()[3]) ])
    z = linear(z,  128, 'lin_1_C_mu', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=activation, noisy_weights=False, trainable=True, gn=False)
    z = linear(z,  cfg.cfg.ZC_dim, 'lin_2_C_mu', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=None, noisy_weights=False, trainable=True, gn=False)

    return z
  
  
def discriminator_mnist( z, is_train, drop_keep):
    use_bn = False #cfg.cfg.D2_use_bn and (not cfg.cfg.wgan2)
    use_wn = False #cfg.cfg.D2_use_wn
    use_ln = False
    use_heNorm=False
    use_gn = False #cfg.cfg.D2_use_gn
    bias = (not use_bn) and (not use_ln)

    zdim = int(z.get_shape()[1])
    
    activation = tf.nn.relu

    z = linear(z,      500, 'lin1', is_train, bias=bias, bn=use_bn, wn=use_wn, ln=use_ln, heNorm=use_heNorm, activation=tf.nn.tanh, gn=use_gn )

    logits = linear(z,       1, 'lin_last', is_train, bias=True, bn=False, wn=False, ln=False, heNorm=use_heNorm, activation=None, gn=False )

    p = logits

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_default_graph().get_name_scope())    

    return update_ops, p
    
