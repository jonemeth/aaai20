import tensorflow as tf
import numpy as np
import math
from utils import *
import cfg

def mse_loss(pred, data):
  loss_val = tf.sqrt(2 * tf.nn.l2_loss(pred - data)) / tf.to_float( tf.shape(data)[0] )
  return loss_val

def upscale(x, scale=2.0):
    C = int(x.get_shape()[1])
    H = int(x.get_shape()[2] * scale)
    W = int(x.get_shape()[3] * scale)

    x = tf.transpose( x, [0, 2, 3, 1] )
    x = tf.image.resize_nearest_neighbor(x, (H, W) )
    return tf.transpose( x, [0, 3, 1, 2] )

def myshuf( z ):
    N = tf.cast( tf.floor( tf.random_uniform( [1], 1.0, 1.0 ) ), tf.int32 )                                                                   
    A, B = tf.split( z, tf.concat( [N, tf.shape(z)[0]-N], axis=0 ), axis=0 )                                                                             
#      A, B = tf.split( z, [N, tf.shape(z)[0]-N], axis=0 )                                                                             
    z = tf.concat( [B, A], axis=0 ) 
    return z

def myshift( z, N, offset ):
    A, B = tf.split( z, [offset, N-offset], axis=0 )                                                                             
    z = tf.concat( [B, A], axis=0 ) 
    return z

def downscale(x, scale=2):
  return tf.layers.average_pooling2d(x, scale, scale, data_format='channels_first')

def prelu(_x, name="prelu"):
  if( len( _x.get_shape() ) == 2 ):
    shape = [1, int(_x.get_shape()[1])]
  else:
    shape = [1, int(_x.get_shape()[1]), 1, 1]

  _alpha = my_get_variable(name, shape=shape, dtype=_x.dtype, initializer=tf.constant_initializer(0.25), constraint=lambda x: tf.clip_by_value(x, 0.01, 0.5) )

  return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)
#  return tf.maximum(_x, _alpha*_x)

def prelu2(_x, name="prelu2"):
  
  if( len( _x.get_shape() ) == 2 ):
    shape = [1, int(_x.get_shape()[1])]
  else:
    shape = [1, int(_x.get_shape()[1]), 1, 1]
  
  alpha1 = my_get_variable(name, shape=shape, dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
  alpha2 = my_get_variable(name, shape=shape, dtype=_x.dtype, initializer=tf.constant_initializer(0.1))

  ass_op1 = tf.assign( alpha1, tf.clip_by_value( alpha1, 0.05, 1.0 ) )
  ass_op2 = tf.assign( alpha2, tf.clip_by_value( alpha2, 0.05, 1.0 ) )

  with tf.control_dependencies([ass_op1]):
    alpha1_ = tf.identity( alpha1 )

  with tf.control_dependencies([ass_op2]):
    alpha2_ = tf.identity( alpha2 )
  
  return tf.minimum( 1.0+tf.maximum(0.0,alpha2_*(_x-1.0)), tf.maximum(_x, alpha1_*_x) )

def lrelu(x, leak=0.01):
  return tf.nn.leaky_relu( x, alpha=leak )

def upperbound(x, b, leak=0.2):
  x = x - b
  x = tf.minimum(x, leak*x)
  x = x + b
  return x

def swish(x):
  return x*tf.nn.sigmoid(x)

def elu(x):
  return tf.where(x >= 0.0, x, tf.exp(x) - 1)

def maxout(inputs, num_units, axis=1):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

# Orthogonal Permutation Linear Unit
def oplu(inputs, group_size=2, axis=-1):
    if( len( inputs.get_shape() ) == 4 ):
      inputs = tf.transpose( inputs, [0, 2, 3, 1] )

    shape = inputs.get_shape().as_list()

    if shape[0] is None:
        shape[0] = -1

    num_channels = shape[axis]
    if num_channels % group_size:
        raise ValueError('number of features({}) is not '
                         'a multiple of group_size({})'.format(num_channels, group_size))
    shape[axis] = num_channels // group_size
    shape += [group_size]
    outputs1 = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    outputs2 = tf.reduce_min(tf.reshape(inputs, shape), -1, keep_dims=False)
    outputs = tf.concat( [outputs1, outputs2], axis=-1 )

    if( len( inputs.get_shape() ) == 4 ):
      outputs = tf.transpose( outputs, [0, 3, 1, 2] )

    return outputs


def pixelwise_norm( x ):
  return x / tf.sqrt(tf.reduce_mean( x*x, axis=1, keep_dims=True ) + 1.0e-8)

def layernorm_OLD(inputs):
  with tf.variable_scope("ln"):
    scope = tf.get_variable_scope()
    return tf.contrib.layers.layer_norm(
        inputs, center=True, scale=True,
	reuse=scope.reuse,
	trainable=True,
	scope=scope )

def layernorm(x, epsilon=1e-5, use_bias=True, use_scale=True, data_format='NCHW'):
  with tf.variable_scope("ln"):
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]

    mean, var = tf.nn.moments(x, list(range(1, len(shape))), keep_dims=True)

    if data_format == 'NCHW':
        chan = shape[1]
        new_shape = [1, chan, 1, 1]
    else:
        chan = shape[-1]
        new_shape = [1, 1, 1, chan]
    if ndims == 2:
        new_shape = [1, chan]

    if use_bias:
        beta = my_get_variable('beta', [chan], initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
    else:
        beta = tf.zeros([1] * ndims, name='beta')
    if use_scale:
        gamma = my_get_variable('gamma', [chan], initializer=tf.constant_initializer(1.0))
        gamma = tf.reshape(gamma, new_shape)
    else:
        gamma = tf.ones([1] * ndims, name='gamma')

    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='output')


def batchnorm2( x, is_train, center=True, scale=True):
  inputs = tf.layers.batch_normalization(
      name="bn",
      reuse=(not is_train),
      inputs=x,
      axis=1,
      momentum=0.999,
      epsilon=0.0001,
      center=center,
      scale=scale,
      training=is_train,
      fused=False,
      renorm=False,
      renorm_momentum=0.99,
      renorm_clipping={'rmax':3.0, 'rmin':1.0/3.0, 'dmax':5.0},
      gamma_initializer=tf.ones_initializer())

  return inputs

def groupnorm(x, is_train, activation=None, G=8, esp=1e-8):
  gain = 2.0 if activation is not None else 1.0
  if len( x.get_shape() ) == 4:
    N, C, H, W = x.get_shape().as_list()
#    G = C // 8
    G = min(G, C)
    x = tf.reshape(x, [-1, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + esp)
    # per channel gamma and beta
    gamma = my_get_variable('gamma', [C], initializer=tf.constant_initializer(gain))
    beta = my_get_variable('beta', [C], initializer=tf.constant_initializer())
    gamma = tf.reshape(gamma, [1, C, 1, 1])
    beta = tf.reshape(beta, [1, C, 1, 1])
    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
  else:
    N, C = x.get_shape().as_list()
#    G = C // 8
    G = min(G, C)
    x = tf.reshape(x, [-1, G, C // G])
    mean, var = tf.nn.moments(x, [2], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + esp)
    # per channel gamma and beta
    gamma = my_get_variable('gamma', [C], initializer=tf.constant_initializer(gain))
    beta = my_get_variable('beta', [C], initializer=tf.constant_initializer())
    gamma = tf.reshape(gamma, [1, C])
    beta = tf.reshape(beta, [1, C])

    output = tf.reshape(x, [-1, C]) * gamma + beta
    
  return output

def groupnorm2( x, is_train ):
  groups = int(x.get_shape()[1]) // 8
  y = tf.contrib.layers.group_norm(
    x,
    groups=groups,
    channels_axis=-3,
    reduction_axes=(-2, -1),
    center=True,
    scale=True,
    epsilon=1e-06,
    activation_fn=None,
    param_initializers=None,
    reuse=(not is_train),
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None,
    mean_close_to_zero=False
  )
  return y


def batchnorm( x, is_train, center=True, scale=True):
  with tf.variable_scope("batchnorm", reuse=tf.AUTO_REUSE):
    scope = tf.get_variable_scope()

    n_out = int(x.get_shape()[1])
    
    needReshape = 2 == len(x.get_shape())
    
    if needReshape:
      x = tf.reshape( x, [-1, n_out, 1, 1] )
    
    x = tf.transpose( x, [0, 2, 3, 1] )

    beta = my_get_variable('beta', [n_out], initializer=tf.constant_initializer(0.0) ) if center else None
    gamma = my_get_variable('gamma', [n_out], initializer=tf.constant_initializer(1.0) ) if scale else None

    mean_moving = my_get_variable('batch_mean_avg', [1, 1, 1, n_out], initializer=tf.constant_initializer(0.0), trainable=False )
    var_moving = my_get_variable('batch_var_avg',  [1, 1, 1, n_out], initializer=tf.constant_initializer(1.0), trainable=False )

    decay = 0.999
    epsilon = 0.001

    if is_train:
      batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments', keep_dims=True)

      ass_op1 = tf.assign(mean_moving, mean_moving*decay+batch_mean*(1-decay))
      ass_op2 = tf.assign(var_moving, var_moving*decay+batch_var*(1-decay))

      with tf.control_dependencies([ass_op1, ass_op2]):
        output = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)

    else:
      output = tf.nn.batch_normalization(x, mean_moving, var_moving, beta, gamma, epsilon)


    output = tf.transpose( output, [0, 3, 1, 2] )
    if needReshape:
      output = tf.reshape( output, [-1, n_out] )

    return output


def bengio_init_conv(filter_size, input_dim, output_dim, stride, he_init=True, uniform=True):
        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev,
                high=stdev,
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)

        if he_init:
            filters_stdev = np.sqrt(12./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(6./(fan_in+fan_out))

        if uniform:
          filter_values = uniform(
            1.0*filters_stdev,
            (filter_size, filter_size, input_dim, output_dim)
          )
          return filter_values
        else:
          filters_stdev *= math.sqrt(1.0/3.0)
          return tf.initializers.random_normal(stddev=filters_stdev)

def bengio_init_deconv(filter_size, input_dim, output_dim, stride, he_init=True, uniform=True):
        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev,
                high=stdev,
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**2 / (stride**2)
        fan_out = output_dim * filter_size**2

        if he_init:
            filters_stdev = np.sqrt(12./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(6./(fan_in+fan_out))

        if uniform:
          filter_values = uniform(
              1.0*filters_stdev,
              (filter_size, filter_size, output_dim, input_dim)
          )
          return filter_values
        else:
          filters_stdev *= math.sqrt(1.0/3.0)
          return tf.initializers.random_normal(stddev=filters_stdev)

def bengio_init_linear(input_dim, output_dim, uniform=True):
        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev,
                high=stdev,
                size=size
            ).astype('float32')

        filters_stdev = np.sqrt(6./(input_dim+output_dim))

        if uniform:
          weight_values = uniform( 
               1.0*filters_stdev, (input_dim, output_dim) )
          return weight_values
        else:
          filters_stdev *= math.sqrt(1.0/3.0)
          return tf.initializers.random_normal(stddev=filters_stdev)


def he_init_conv(filter_size, input_dim, output_dim, stride, gain, unif=False):
  fan_in = input_dim * filter_size**2
  fan_out = output_dim * filter_size**2 / (stride**2)
  filters_stdev = math.sqrt(gain*2.0/(fan_in+fan_out))
  if unif:
    filters_stdev *= 1.73205080757 #sqrt(3)
    return tf.initializers.random_uniform(minval=-filters_stdev, maxval=filters_stdev)
  else:
    return tf.initializers.truncated_normal(stddev=filters_stdev)

def he_init_deconv(filter_size, input_dim, output_dim, stride, gain, unif=False):
  fan_in = input_dim * filter_size**2 / (stride**2)
  fan_out = output_dim * filter_size**2
  filters_stdev = math.sqrt(gain*2.0/(fan_in+fan_out))
  if unif:
    filters_stdev *= 1.73205080757 #sqrt(3)
    return tf.initializers.random_uniform(minval=-filters_stdev, maxval=filters_stdev)
  else:
    return tf.initializers.truncated_normal(stddev=filters_stdev)

def he_init_linear(input_dim, output_dim, gain, unif=False):
  filters_stdev = math.sqrt(gain*2.0/(input_dim+output_dim))
  if unif:
    filters_stdev *= 1.73205080757 #sqrt(3)
    return tf.initializers.random_uniform(minval=-filters_stdev, maxval=filters_stdev)
  else:
    return tf.initializers.truncated_normal(stddev=filters_stdev)



def conv2d(x, output_dim, name, is_train, k_h=5, k_w=5, d_h=2, d_w=2, bias=False, padding='SAME', bn=False, wn=False, activation=None, ln=False, heNorm=False, bn_trans=True, gainMult=1.0, pr=True, pn=False, gn=False):
  with tf.variable_scope(name):
    input_dim = int(x.get_shape()[1])

    gain = 1.0 if activation is None else 2.0
    gain *= gainMult
    if heNorm:
      w = my_get_variable('w', shape=[k_h, k_w, input_dim, output_dim], initializer=tf.initializers.random_normal() )
      fan_in = input_dim * k_h*k_w
      fan_out = output_dim * k_h*k_w / (d_h*d_w)
      cv = math.sqrt(gain/(fan_in))#+fan_out))
      c = tf.constant(cv, dtype=tf.float32)
      w = w * c
    else:
      w = my_get_variable('w', shape=[k_h, k_w, input_dim, output_dim], initializer=he_init_conv( k_h, input_dim, output_dim, d_h, gain ) )

    if wn:
      fan_in = input_dim * k_h**2
      fan_out = output_dim * k_h**2 / (d_h**2)
      filters_stdev = math.sqrt(gain*2.0/(fan_in+fan_out))
      expected_norm = filters_stdev * math.sqrt( float(k_h * k_w * input_dim) )
      
      w = tf.nn.l2_normalize(w, [0, 1, 2])
      g = my_get_variable("g", [1, 1, 1, output_dim], initializer=tf.constant_initializer(expected_norm))
      w = g*w

    output = tf.nn.conv2d(x, w, strides=[1, 1, d_h, d_w], padding=padding, data_format='NCHW')

    if bias:
      bias = my_get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
      output = tf.nn.bias_add(output, bias, data_format='NCHW')


    if ln:
      output = layernorm(output)

    if bn:
      output = batchnorm(output, is_train, center=bn_trans, scale=bn_trans )

    if gn:
      output = groupnorm(output, is_train, activation)
    if activation:
      output = activation(output)


    if pn:
      output = pixelwise_norm( output )
      
    if pr:
      print( 'conv:   [' + str(int(x.get_shape()[1])) +','+ str(int(x.get_shape()[2])) +','+ str(int(x.get_shape()[3]))
	+ '] -> [' 
	+ str(int(output.get_shape()[1])) +','+ str(int(output.get_shape()[2])) +','+ str(int(output.get_shape()[3])) +  ']' )
    
    return output


def deconv2d(x, output_dim, name, is_train, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME', bias=False, bn=False, wn=False, activation=None, ln=False, heNorm=False, pr=True, out_shape=None, pn=False, gn=False):
  with tf.variable_scope(name):
    input_dim = int(x.get_shape()[1])
    batch_size = tf.shape(x)[0]

    if out_shape is None:
      output_shape = [batch_size, output_dim, int(x.get_shape()[2])*d_h, int(x.get_shape()[3])*d_w]
    else:
      output_shape = [batch_size, output_dim, out_shape[0], out_shape[1]]

    if pr:
      print( 'deconv: [' + str(int(x.get_shape()[1])) +','+ str(int(x.get_shape()[2])) +','+ str(int(x.get_shape()[3]))
	+ '] -> [' 
	+ str(output_dim) +','+ str(output_shape[2]) +','+  str(output_shape[3]) +  ']' )
    
    gain = 1.0 if activation is None else 2.0
    if heNorm:
      w = my_get_variable('w', shape=[k_h, k_w, output_dim, input_dim], initializer=tf.initializers.random_normal() )
      fan_in = input_dim * k_h*k_w / (d_h*d_w)
      fan_out = output_dim * k_h*k_w
      cv = math.sqrt(gain/(fan_in))#+fan_out))
      c = tf.constant(cv, dtype=tf.float32)
      w = w * c
    else:
      w = my_get_variable('weights', shape=[k_h, k_w, output_dim, input_dim], initializer=he_init_deconv( k_h, input_dim, output_dim, d_h, gain ) )

    if wn:
      fan_in = input_dim * k_h**2 / (d_h**2)
      fan_out = output_dim * k_h**2
      filters_stdev = math.sqrt(gain*2.0/(fan_in+fan_out))
      expected_norm = filters_stdev * math.sqrt( float(k_h * k_w * input_dim) )
      
      w = tf.nn.l2_normalize(w, [0, 1, 3])
      g = my_get_variable("g", [1, 1, output_dim, 1], initializer=tf.constant_initializer(expected_norm))
      w = g*w

    output = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, 1, d_h, d_w], data_format='NCHW', padding=padding)

    if bias:
      bias = my_get_variable('bias', [output_shape[1]], initializer=tf.constant_initializer(0.0))
      output = tf.nn.bias_add(output, bias, data_format='NCHW')

    output = tf.reshape(output, output_shape)


    if ln:
      output = layernorm(output)

    if bn:
      output = batchnorm(output, is_train)

    if gn:
      output = groupnorm(output, is_train, activation)
    if activation:
      output = activation(output)


    if pn:
      output = pixelwise_norm( output )

    return output


def linear(x, output_dim, name, is_train, trainable=True, bias=False, bn=False, wn=False, activation=None, ln=False, heNorm=False, bn_trans=True, noisy_weights=False, pr=True, pn=False, gn=False):
  shape = x.get_shape().as_list()
  
  with tf.variable_scope(name):

    input_dim = int(x.get_shape()[1])
    
    gain = 1.0 if activation is None else 2.0
    if heNorm:
      w = my_get_variable('w', shape=[input_dim, output_dim], initializer=tf.initializers.random_normal(), trainable=trainable)
      cv = math.sqrt(gain/float(input_dim))#+output_dim))
      c = tf.constant(cv, dtype=tf.float32)
      w = w * c
    else:
      w = my_get_variable("weights", shape=[input_dim, output_dim], initializer= he_init_linear(input_dim, output_dim, gain), trainable=trainable)


    if wn:
      filters_stdev = math.sqrt(gain*2.0/(input_dim+output_dim))
      expected_norm = filters_stdev * math.sqrt( float(input_dim) )
      
      w = tf.nn.l2_normalize(w, [0])
      g = my_get_variable("g", [1, output_dim], initializer=tf.constant_initializer(expected_norm), trainable=trainable)
      w = g*w

    if noisy_weights:
      w = w + tf.random_normal( [input_dim, output_dim] ) * (math.sqrt(2.0/input_dim) * 0.01)

    output = tf.matmul(x, w)

    if bias:
      bias = my_get_variable("bias", [output_dim], initializer=tf.constant_initializer(0.0), trainable=trainable)
      output = tf.nn.bias_add(output, bias)

    if ln:
      output = layernorm(output)

    if bn:
      output = batchnorm(output, is_train, center=bn_trans, scale=bn_trans )

    if gn:
      output = groupnorm(output, is_train, activation)
    if activation:
      output = activation(output)
      

    if pr:
      print( 'linear: [' + str(int(x.get_shape()[1])) + '] -> [' + str(output_dim) + ']' )
       
    if pn:
      output = pixelwise_norm( output )

    return output
