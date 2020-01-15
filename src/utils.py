import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use("AGG")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import cfg


def my_get_variable( name, shape=None, dtype=None, initializer=None, trainable=True, constraint=None ):
  return tf.get_variable( name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, constraint=constraint )

def initialize_uninitialized(sess):
    print(' --- initialize_uninitialized ---')
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print( [str(i.name) for i in not_initialized_vars], flush=True)# only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def to_rgb(im):
  w = im.shape[0]
  h = im.shape[1]
  ret = np.empty((w, h, 3), dtype=np.float)
  ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im
  return ret


def plot(samples, gridSize, filename, shiftMean):
  if shiftMean:
    samples = (samples+1.0)/2.0
    
  samples[ samples < 0.0 ] = 0.0
  samples[ samples > 1.0 ] = 1.0

  fig = plt.figure(figsize=(gridSize, gridSize))
  gs = gridspec.GridSpec(gridSize, gridSize)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
      ax = plt.subplot(gs[i])
      plt.axis('off')
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.set_aspect('equal')

      size = sample.shape[1]
      channels = sample.shape[0]
      
      if 1 == channels:
        sample = to_rgb( sample.reshape(size, size) )
        channels = 3
      else:
        sample = sample.transpose([1, 2, 0])

      plt.imshow(sample)

  plt.savefig(filename, bbox_inches='tight')
  plt.close(fig)





def scatterPlot(it, sess, model, imageSourceTest):
  images, labels = sess.run( [imageSourceTest.image_batch, imageSourceTest.label_batch], feed_dict={imageSourceTest.batch_size: 1000} )
  z, z_mu = sess.run( [model.z_sample, model.z_mu], feed_dict={model.X: images} )


  for i in range( 1 ):
    X = z_mu[:,2*i]
    Y = z_mu[:,2*i+1]
    plt.scatter( X, Y, c=labels, cmap=plt.cm.get_cmap('tab10') )
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.colorbar()
    plt.savefig( cfg.cfg.outdirImages+'/{}scatter{}_mean.png'.format(str(it).zfill(8),str(i).zfill(2)) )
    plt.close()

  for i in range( 1 ):
    X = z[:,2*i]
    Y = z[:,2*i+1]
    plt.scatter( X, Y, c=labels, cmap=plt.cm.get_cmap('tab10') )
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.colorbar()
    plt.savefig( cfg.cfg.outdirImages+'/{}scatter{}_sample.png'.format(str(it).zfill(8),str(i).zfill(2)) )
    plt.close()


def crossSampler(it, sess, model, imageSourceCrossSample):
  N = 1+10

  images1 = sess.run( imageSourceCrossSample.image_batch, feed_dict={imageSourceCrossSample.batch_size: N-1} )
  images2 = sess.run( imageSourceCrossSample.image_batch, feed_dict={imageSourceCrossSample.batch_size: N-1} )
  z1 = sess.run( model.z_sample, feed_dict={model.X: images1} )
  z2 = sess.run( model.z_sample, feed_dict={model.X: images2} )
  z1C, z1O = np.split( z1, [cfg.cfg.ZC_dim], axis=1 )
  z2C, z2O = np.split( z2, [cfg.cfg.ZC_dim], axis=1 )

  R = np.zeros( ( (N)*(N), cfg.cfg.Z_dim ) )

  for i in range(N):
    for j in range(N):
      if i==0 and j==0:
        continue
      ix = i*N+j

      if j==0:
        R[ix, 0:cfg.cfg.Z_dim] = z1[i-1, 0:cfg.cfg.Z_dim]
        continue

      if i==0:
        R[ix, 0:cfg.cfg.Z_dim] = z2[j-1, 0:cfg.cfg.Z_dim]
        continue

      R[ix, 0:cfg.cfg.ZC_dim] = z1C[i-1, 0:cfg.cfg.ZC_dim]
      R[ix, cfg.cfg.ZC_dim:cfg.cfg.Z_dim] = z2S[j-1, 0:cfg.cfg.ZS_dim]

  samples = sess.run(model.rec, feed_dict={model.z_sample: R})
  samples = np.reshape( samples, (N, N, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size) )
  
  images2 = np.reshape( images2, (1, N-1, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size) )
  images2 = np.concatenate( (np.ones((1, 1, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size)), images2), axis= 1)
  samples = np.concatenate( (images2, samples), axis=0 )
  
  images1 = np.reshape( images1, (N-1, 1, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size) )
  images1 = np.concatenate( (np.ones((2, 1, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size)), images1), axis= 0)
  samples = np.concatenate( (images1, samples), axis=1 )
  
  samples = np.reshape( samples, ((N+1)*(N+1), cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size) )
  
  plot(samples, N+1, cfg.cfg.outdirImages+'/{}i.png'.format(str(it).zfill(8)), cfg.cfg.shiftMean)
  
def crossSampler2(nr, N, it, sess, model, imageSourceCrossSample):
  images1 = sess.run( imageSourceCrossSample.image_batch, feed_dict={imageSourceCrossSample.batch_size: N} )

  if N < cfg.cfg.groupSize:
    images1 = np.concatenate( (images1, np.ones((cfg.cfg.groupSize-N, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size))), axis= 0)
  z1 = sess.run( model.z_sample, feed_dict={model.X: images1} )
  if N < cfg.cfg.groupSize:
    z1 = z1[0:N, :]
    images1 = images1[0:N, :, :, :]


  z1C, z1S = np.split( z1, [cfg.cfg.ZC_dim], axis=1 )

  R = np.zeros( ( (N)*(N), cfg.cfg.Z_dim ) )

  for i in range(N):
    for j in range(N):
      ix = i*N+j

      R[ix, 0:cfg.cfg.ZC_dim] = z1C[i, 0:cfg.cfg.ZC_dim]
      R[ix, cfg.cfg.ZC_dim:cfg.cfg.Z_dim] = z1S[j, 0:cfg.cfg.ZS_dim]

  samples = sess.run(model.rec, feed_dict={model.z_sample: R})
  samples = np.reshape( samples, (N, N, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size) )
  
  images2 = np.reshape( images1, (1, N, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size) )
  samples = np.concatenate( (images2, samples), axis=0 )
  
  images2 = np.reshape( images1, (N, 1, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size) )
  images2 = np.concatenate( (np.ones((1, 1, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size)), images2), axis= 0)
  samples = np.concatenate( (images2, samples), axis=1 )
  
  samples = np.reshape( samples, ((N+1)*(N+1), cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size) )
  
  plot(samples, N+1, cfg.cfg.outdirImages+'/{}j{}.png'.format(str(it).zfill(8), str(nr)), cfg.cfg.shiftMean)
