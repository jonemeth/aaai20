

import tensorflow as tf
import numpy as np

import cfg
from ops import *

class LatentSource:
  def __init__(self):
    self.batch_size = tf.placeholder(tf.int32)
    self.latent_batch = tf.random_normal( [self.batch_size, cfg.cfg.Z_dim] )

  def sample( self, sess, batch_size ):
    return sess.run( self.latent_batch, feed_dict={self.batch_size: batch_size} )

  def crossSample( self, sess, N ):
    zC = tf.random_normal( [N, cfg.cfg.ZC_dim] )
    zS = tf.random_normal( [N, cfg.cfg.ZS_dim] )
    
    Z = tf.concat( [zC, zS], axis=1 )
    L = sess.run( Z )

    R = np.zeros( (N*N, cfg.cfg.Z_dim) )

    for i in range(N):
      for j in range(N):
        ix = i*N+j
        R[ix, 0:cfg.cfg.ZC_dim] = L[i, 0:cfg.cfg.ZC_dim]
        R[ix, cfg.cfg.ZC_dim:cfg.cfg.Z_dim] = L[j, cfg.cfg.ZC_dim:cfg.cfg.Z_dim]

    return R
