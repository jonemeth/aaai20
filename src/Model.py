from time import gmtime, strftime
from sys import stdout

from utils import *
import cfg

from Model_mnist import *

def sampleZ( z_mu, z_var, z_logvar, z_prec ):
  if not cfg.cfg.gaussianEncoder:
    return z_mu

  if z_logvar is not None:
    return z_mu + tf.exp(z_logvar/2.0) * tf.random_normal(shape=tf.shape(z_mu))
  if z_var is not None:
    return z_mu + tf.sqrt(z_var) * tf.random_normal(shape=tf.shape(z_mu))
  if z_prec is not None:
    return z_mu + tf.sqrt(1.0/z_prec) * tf.random_normal(shape=tf.shape(z_mu))
    
  
def encode(name, x, is_train, drop_keep):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    print( 'encode: ', x.get_shape(), flush=True )

    return encode_mnist(x, is_train, drop_keep)

def decode(name, z, output_size, is_train):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    print( 'decode: ', z.get_shape(), flush=True )
    
    return decode_mnist(z, output_size, is_train)


def likelihood( x, y_mu, y_logvar=None ):
  if cfg.cfg.decoderVariance:
    assert( False )
  
  y = y_mu
  
  if cfg.cfg.binary:
    diff = -tf.reduce_sum(x * tf.log(1e-6+y) + (1.0 - x) * tf.log(1e-6+1.0 - y), axis=[1,2,3])
  else:
    err = 0.5*(x - y)**2
    diff = tf.reduce_sum( err, axis=[1,2,3])

  return diff
  

def decoder(name, z, size, is_train ):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    y, _ = decode( 'decode', z, size, is_train )
    
    return y
  
def encoder(name, x, is_train, drop_keep ):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    return encode( 'encode', x, is_train, drop_keep)

def autoencoder(name, x, is_train, drop_keep, accum ):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    
    input_size = int(x.get_shape()[2])

    print('autoencoder', flush=True )
    
    z_mu, z_var, z_logvar, z_prec = encode( 'encode', x, is_train, drop_keep)
    
    mb_size = tf.shape(x)[0]

    z_mu_ori = z_mu
    z_var_ori = z_var
    z_logvar_ori = z_logvar
    z_prec_ori = z_prec
    z_ori = sampleZ(z_mu_ori, z_var_ori, z_logvar_ori, z_prec_ori)
    
    if cfg.cfg.accumulation == cfg.AccumulationType.MLVAE:
      assert( cfg.cfg.gaussianEncoder )
      zC_mu, zS_mu = tf.split( z_mu, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )      
      zC_lv, zS_lv = tf.split( z_logvar, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )      

      zC_mu = tf.reshape( zC_mu, [cfg.cfg.groupSize, mb_size//cfg.cfg.groupSize, cfg.cfg.ZC_dim] )
      zC_mu = tf.transpose( zC_mu, [1, 2, 0] )
      zC_lv = tf.reshape( zC_lv, [cfg.cfg.groupSize, mb_size//cfg.cfg.groupSize, cfg.cfg.ZC_dim] )
      zC_lv = tf.transpose( zC_lv, [1, 2, 0] )

      zC_va = tf.exp( zC_lv )

      zC_va_acc = 1.0 / tf.reduce_sum( 1.0 / zC_va, axis=2, keep_dims=False )
      zC_mu_acc = tf.reduce_sum( zC_mu / zC_va, axis=2, keep_dims=False ) / tf.reduce_sum( 1.0 / zC_va, axis=2, keep_dims=False )

      zC_lv_acc = tf.log( zC_va_acc )

      zC_mu_acc = tf.tile( zC_mu_acc, [cfg.cfg.groupSize, 1] )
      zC_lv_acc = tf.tile( zC_lv_acc, [cfg.cfg.groupSize, 1] )

      z_mu = tf.concat( [zC_mu_acc, zS_mu], axis = 1 )
      z_logvar = tf.concat( [zC_lv_acc, zS_lv], axis = 1 )

      z = z_mu + tf.exp(z_logvar/2.0) * tf.random_normal(shape=tf.shape(z_mu))

    elif cfg.cfg.accumulation == cfg.AccumulationType.GVAE:
      assert( cfg.cfg.gaussianEncoder )
      zC_mu, zS_mu = tf.split( z_mu, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )      
      zC_va, zS_va = tf.split( z_var, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )      

      zC_mu_acc = tf.reshape( zC_mu, [cfg.cfg.groupSize, mb_size//cfg.cfg.groupSize, cfg.cfg.ZC_dim] )
      zC_mu_acc = tf.transpose( zC_mu_acc, [1, 2, 0] )
      zC_mu_acc = tf.reduce_mean( zC_mu_acc, axis=2, keep_dims=False )        
      zC_mu_acc = tf.tile( zC_mu_acc, [cfg.cfg.groupSize, 1] )

      zC_va_acc = tf.reshape( zC_va, [cfg.cfg.groupSize, mb_size//cfg.cfg.groupSize, cfg.cfg.ZC_dim] )
      zC_va_acc = tf.transpose( zC_va_acc, [1, 2, 0] )
      zC_va_acc = tf.reduce_mean( zC_va_acc, axis=2, keep_dims=False )        
      zC_va_acc = tf.tile( zC_va_acc, [cfg.cfg.groupSize, 1] )

      z_mu = tf.concat( [zC_mu_acc, zS_mu], axis = 1 )
      z_var = tf.concat( [zC_va_acc, zS_va], axis = 1 )
        
      z = z_mu + tf.sqrt(z_var) * tf.random_normal(shape=tf.shape(z_mu))
    else:
      assert( cfg.cfg.accumulation == cfg.AccumulationType.NONE )
      assert( z_var is None or z_logvar is None )
      
      if z_var is not None:
        z = z_mu + tf.sqrt(z_var) * tf.random_normal(shape=tf.shape(z_mu))
      if z_logvar is not None:
        z = z_mu + tf.exp(z_logvar/2.0) * tf.random_normal(shape=tf.shape(z_mu))
        
        
    z = tf.cond( accum, lambda: z, lambda: z_ori )
    z = tf.placeholder_with_default(z, shape=[None, cfg.cfg.Z_dim])
    z_mu = tf.cond( accum, lambda: z_mu, lambda: z_mu_ori )
    z_var = tf.cond( accum, lambda: z_var, lambda: z_var_ori ) if z_var_ori is not None else None
    z_logvar = tf.cond( accum, lambda: z_logvar, lambda: z_logvar_ori ) if z_logvar_ori is not None else None
    z_prec = tf.cond( accum, lambda: z_prec, lambda: z_prec_ori ) if z_prec_ori is not None else None
      
    y_mu, y_logvar = decode( 'decode', z, input_size, is_train )

    diff = likelihood( x, y_mu, y_logvar )

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_default_graph().get_name_scope())

    return update_ops, y_mu, diff, z, z_mu, z_var, z_logvar, z_prec


def discriminator_encode(name, x, is_train, drop_keep):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
   with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
     return discriminator_encode_mnist(x, is_train, drop_keep)
    

def discriminator(name, z, is_train, drop_keep):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
   with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
    return discriminator_mnist( z, is_train, drop_keep)
    


def optimize( fn, var_list, lr ):
  solver_op = tf.train.AdamOptimizer( learning_rate=lr , beta1=cfg.cfg.learning_beta1, beta2=cfg.cfg.learning_beta2, epsilon=0.00000001).minimize(fn, var_list=var_list)
  return solver_op

def sample_r( x, s ):
  groupSize = cfg.cfg.groupSize
  groupNum = cfg.cfg.mb_size // groupSize

  indices = groupNum*tf.random.shuffle(tf.range(groupSize))
  for i in range(groupNum-1):
    indices = tf.concat( [indices, 1+i+groupNum*tf.random.shuffle(tf.range(groupSize))], axis=0 )

  indices = tf.reshape( indices, [groupNum, groupSize] )
  indices = tf.transpose( indices, [1, 0] )
  indices = tf.reshape( indices, [groupSize*groupNum] )

  ind_x, ind_s, _ = tf.split( indices, [groupNum, groupNum, cfg.cfg.mb_size-2*groupNum], axis=0)

  x = tf.gather(x, ind_x)
  s = tf.gather(s, ind_s)

  return x, s

def sample_r_bar( x, s ):
  groupSize = cfg.cfg.groupSize
  groupNum = cfg.cfg.mb_size // groupSize

  indices = groupNum*tf.random.shuffle(tf.range(groupSize))
  for i in range(groupNum-1):
    indices = tf.concat( [indices, 1+i+groupNum*tf.random.shuffle(tf.range(groupSize))], axis=0 )

  indices = tf.reshape( indices, [groupNum, groupSize] )
  indices = tf.transpose( indices, [1, 0] )
  indices = tf.reshape( indices, [groupSize*groupNum] )

  ind_x, ind_s, _ = tf.split( indices, [groupNum, groupNum, cfg.cfg.mb_size-2*groupNum], axis=0)
  ind_s = myshift(ind_s, groupNum, 1)

  x = tf.gather(x, ind_x)
  s = tf.gather(s, ind_s)

  return x, s



class Model:

  def __init__(self, sess, imageSource):
   with tf.variable_scope('Model', reuse=tf.AUTO_REUSE) as scope:
    self.sess = sess
    
    self.drop_keep = tf.placeholder(tf.float32) if cfg.cfg.train_drop_keep < 1.0 else None
    self.learning_rate = tf.placeholder(tf.float32, shape=[])
    self.lambda2 = tf.placeholder(tf.float32, shape=[])
    self.cur_lambda2 = 1.0
    
    self.imageSource = imageSource

    self.X = tf.placeholder_with_default(imageSource.image_batch, shape=[None, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size])

    self.is_train = tf.placeholder_with_default(False, shape=[])
    self.accum = tf.placeholder_with_default(False, shape=[])

    self.stat_ema = tf.train.ExponentialMovingAverage(decay=cfg.cfg.stat_moving_decay)
    
    self.build()

    
  def train_AE( self, learning_rate, it):
    feed_dict = {
	  self.imageSource.batch_size: cfg.cfg.mb_size
	  , self.learning_rate: learning_rate
	  , self.lambda2: self.cur_lambda2
	  , self.is_train: True
	  , self.accum: True
    }
    if self.drop_keep is not None:
      feed_dict.update({self.drop_keep: 1.0})

    _ , curMI = self.sess.run( [self.AE_update,  self.G_loss], feed_dict=feed_dict)

    
    if cfg.cfg.mutualInformation != cfg.MutualInformationType.NONE:
      self.cur_lambda2 = max(0, self.cur_lambda2 + 0.1*((curMI/cfg.cfg.Istar)-1.0))


    
  def train_D( self, learning_rate, it ):
    feed_dict = {
	  self.imageSource.batch_size: cfg.cfg.mb_size
	  , self.learning_rate: learning_rate
	  , self.is_train: True
	  , self.accum: True
    }
    if self.drop_keep is not None:
      feed_dict.update({self.drop_keep: 1.0})

    updateList = [self.D_update]

    self.sess.run(updateList, feed_dict=feed_dict)


  def build(self):
    groupNum = cfg.cfg.mb_size // cfg.cfg.groupSize

    AE_update_ops, self.rec, self.diff, self.z_sample, self.z_mu, self.z_var, self.z_logvar, self.z_prec = autoencoder('Model', self.X, is_train=self.is_train, drop_keep=self.drop_keep, accum=self.accum )

    # Creating discriminator
    _, Z_s = tf.split( sampleZ(self.z_mu, self.z_var, self.z_logvar, self.z_prec), [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )

    sample_r_x, sample_r_s = sample_r( self.X, Z_s )
    sample_r_bar_x, sample_r_bar_s = sample_r_bar( self.X, Z_s )
    cond_sg1, cond_sg2 = tf.split( discriminator_encode( 'Model', tf.concat([sample_r_x, sample_r_bar_x],axis=0), is_train=True, drop_keep=None  ), [(tf.shape(sample_r_x)[0]), (tf.shape(sample_r_bar_x)[0])], axis=0 )
      
    ZS_pos = tf.concat( [cond_sg1, sample_r_s], axis=1 )
    ZS_neg = tf.concat( [cond_sg2, sample_r_bar_s], axis=1 )
    D_input = tf.concat( [ZS_pos, ZS_neg], axis=0 )

    D_update_ops, D_output = discriminator('Model', D_input, is_train=True, drop_keep=self.drop_keep)
    self.D_fake, self.D_real = tf.split( D_output, [(tf.shape(ZS_pos)[0]), (tf.shape(ZS_neg)[0])], axis=0 )



    trainable_vars = tf.trainable_variables()
    theta_AE = [var for var in trainable_vars if ('Model' in var.name and ('encode' in var.name or 'decode' in var.name or 'resampler' in var.name))]
    theta_D = [var for var in trainable_vars if ('Model' in var.name and 'discriminator' in var.name)]

    print( 'theta_AE: ', theta_AE, flush=True )
    print( 'theta_D: ', theta_D, flush=True )

    print( 'AE_update_ops: ', AE_update_ops, flush=True )
    print( 'D_update_ops: ', D_update_ops, flush=True )


    D_real_max = tf.reduce_max( self.D_real )
    D_real_residue = self.D_real - D_real_max
    self.D_loss = -( tf.reduce_mean(self.D_fake) - (tf.log(0.000001+tf.reduce_mean(tf.exp(D_real_residue)))+D_real_max) )
    self.G_loss = -self.D_loss

    if cfg.cfg.regularization == cfg.RegularizationType.KL:
        
      if cfg.cfg.accumulation == cfg.AccumulationType.MLVAE:
        zC_mu , zS_mu = tf.split( self.z_mu, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )
        zC_logvar , zS_logvar = tf.split( self.z_logvar, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )
        
        zC_mu, _ = tf.split( zC_mu, [groupNum, cfg.cfg.mb_size-groupNum], axis=0 )
        zC_logvar, _ = tf.split( zC_logvar, [groupNum, cfg.cfg.mb_size-groupNum], axis=0 )
    
        self.KL1_loss = tf.reduce_sum( 0.5*tf.reduce_sum( tf.square(zC_mu) + tf.exp(zC_logvar) - zC_logvar - 1.0, 1) )   
        self.KL2_loss = tf.reduce_sum( 0.5*tf.reduce_sum( tf.square(zS_mu) + tf.exp(zS_logvar) - zS_logvar - 1.0, 1) )
        
        self.KL_loss = cfg.cfg.beta1*self.KL1_loss + cfg.cfg.beta2*self.KL2_loss
        self.KL_loss /= float(groupNum)
        
      elif cfg.cfg.accumulation == cfg.AccumulationType.GVAE:
        zC_mu , zS_mu = tf.split( self.z_mu, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )
        zC_var , zS_var = tf.split( self.z_var, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )
        
        zC_mu, _ = tf.split( zC_mu, [groupNum, cfg.cfg.mb_size-groupNum], axis=0 )
        zC_var, _ = tf.split( zC_var, [groupNum, cfg.cfg.mb_size-groupNum], axis=0 )
    
        self.KL1_loss = tf.reduce_sum( 0.5*tf.reduce_sum( tf.square(zC_mu) + zC_var - tf.log(0.000001+zC_var) - 1.0, 1) )   
        self.KL2_loss = tf.reduce_sum( 0.5*tf.reduce_sum( tf.square(zS_mu) + zS_var - tf.log(0.000001+zS_var) - 1.0, 1) )
        
        self.KL_loss = cfg.cfg.beta1*self.KL1_loss + cfg.cfg.beta2*self.KL2_loss
        self.KL_loss /= float(groupNum)
        
      else:
        zC_mu , zS_mu = tf.split( self.z_mu, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )
        zC_logvar , zS_logvar = tf.split( self.z_logvar, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )
        self.KL1_loss = tf.reduce_sum( 0.5*tf.reduce_sum( tf.square(zC_mu) + tf.exp(zC_logvar) - zC_logvar - 1.0, 1) )   
        self.KL2_loss = tf.reduce_sum( 0.5*tf.reduce_sum( tf.square(zS_mu) + tf.exp(zS_logvar) - zS_logvar - 1.0, 1) )
        
        self.KL_loss = tf.reduce_mean( 0.5*tf.reduce_sum( tf.square(self.z_mu) + tf.exp(self.z_logvar) - self.z_logvar - 1.0, 1) )
    else:
      assert( False )

    self.recon_loss = tf.reduce_sum( self.diff ) / float(groupNum)
    
    if cfg.cfg.mutualInformation == cfg.MutualInformationType.NONE:
      self.total_loss = self.recon_loss + self.KL_loss
    else:
      self.total_loss = self.recon_loss + self.KL_loss +  self.lambda2*self.G_loss


    recon_loss_mean_ema_op = self.stat_ema.apply( [self.recon_loss] )
    AE_update_ops += [recon_loss_mean_ema_op]
    self.recon_loss_moving = self.stat_ema.average( self.recon_loss )

    total_loss_mean_ema_op = self.stat_ema.apply( [self.total_loss] )
    AE_update_ops += [total_loss_mean_ema_op]
    self.total_loss_moving = self.stat_ema.average( self.total_loss )

    D_loss_ema_op = self.stat_ema.apply( [self.D_loss] )
    D_update_ops += [D_loss_ema_op]
    self.D_loss_moving = self.stat_ema.average( self.D_loss )

    D_real_mean = tf.reduce_mean( self.D_real )
    D_real_mean_ema_op = self.stat_ema.apply( [D_real_mean] )
    D_update_ops += [D_real_mean_ema_op]
    self.D_real_moving = self.stat_ema.average( D_real_mean )

    D_fake_mean = tf.reduce_mean( self.D_fake )
    D_fake_mean_ema_op = self.stat_ema.apply( [D_fake_mean] )
    D_update_ops += [D_fake_mean_ema_op]
    self.D_fake_moving = self.stat_ema.average( D_fake_mean )

    KL_loss_mean_ema_op = self.stat_ema.apply( [self.KL_loss] )
    AE_update_ops += [KL_loss_mean_ema_op]
    self.KL_loss_moving = self.stat_ema.average( self.KL_loss )

    KL1_loss_mean_ema_op = self.stat_ema.apply( [self.KL1_loss] )
    KL2_loss_mean_ema_op = self.stat_ema.apply( [self.KL2_loss] )
    AE_update_ops += [KL1_loss_mean_ema_op, KL2_loss_mean_ema_op]
    self.KL1_loss_moving = self.stat_ema.average( self.KL1_loss )
    self.KL2_loss_moving = self.stat_ema.average( self.KL2_loss )
    
  
    zC, zS = tf.split( self.z_sample, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )
    
    _, zC_var = tf.nn.moments( zC, axes=[1])
    self.zC_std = tf.reduce_mean( tf.sqrt(zC_var * (float(cfg.cfg.ZC_dim) /  float(cfg.cfg.ZC_dim-1))) )
    AE_update_ops += [self.stat_ema.apply( [self.zC_std] )]
    self.zC_std_moving = self.stat_ema.average( self.zC_std )

    _, zS_var = tf.nn.moments( zS, axes=[1])
    self.zS_std = tf.reduce_mean( tf.sqrt(zS_var * (float(cfg.cfg.ZS_dim) /  float(cfg.cfg.ZS_dim-1))) )
    AE_update_ops += [self.stat_ema.apply( [self.zS_std] )]
    self.zS_std_moving = self.stat_ema.average( self.zS_std )
    
    zC1, zC2, _ = tf.split( zC, [groupNum, groupNum, cfg.cfg.mb_size-2*groupNum], axis=0 )
    _, zCd_var = tf.nn.moments( zC1-zC2, axes=[1])
    self.zCd_std = tf.reduce_mean( tf.sqrt(zCd_var * (float(cfg.cfg.ZC_dim) /  float(cfg.cfg.ZC_dim-1))) )
    AE_update_ops += [self.stat_ema.apply( [self.zCd_std] )]
    self.zCd_std_moving = self.stat_ema.average( self.zCd_std )
    
    zS1, zS2, _ = tf.split( zS, [groupNum, groupNum, cfg.cfg.mb_size-2*groupNum], axis=0 )
    _, zSd_var = tf.nn.moments( zS1-zS2, axes=[1])
    self.zSd_std = tf.reduce_mean( tf.sqrt(zSd_var * (float(cfg.cfg.ZS_dim) /  float(cfg.cfg.ZS_dim-1))) )
    AE_update_ops += [self.stat_ema.apply( [self.zSd_std] )]
    self.zSd_std_moving = self.stat_ema.average( self.zSd_std )

    _, zCnd_var = tf.nn.moments( zC-tf.reverse( zC, axis=[0]), axes=[1])
    self.zCnd_std = tf.reduce_mean( tf.sqrt(zCnd_var * (float(cfg.cfg.ZC_dim) /  float(cfg.cfg.ZC_dim-1))) )
    AE_update_ops += [self.stat_ema.apply( [self.zCnd_std] )]
    self.zCnd_std_moving = self.stat_ema.average( self.zCnd_std )
    
    _, zSnd_var = tf.nn.moments( zS-tf.reverse( zS, axis=[0]), axes=[1])
    self.zSnd_std = tf.reduce_mean( tf.sqrt(zSnd_var * (float(cfg.cfg.ZS_dim) /  float(cfg.cfg.ZS_dim-1))) )
    AE_update_ops += [self.stat_ema.apply( [self.zSnd_std] )]
    self.zSnd_std_moving = self.stat_ema.average( self.zSnd_std )


    # Create solvers
    self.AE_solver_op = optimize(self.total_loss, [theta_AE], self.learning_rate)    
    self.D_solver_op = optimize(self.D_loss, [theta_D], self.learning_rate)


    with tf.control_dependencies( [self.AE_solver_op] + AE_update_ops ):
      self.AE_update = self.stat_ema.apply([])
      
    with tf.control_dependencies( [self.D_solver_op] + D_update_ops ):
      self.D_update = self.stat_ema.apply([])
      
    initialize_uninitialized( self.sess )
    self.saver = tf.train.Saver()
    
  def print_stats( self, it ):
    recon_loss_moving = self.sess.run( self.recon_loss_moving )
    total_loss_moving = self.sess.run( self.total_loss_moving )
    KL_loss_moving = self.sess.run( self.KL_loss_moving )
    KL1_loss_moving = self.sess.run( self.KL1_loss_moving )
    KL2_loss_moving = self.sess.run( self.KL2_loss_moving )
    
    D_real_moving = self.sess.run( self.D_real_moving )
    D_fake_moving = self.sess.run( self.D_fake_moving )
    D_loss_moving = self.sess.run( self.D_loss_moving )
    
    self.zC_std_curr, self.zS_std_curr, self.zCd_std_curr , self.zSd_std_curr, self.zCnd_std_curr, self.zSnd_std_curr = self.sess.run( [
				  self.zC_std_moving, self.zS_std_moving
				  , self.zCd_std_moving, self.zSd_std_moving
				  , self.zCnd_std_moving, self.zSnd_std_moving])
    
    print(strftime("%m-%d %H:%M:%S", gmtime()) + ' it: {0:6d}'.format(it)
	  , ' tot: {0:7.5f}'.format(total_loss_moving)
	  , ' rec: {0:7.5f}'.format(recon_loss_moving)
	  , ' G: {0:5.3f}'.format(KL_loss_moving)
	  , ' KL1: {0:5.3f}'.format(KL1_loss_moving)
	  , ' KL2: {0:5.3f}'.format(KL2_loss_moving)
	  , '  Dloss: {0:6.3f}'.format(D_loss_moving)
	  , '  D: {0:5.3f}'.format(D_real_moving)
	  , '/{0:5.3f}'.format(D_fake_moving)
	  , '  lambda: {0:6.3f}'.format(self.cur_lambda2)
	  , '  Isd: {0:5.3f}'.format(self.zC_std_curr)
	  , '/{0:5.3f}'.format(self.zCd_std_curr)
	  , '/{0:5.3f}'.format(self.zCnd_std_curr)
	  , '  Osd: {0:5.3f}'.format(self.zS_std_curr)
	  , '/{0:5.3f}'.format(self.zSd_std_curr)
	  , '/{0:5.3f}'.format(self.zSnd_std_curr)
	  , end = "", sep='')
    
    print( "" )

    stdout.flush()
      
      

