
import os
import math
import sys
from time import gmtime, strftime
import time
import argparse
import pickle

import tensorflow as tf
import numpy as np
from sklearn import svm

from utils import *
import ops
from ImageSource import *
from Model import *



parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='', help="Experiment: mnist or chairs or msceleb")
parser.add_argument('--modeldir', type=str, default='.', help="model directory")
parser.add_argument('--cfgfile', type=str, default='.', help="config pickle file")
parser.add_argument('--outfile', type=str, default='test_chairs.txt', help="output file")
parser.add_argument('--decision_function_shape', type=str, default='ovo', help="svm decision_function_shape")


FLAGS = parser.parse_args()

import cfg
cfg.config(FLAGS.experiment)

with open( FLAGS.cfgfile, 'rb') as f:
  cfg.cfg = pickle.load(f)


def optimize( fn, var_list, lr ):
  solver_op = tf.train.AdamOptimizer( learning_rate=lr , beta1=cfg.cfg.learning_beta1, beta2=cfg.cfg.learning_beta2, epsilon=0.000000000001).minimize(fn, var_list=var_list)
  return solver_op



sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.force_gpu_compatible = True
sess = tf.Session(config=sess_config)

if FLAGS.experiment == 'mnist':
  rootDir = '../mnist'
  imageSourceTrain = ImageSource( sess, rootDir+'/training', rootDir+'/training/training.cl.csv' )
  imageSourceTest = ImageSource( sess, rootDir+'/testing', rootDir+'/testing/testing.csv' )
  cfg.cfg.numClasses = 10
  num_train_examples = 5000
  num_test_examples = 10000

image_batch = tf.placeholder(tf.float32, shape=[None, cfg.cfg.color_channels, cfg.cfg.target_size, cfg.cfg.target_size])
label_batch = tf.placeholder(tf.int32, shape=[None])

labels_one_hot = tf.one_hot( label_batch, cfg.cfg.numClasses )


with tf.variable_scope('Model', reuse=tf.AUTO_REUSE) as scope:
 with tf.variable_scope('Model', reuse=tf.AUTO_REUSE) as scope:
  z_mu, z_va, z_lv, z_pr = encode( 'encode', image_batch, is_train=True, drop_keep=None)

assert( z_va is None or z_lv is None )

if z_va is not None:
  z = z_mu + tf.sqrt( z_va ) * tf.random_normal(shape=tf.shape(z_mu))
elif z_lv is not None:
  z = z_mu + tf.exp( z_lv/2.0 ) * tf.random_normal(shape=tf.shape(z_mu))
elif z_pr is not None:
  assert( z_va is None and z_lv is None )
  z = z_mu + tf.sqrt( 1.0/z_pr ) * tf.random_normal(shape=tf.shape(z_mu))

with tf.variable_scope('Model', reuse=tf.AUTO_REUSE) as scope:
 with tf.variable_scope('Model', reuse=tf.AUTO_REUSE) as scope:
  recon, _ = decode( 'decode', z, cfg.cfg.target_size, False )

simil = likelihood( image_batch, recon )
sumdiff = tf.reduce_sum( simil )

zC, zS = tf.split( z, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )
zC_mu, zS_mu = tf.split( z_mu, [cfg.cfg.ZC_dim, cfg.cfg.ZS_dim], axis=1 )

trainable_vars = tf.global_variables()
theta_model = [var for var in trainable_vars if ('Model' in var.name and ('encode' in var.name or 'decode' in var.name))]

initialize_uninitialized( sess )

saver = tf.train.Saver(theta_model)
saver.restore(sess, FLAGS.modeldir+'/model.ckpt')
with open( FLAGS.modeldir + '/iter.txt' ) as f:
  iters = int(f.read()) 

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
time.sleep(2)

trainable_vars = tf.trainable_variables()
theta_CL = [var for var in trainable_vars if ('test/class' in var.name)]


print( 'Start...', flush=True )

if os.path.exists(FLAGS.outfile):
  os.remove( FLAGS.outfile )

for round in range(1):
    
  sess.run(tf.variables_initializer(theta_CL))

  labelList = np.zeros( (0) )
  zClist = np.zeros( (0, cfg.cfg.ZC_dim ) )
  zSlist = np.zeros( (0, cfg.cfg.ZS_dim ) )
  zC_mu_list = np.zeros( (0, cfg.cfg.ZC_dim ) )
  zS_mu_list = np.zeros( (0, cfg.cfg.ZS_dim ) )

        
  for it in range(num_train_examples//50):
    images, labels = sess.run( [imageSourceTrain.image_batch, imageSourceTrain.label_batch], {imageSourceTrain.batch_size: 50} )
  
    currZC, currZS, currZC_mu, currZS_mu =  sess.run([zC, zS, zC_mu, zS_mu], feed_dict={image_batch:images, label_batch:labels})

    zClist=np.concatenate( (zClist, currZC), axis=0 )
    zSlist=np.concatenate( (zSlist, currZS), axis=0 )
    zC_mu_list=np.concatenate( (zC_mu_list, currZC_mu), axis=0 )
    zS_mu_list=np.concatenate( (zS_mu_list, currZS_mu), axis=0 )
    labelList=np.concatenate( (labelList, labels), axis=0 )

  clf_C = svm.SVC(gamma='scale', decision_function_shape=FLAGS.decision_function_shape)
  clf_C.fit(zClist, labelList) 
  clf_S = svm.SVC(gamma='scale', decision_function_shape=FLAGS.decision_function_shape)
  clf_S.fit(zSlist, labelList) 
  
  clfC_mu = svm.SVC(gamma='scale', decision_function_shape=FLAGS.decision_function_shape)
  clfC_mu.fit(zC_mu_list, labelList) 
  clfS_mu = svm.SVC(gamma='scale', decision_function_shape=FLAGS.decision_function_shape)
  clfS_mu.fit(zS_mu_list, labelList) 

  sumHitsC = 0
  sumHitsS = 0
  sumHitsC_mu = 0
  sumHitsS_mu = 0
  alldiff = 0
  
  for it in range(num_test_examples//50):
    images, labels = sess.run( [imageSourceTest.image_batch, imageSourceTest.label_batch], {imageSourceTest.batch_size: 50} )
  
    sumdiff_ = sess.run(sumdiff, feed_dict={image_batch:images, label_batch:labels})
    alldiff += sumdiff_
  
    currZC, currZS, currZC_mu, currZS_mu =  sess.run([zC, zS, zC_mu, zS_mu], feed_dict={image_batch:images, label_batch:labels})
    
    predC = clf_C.predict(currZC)
    predS = clf_S.predict(currZS)
    predC_mu = clfC_mu.predict(currZC_mu)
    predS_mu = clfS_mu.predict(currZS_mu)
    
    hitsC = np.sum( predC == labels )
    hitsS = np.sum( predS == labels )
    hitsC_mu = np.sum( predC_mu == labels )
    hitsS_mu = np.sum( predS_mu == labels )
    
    sumHitsC += hitsC
    sumHitsS += hitsS
    sumHitsC_mu += hitsC_mu
    sumHitsS_mu += hitsS_mu
    
  accC = sumHitsC / float(num_test_examples)
  accS = sumHitsS / float(num_test_examples)
  accC_mu = sumHitsC_mu / float(num_test_examples)
  accS_mu = sumHitsS_mu / float(num_test_examples)
  meandiff = alldiff / float(num_test_examples)
  
  with open( FLAGS.outfile, 'a') as out:
    out.write(str(accC) + ' ')
    out.write(str(accS) + ' ')
    out.write(str(accC_mu) + ' ')
    out.write(str(accS_mu) + ' ')
    out.write(str(meandiff) + '\n')




coord.request_stop()
coord.join(threads)

