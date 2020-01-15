import tensorflow as tf
import numpy as np
import time
import os
import math
import sys
from sys import stdout
import shutil
from time import gmtime, strftime
import matplotlib                                                                                                                                    
matplotlib.use("AGG")                                                                                                                                
import matplotlib.pyplot as plt                                                                                                                      
import matplotlib.gridspec as gridspec 
import argparse
import pickle
import random

from utils import *
from ImageSource import *
from ImageGroupSource import *
from LatentSource import *
from Model import *
import ops
import cfg




parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='', help="Experiment: mnist or chairs or msceleb")
parser.add_argument('--dc', type=int, default=10, help="content dimensions")
parser.add_argument('--ds', type=int, default=10, help="style dimensions")
parser.add_argument('--ac', type=str, default="NONE", help="accumulation")
parser.add_argument('--gs', type=int, default=2, help="group size")
parser.add_argument('--mi', type=str, default="NONE", help="mutual information minimization type")
parser.add_argument('--b1', type=float, default=1.0, help="beta1")
parser.add_argument('--b2', type=float, default=1.0, help="beta2")
parser.add_argument('--outdir', type=str, default="out", help="output directory")
parser.add_argument('--resume', type=bool, default=False, help="resume training")

FLAGS = parser.parse_args()

cfg.config(FLAGS.experiment)

cfg.cfg.ZC_dim = FLAGS.dc
cfg.cfg.ZS_dim = FLAGS.ds
cfg.cfg.Z_dim = cfg.cfg.ZC_dim+cfg.cfg.ZS_dim

cfg.cfg.accumulation = cfg.AccumulationType[FLAGS.ac]
cfg.cfg.groupSize = FLAGS.gs
cfg.cfg.mutualInformation = cfg.MutualInformationType[FLAGS.mi]
cfg.cfg.beta1 = FLAGS.b1
cfg.cfg.beta2 = FLAGS.b2

cfg.cfg.outdir = FLAGS.outdir
cfg.cfg.outdirImages = cfg.cfg.outdir + '/images'
cfg.cfg.save_path = cfg.cfg.outdir + '/saves'

if not os.path.exists(cfg.cfg.outdir):
  os.makedirs(cfg.cfg.outdir)
if not os.path.exists(cfg.cfg.outdirImages):
  os.makedirs(cfg.cfg.outdirImages)

with open( cfg.cfg.outdir + '/flags.txt', 'w' ) as f:
  f.write(str(FLAGS))


with open( cfg.cfg.outdir + '/cfg.txt', 'w' ) as f:
  f.write(str(vars(cfg.cfg)))
with open( cfg.cfg.outdir + '/cfg.dat', 'wb' ) as f:
  pickle.dump(cfg.cfg, f)


print(str(FLAGS))
print(str(vars(cfg.cfg)))

tf.reset_default_graph()
random.seed(cfg.cfg.seed)
tf.random.set_random_seed(seed=cfg.cfg.seed)
np.random.seed(seed=cfg.cfg.seed)


sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.force_gpu_compatible = True
sess = tf.Session(config=sess_config)



if 'mnist' == FLAGS.experiment:
    rootDir = '../mnist'
    imageSource = ImageGroupSource( sess, rootDir+'/training', rootDir+'/training/training-groups-'+str(cfg.cfg.groupSize)+'.csv' )
    imageSourceTest = ImageSource( sess, rootDir+'/testing', rootDir+'/testing/testing.csv' )
    imageSourceCrossSample = ImageSource( sess, rootDir+'/testing', rootDir+'/testing/testing-crossSample.csv', shuffle=False)
else:
    print('Unknown FLAGS.experiment')


# Latent Source
latentSource = LatentSource( )
latents_for_sampling = latentSource.sample( sess, cfg.cfg.sampleGridSize*cfg.cfg.sampleGridSize)
latents_for_crossSampling = latentSource.crossSample( sess, cfg.cfg.sampleGridSize)

model = Model( sess, imageSource)


current_learning_rate = cfg.cfg.initial_learning_rate
it = 0
restored_it = -1

if not os.path.exists(cfg.cfg.save_path):
  os.makedirs(cfg.cfg.save_path)

def Save( ):
  print( 'Saving at iteration: ', it, flush=True )
  with open( cfg.cfg.save_path + '/iter.txt', 'w' ) as f:
    f.write(str(it))
  model.saver.save(sess, cfg.cfg.save_path+'/model.ckpt')

def Restore( ):
  global it
  global restored_it
  with open( cfg.cfg.save_path + '/iter.txt' ) as f:
    it=int(f.read())    
  restored_it = it
  print( 'Resuming from iteration: ', it, flush=True )
  model.saver.restore(sess, cfg.cfg.save_path+'/model.ckpt')

if FLAGS.resume:
  Restore( )

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
time.sleep(15)

####################################################################################
# main
print( 'Learning rate: ', current_learning_rate )
print( 'cfg.cfg.max_it: ', cfg.cfg.max_it )

while True:

    it += 1

    if it > 0 and it % 10000 == 0 and restored_it!=it and it < cfg.cfg.max_it:
      Save( )

    if 0<it and ( it % 2000 == 0 ):
      if cfg.cfg.scatter:
        scatterPlot(it, sess, model, imageSourceTest)
      
      for i in range(imageSourceCrossSample.count // 10):
        crossSampler2( i, 10, it, sess, model, imageSourceCrossSample )
      
        samples1d = sess.run(model.rec, feed_dict={model.z_sample: latents_for_crossSampling})
        plot(samples1d, cfg.cfg.sampleGridSize, cfg.cfg.outdirImages+'/{}d.png'.format(str(it).zfill(8)), cfg.cfg.shiftMean)

        X1, Y1 = sess.run( [model.X, model.rec], feed_dict={model.imageSource.batch_size: cfg.cfg.sampleGridSize*cfg.cfg.sampleGridSize } )
        plot(X1, cfg.cfg.sampleGridSize, cfg.cfg.outdirImages+'/{}x1.png'.format(str(it).zfill(8)), cfg.cfg.shiftMean)
        plot(Y1, cfg.cfg.sampleGridSize, cfg.cfg.outdirImages+'/{}y1.png'.format(str(it).zfill(8)), cfg.cfg.shiftMean)
        

    model.train_AE( current_learning_rate, it )

    if cfg.cfg.mutualInformation != cfg.MutualInformationType.NONE:
      for _ in range(cfg.cfg.D_inner_iters):
        model.train_D( current_learning_rate, it )

    if it % 100 == 0:
      model.print_stats( it )

    stdout.flush()
  
    if it >= cfg.cfg.max_it:
      break

Save()

coord.request_stop()
coord.join(threads)
