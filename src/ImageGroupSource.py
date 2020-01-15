import tensorflow as tf
import numpy as np
import cfg
from utils import *

class ImageGroupSource:
  
  def read_labeled_image_list(base_image_dir, image_list_file):
    f = open(image_list_file, 'r')
    
    r = []
    for i in range(cfg.cfg.groupSize+1):
      r.append([])
    
    for line in f:
        
      tokens = line[:-1].split(' ')
      
      for i in range(cfg.cfg.groupSize):
        
        filename = base_image_dir +  '/' + tokens[i]
        r[i].append( filename )

      r[cfg.cfg.groupSize].append(int(tokens[cfg.cfg.groupSize])) #label
      
    print( 'train file loaded', flush=True )
    return r

  def read_images_from_disk(input_queue):
    r = []
    
    for i in range(cfg.cfg.groupSize):
      file_contents = tf.read_file(input_queue[i])
      image = tf.image.decode_png(file_contents, channels=cfg.cfg.color_channels)

      if cfg.cfg.binary:
        image = tf.cast( image>127, image.dtype )*255

      r.append( image )

    r.append( input_queue[cfg.cfg.groupSize] )

    
    print( 'read_images_from_disk built', flush=True)
    return r

  def get_image_batches( base_image_dir, batch_size, trainFilename ):
    
    lists = ImageGroupSource.read_labeled_image_list(base_image_dir, trainFilename)
    
    image_lists = lists[0:cfg.cfg.groupSize]
    label_list = lists[cfg.cfg.groupSize]
    
    
    tensorLists = []
    for i in range(cfg.cfg.groupSize):
      tensorLists.append( tf.convert_to_tensor(image_lists[i], dtype=tf.string) )
    tensorLists.append( tf.convert_to_tensor(label_list, dtype=tf.int32) )
    input_queue = tf.train.slice_input_producer(tensorLists, shuffle=True, capacity=32*10)
    

    lists = ImageGroupSource.read_images_from_disk(input_queue)

    expected_shape = [cfg.cfg.target_size, cfg.cfg.target_size, cfg.cfg.color_channels]

    r = []
    for i in range(cfg.cfg.groupSize):
      image1 = tf.image.resize_image_with_crop_or_pad(lists[i], cfg.cfg.target_size, cfg.cfg.target_size )

      scale = 255.0 if not cfg.cfg.shiftMean else 127.4
      shift = 0.0 if not cfg.cfg.shiftMean else 127.5
      image1 = tf.divide( tf.subtract( tf.to_float(image1), tf.constant(shift) ), tf.constant(scale) )
    
      image_shape1 = tf.shape(image1)
    
      assert_op1 = tf.assert_equal(image_shape1, expected_shape)
    
      with tf.control_dependencies([assert_op1]):
        image1.set_shape( expected_shape )
        image1 = tf.transpose(image1, [2, 0, 1])
        
        r.append(image1)
    
    label = lists[cfg.cfg.groupSize]
    r.append( label )
      
      
    return tf.train.batch(r, batch_size=batch_size, capacity=32*10, num_threads=4)

  def __init__( self, sess, path, trainFile ):
    self.batch_size = tf.placeholder(tf.int32)
    print( 'trainFile: ', trainFile, flush=True )
    
    batches = ImageGroupSource.get_image_batches( path, self.batch_size//cfg.cfg.groupSize, trainFile )
    
    self.image_batch = batches[0]
    self.label_batch = batches[cfg.cfg.groupSize]
    
    for i in range(1, cfg.cfg.groupSize):
      self.image_batch = tf.concat( [self.image_batch, batches[i]], axis=0 )    
      self.label_batch = tf.concat( [self.label_batch, batches[cfg.cfg.groupSize]], axis=0 )
    
    initialize_uninitialized( sess )
