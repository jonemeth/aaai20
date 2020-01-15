import tensorflow as tf
import numpy as np
import cfg
from utils import *

class ImageSource:
  
  def read_labeled_image_list(self, base_image_dir, image_list_file):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    count = 0
    
    for line in f:
      filename, label = line[:-1].split(' ')
      filename = base_image_dir +  '/' + filename
      filenames.append(filename)
      labels.append(int(label))
      count += 1
      
    print( 'train file loaded', flush=True )
    
    self.count = count
    
    return filenames, labels

  def read_images_from_disk(input_queue):
    file_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    example = tf.image.decode_png(file_contents, channels=cfg.cfg.color_channels)
    
    if cfg.cfg.binary:
      example = tf.cast( example>127, example.dtype )*255
    
    print( 'read_images_from_disk built', flush=True)
    return example, label

  def get_image_batches( self, base_image_dir, batch_size, trainFilename, shuffle ):
    image_list, label_list = self.read_labeled_image_list(base_image_dir, trainFilename)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

    input_queue = tf.train.slice_input_producer([images, labels], shuffle=shuffle)

    image, label = ImageSource.read_images_from_disk(input_queue)

    image = tf.image.resize_image_with_crop_or_pad(image, cfg.cfg.target_size, cfg.cfg.target_size )

    scale = 255.0 if not cfg.cfg.shiftMean else 127.4
    shift = 0.0 if not cfg.cfg.shiftMean else 127.5
    image = tf.divide( tf.subtract( tf.to_float(image), tf.constant(shift) ), tf.constant(scale) )

    image_shape = tf.shape(image)
    expected_shape = [cfg.cfg.target_size, cfg.cfg.target_size, cfg.cfg.color_channels]
    
    assert_op = tf.assert_equal(image_shape, expected_shape)
    
    with tf.control_dependencies([assert_op]):
      image.set_shape( expected_shape )
      image = tf.transpose(image, [2, 0, 1])
    
    return tf.train.batch([image, label], batch_size=batch_size)

  def __init__( self, sess, path, trainFile, shuffle=True ):
    self.batch_size = tf.placeholder(tf.int32)
    print( 'trainFile: ', trainFile, flush=True )
    self.image_batch, self.label_batch = self.get_image_batches( path, self.batch_size, trainFile, shuffle )
    initialize_uninitialized( sess )
