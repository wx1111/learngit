import tensorflow as tf
import os

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.python.framework import function
from tensorflow.python.platform import gfile
import numpy 
from preprocessing import preprocessing_factory
import matplotlib.pylab as plt
import time
import cv2

tf.app.flags.DEFINE_string('checkpoint_dir', 'imagenet/resnet_v1_50.ckpt', '')
tf.app.flags.DEFINE_string('image_dir', 'imagenet/resnet_v1_50.ckpt', '')
tf.app.flags.DEFINE_string('model_name', 'inception_v4', '')

tf.app.flags.DEFINE_integer('num_classes', 10, 'the number of training sample categories')

FLAGS = tf.app.flags.FLAGS 

tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim

import Configure_file
height = Configure_file.configure_image_size(FLAGS.model_name)
width = height

def main(argv=None):
  
  #image_file = tf.placeholder(dtype=tf.string,name='image_file')
  #image = tf.gfile.GFile(image_file,'rb').read()
  #image = tf.placeholder(dtype=tf.string,name='image')
  #image = tf.image.decode_image(image,channels=3,dtype=tf.float32)
  sess = tf.InteractiveSession()
  image = tf.placeholder(dtype=tf.float32,shape=[None,None,None,3],name='image')
  #image = tf.reshape(image, shape=[-1, height, width, 3])
  
  image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.model_name,is_training=False) 
  def preprocessing_fn(one_image):
    #one_image = tf.stack([one_image[:,:,2],one_image[:,:,1],one_image[:,:,0]],axis=-1)# bgr->rgb
    #print(one_image)
    return image_preprocessing_fn(one_image, height, width)
  #images = image_preprocessing_fn(image, height, width)
  images_batch = tf.map_fn(preprocessing_fn,image,dtype=tf.float32,swap_memory=True)
  #from preprocessing import vgg_preprocessing
  #images = vgg_preprocessing.preprocess_for_eval(image, height, width,256)
  images = tf.reshape(images_batch, shape=[-1, height, width, 3])
  
  from nets import nets_factory
  network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes = FLAGS.num_classes,
    is_training = False)

  logits, end_points = network_fn(images)
  pred_soft = tf.nn.softmax(logits)

  latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

  g_list = tf.global_variables()
  bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
  bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
  #print(bn_moving_vars)
  store_restore_var_list = tf.trainable_variables() + bn_moving_vars
  saver = tf.train.Saver(store_restore_var_list)
  saver.restore(sess, latest_checkpoint)
  
  config = tf.ConfigProto()
  jit_level = tf.OptimizerOptions.ON_1
  config.graph_options.optimizer_options.global_jit_level = jit_level
  
  t0 = time.time()
  #image_file = plt.imread(FLAGS.image_dir + '18437870.jpg')
  image_file = cv2.imread(FLAGS.image_dir + '18455308.jpg')
  image_file = cv2.cvtColor(image_file,cv2.COLOR_BGR2RGB)
  print(image_file)
  h,w,d = image_file.shape
  print(image_file.shape) #(1, 816, 612, 3)
  image_file = numpy.reshape(image_file,(1,h,w,d))
  print(image_file.shape)
  print(type(image_file))
  #print(image_file)
  image_file = image_file.astype(numpy.float32)
  #print(image_file)
  '''
  print(numpy.shape(image))
  print(h,w)
  image_file = numpy.resize(image,[1,h,w,3])
  '''
  t1 = time.time()
  p = sess.run(pred_soft,feed_dict={image: image_file})
  print(p)
  print(time.time()-t1)
  print(t1-t0)
  '''
  #legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  builder.add_meta_graph_and_variables(
  	sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
    	'serving_default':prediction_signature
        },
        strip_default_attrs=True
  )
  builder.save()
  print('Done exporting!')
  '''
  
if __name__ == '__main__':
  tf.app.run()
