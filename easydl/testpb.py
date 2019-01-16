import tensorflow as tf
import os

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.python.framework import function
from tensorflow.python.platform import gfile
import numpy 
from preprocessing import preprocessing_factory

tf.app.flags.DEFINE_string('checkpoint_dir', '/weixue/my_bench/imagenet/resnet_v1_50.ckpt', '')
tf.app.flags.DEFINE_string('model_name', 'vgg_16', '')
tf.app.flags.DEFINE_string('export_path_base', '', '')

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
  image_place = tf.placeholder(dtype=tf.float32,shape=[None,None,None,3],name='image_place')
  image = tf.identity(image_place,name='image')
  images = tf.reshape(image, shape=[-1, height, width, 3])
  '''
  image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.model_name,is_training=False) 
  def preprocessing_fn(one_image):
    one_image = tf.stack([one_image[:,:,2],one_image[:,:,1],one_image[:,:,0]],axis=-1)# bgr->rgb
    #print(one_image)
    return image_preprocessing_fn(one_image, height, width)
  #images = image_preprocessing_fn(image, height, width)
  images_batch = tf.map_fn(preprocessing_fn,image,dtype=tf.float32,swap_memory=True)
  #from preprocessing import vgg_preprocessing
  #images = vgg_preprocessing.preprocess_for_eval(image, height, width,256)
  images = tf.reshape(images_batch, shape=[-1, height, width, 3])
  '''
  from nets import nets_factory
  network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes = FLAGS.num_classes,
    is_training = False)

  logits, end_points = network_fn(images)
  pred_soft = tf.nn.softmax(logits)

  g_list = tf.global_variables()
  bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
  bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
  
  store_restore_var_list = tf.trainable_variables() + bn_moving_vars
  latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  saver = tf.train.Saver(store_restore_var_list)
  saver.restore(sess, latest_checkpoint)
  
  export_path_base = FLAGS.export_path_base
  export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str('image_class'))
  )
  print('Exporting trained model to', export_path)
  
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  # Build the signature_def_map.
  prediction_input = tf.saved_model.utils.build_tensor_info(image)
  prediction_output = tf.saved_model.utils.build_tensor_info(pred_soft)
  prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'input': prediction_input},
      outputs={'output': prediction_output},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
  '''
  config = tf.ConfigProto()
  jit_level = tf.OptimizerOptions.ON_1
  config.graph_options.optimizer_options.global_jit_level = jit_level
  with tf.Session(config=config) as sess:
  '''
  builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
      'serving_default':prediction_signature
    },
    strip_default_attrs=True
  )
  builder.save()
  print('Done exporting!')
  
if __name__ == '__main__':
  tf.app.run()
