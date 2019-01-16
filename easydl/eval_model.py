import tensorflow as tf
import os

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.python.framework import function
from tensorflow.python.platform import gfile
import numpy 

tf.app.flags.DEFINE_string('checkpoint_dir', '/weixue/my_bench/imagenet/resnet_v1_50.ckpt', '')
tf.app.flags.DEFINE_string('data_dir', '/weixue/imagenet-tf/', '')
tf.app.flags.DEFINE_string('output_dir', '', '')

tf.app.flags.DEFINE_string('model_name', 'inception_v4', '')
tf.app.flags.DEFINE_integer('num_class', 10, 'the number of training sample categories')
tf.app.flags.DEFINE_integer('batch_size', 8 , '')
   
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim

batch_size = FLAGS.batch_size
import Configure_file
height = Configure_file.configure_image_size(FLAGS.model_name)
width = height

import json
num_examples_path = FLAGS.data_dir + 'num_examples.json'
with open(num_examples_path) as load_f:
  load_dict = json.load(load_f)
  num_examples = load_dict['the total number of available samples']

num_classes = FLAGS.num_class
eval_steps = int(num_examples / FLAGS.batch_size)

if eval_steps == 0:
  eval_steps = int(1)
  batch_size = num_examples
elif (num_examples % FLAGS.batch_size) > 0:
  eval_steps = eval_steps + 1
 
print('number of validation samples %d' % num_examples)
print('eval steps : %d' % eval_steps)
print('batch size %d' % batch_size)
  
def preprocess_fn(value, batch_position, is_training):

    from pre import parse_example_proto
    image_buffer, label,  filename = parse_example_proto(value)
    
    from pre import preprocess
    images = preprocess(image_buffer,is_training, FLAGS.model_name, height, width)

    return (images, label, filename)

def validation_data_generator():

  with tf.name_scope('validation_batch_processing'):

    data_dir = FLAGS.data_dir
    glob_pattern = os.path.join(data_dir, 'validation-*-of-*')
    file_names = gfile.Glob(glob_pattern)
    ds = tf.data.TFRecordDataset.list_files(file_names)

    ds = ds.apply(interleave_ops.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))
    counter = tf.data.Dataset.range(batch_size)
    counter = counter.repeat()
    flags = tf.data.Dataset.from_tensors(tf.constant('validation'))
    flags = flags.repeat()
    ds = tf.data.Dataset.zip((ds, counter, flags))
    ds = ds.prefetch(buffer_size=batch_size*4)
    ds = ds.repeat()
    ds = ds.apply(batching.map_and_batch(map_func=preprocess_fn,batch_size=batch_size,num_parallel_batches=10))
    ds = ds.prefetch(buffer_size=10)
    from tensorflow.contrib.data.python.ops import threadpool
    ds = threadpool.override_threadpool(ds,threadpool.PrivateThreadPool(10,
                                                                        display_name='input_pipeline_thread_pool'))

    return ds

def main(argv=None):
  
  validation_dataset = validation_data_generator()
  iterator = tf.data.Iterator.from_structure(output_types=validation_dataset.output_types,
                                             output_shapes=validation_dataset.output_shapes)

  validation_init_op = iterator.make_initializer(validation_dataset)
  images, labels, filenames = iterator.get_next()
  images = tf.reshape(images, shape=[batch_size, height, width, 3])
  #images = tf.Print(images,[tf.convert_to_tensor('jianjunzhi,hou'),images],first_n=1,summarize=100)
  labels = tf.reshape(labels, [batch_size])
  
  from nets import nets_factory
  network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes = num_classes,
    is_training = False)
    
  logits, end_points = network_fn(images)
  pred_soft = tf.nn.softmax(logits)

  with tf.control_dependencies([pred_soft]):#, variables_averages_op,batch_norm_updates_op]):
    eval_op = tf.no_op(name='eval_op')

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(validation_init_op)
  g_list = tf.global_variables()
  bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
  bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
  #print(bn_moving_vars)
  store_restore_var_list = tf.trainable_variables() + bn_moving_vars
  saver = tf.train.Saver(store_restore_var_list)

  ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

  if ckpt:
    saver.restore(sess, ckpt)
    print("restore from the checkpoint {0}".format(ckpt))
  
  print('validation strarts')
  
  log, lab, f = [],[],[]
  import time
  t0 = time.time()
  for i in range(eval_steps):
    _, logit, label_batch,fis = sess.run([eval_op, pred_soft,labels,filenames])
    log.extend(logit.tolist())
    lab.extend(label_batch)
    f.extend(fis)
    
  print('time : %f' %(time.time()-t0))
  
  log = log[:num_examples]
  lab = lab[:num_examples]
  f = f[:num_examples]

  result = {}
  result['files'] = f
  result['labels'] = str(lab)
  result['logits'] = log
  
  result_path = FLAGS.output_dir + 'result.json'
  import json
  jsObj = json.dumps(result)
  fileObject = open(result_path, 'w')
  fileObject.write(jsObj)
  fileObject.close()
  
  print('finished')
  
  dic_result = {}
  for i in range(len(f)):
    print(f[i] + '\t\t' + str(log[i]) + '\t\t' + str(lab[i]))
    #dic_result[f[i]] = [log[i],lab[i]]
  #print(dic_result)
  

if __name__ == '__main__':
  tf.app.run()
