import tensorflow as tf
# the same as neuhub
import os
import numpy
from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.python.framework import function
from tensorflow.python.platform import gfile
import Configure_file

tf.app.flags.DEFINE_string('pre_trained_model_ckpt_path', '', '')
tf.app.flags.DEFINE_string('checkpoint_dir', '/weixue/my_bench/imagenet/checkpoint/', '')
tf.app.flags.DEFINE_string('summary_dir', '/weixue/my_bench/imagenet/checkpoint/', '')
tf.app.flags.DEFINE_string('train_data_dir', '/weixue/imagenet-tf/', '')
tf.app.flags.DEFINE_string('validation_data_dir', '/weixue/imagenet-tf/', '')
tf.app.flags.DEFINE_string('export_path_base', '', 'export file')

tf.app.flags.DEFINE_integer('batch_size', 16, '')
#tf.app.flags.DEFINE_integer('input_height', 224, '')
#tf.app.flags.DEFINE_integer('input_width', 224, '')

#tf.app.flags.DEFINE_integer('num_examples', 7000, 'the number of training samples')

tf.app.flags.DEFINE_integer('num_class', 37, 'the number of training sample categories')
tf.app.flags.DEFINE_float('epochs', 10 , '')
tf.app.flags.DEFINE_integer('warmup_epochs', 0, '')

tf.app.flags.DEFINE_float('init_learning_rate', 0.0001, '')
tf.app.flags.DEFINE_float('decay_rate', 0.9, '')
tf.app.flags.DEFINE_float('decay_steps', 100, '')
tf.app.flags.DEFINE_float('warm_lr', 0.1, '')

tf.app.flags.DEFINE_string('lr_decay', 'exponential_decay', 'exponential_decay, natural_exp_decay,polynomial_decay')
tf.app.flags.DEFINE_string('optimizer', 'rmsp', 'rmsp,adam,sgd,mometum,lars')
tf.app.flags.DEFINE_string('model_name', 'vgg_16','')

tf.app.flags.DEFINE_integer('display_every_steps', 10, '')
tf.app.flags.DEFINE_integer('eval_every_steps', 100, '')
tf.app.flags.DEFINE_integer('fine_tune', 1, 'whether the model is trained from a pre-trained model')
tf.app.flags.DEFINE_integer('early_stop', 1, 'whether to stop training model early')

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim

batch_size = FLAGS.batch_size
import Configure_file
height = Configure_file.configure_image_size(FLAGS.model_name)
width = height
display_every_steps = FLAGS.display_every_steps

def preprocess_fn(value, batch_position, is_training):

    from pre import parse_example_proto
    image_buffer, label, label_text, filename = parse_example_proto(value)
    
    from pre import preprocess
    images = preprocess(image_buffer,is_training, FLAGS.model_name, height, width)

    return (images, label, label_text, filename)

def train_data_generator():
    with tf.name_scope('train_batch_processing'):
        data_dir = FLAGS.train_data_dir
        glob_pattern = os.path.join(data_dir, 'train-*-of-*')
        file_names = gfile.Glob(glob_pattern)
        import random
        random.shuffle(file_names)
        ds = tf.data.TFRecordDataset.list_files(file_names)

        ds = ds.apply(interleave_ops.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))
        counter = tf.data.Dataset.range(batch_size)
        counter = counter.repeat()
        flags = tf.data.Dataset.from_tensors(tf.constant('train'))
        flags = flags.repeat()
        ds = tf.data.Dataset.zip((ds, counter, flags))
        ds = ds.prefetch(buffer_size=batch_size * 4)
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.repeat()
        ds = ds.apply(batching.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_batches=10))
        ds = ds.prefetch(buffer_size=10)
        from tensorflow.contrib.data.python.ops import threadpool
        ds = threadpool.override_threadpool(ds, threadpool.PrivateThreadPool(10,
                                                                             display_name='input_pipeline_thread_pool'))
        #    ds_iterator = ds.make_initializable_iterator()
        return ds


def validation_data_generator():
    with tf.name_scope('validation_batch_processing'):
        data_dir = FLAGS.validation_data_dir
        glob_pattern = os.path.join(data_dir, 'validation-*-of-*')
        file_names = gfile.Glob(glob_pattern)
        ds = tf.data.TFRecordDataset.list_files(file_names)

        ds = ds.apply(interleave_ops.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))
        counter = tf.data.Dataset.range(batch_size)
        counter = counter.repeat()
        flags = tf.data.Dataset.from_tensors(tf.constant('validation'))
        flags = flags.repeat()
        ds = tf.data.Dataset.zip((ds, counter, flags))
        ds = ds.prefetch(buffer_size=batch_size * 4)
        ds = ds.apply(batching.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_batches=10))
        ds = ds.prefetch(buffer_size=10)
        from tensorflow.contrib.data.python.ops import threadpool
        ds = threadpool.override_threadpool(ds, threadpool.PrivateThreadPool(10,
                                                                             display_name='input_pipeline_thread_pool'))
        #    ds_iterator = ds.make_initializable_iterator()
        return ds


def main(argv=None):
	
    import json
    train_num_examples_path = FLAGS.train_data_dir + 'num_examples.json'
    validation_num_examples_path = FLAGS.validation_data_dir + 'num_examples.json'
    with open(train_num_examples_path) as load_f:
      load_dict = json.load(load_f)     
      train_num_examples = load_dict['the total number of available samples']
    with open(validation_num_examples_path) as load_f:
      load_dict = json.load(load_f)     
      validation_num_examples = load_dict['the total number of available samples']
      
    num_classes = FLAGS.num_class
    eval_every_steps = FLAGS.eval_every_steps
    train_steps = int(FLAGS.epochs * train_num_examples) / (FLAGS.batch_size)
    eval_steps = int(validation_num_examples / (FLAGS.batch_size))
    warm_steps = int(FLAGS.warmup_epochs * train_num_examples / (FLAGS.batch_size))
    fine_tune_flag = 'True' if FLAGS.fine_tune == 1 else 'False'
    early_stop_flag = 'True' if FLAGS.early_stop == 1 else 'False'
    print('---' * 20)
    print('model for classification : %s' % FLAGS.model_name)
    print('input height and width : %d' % height)
    print('whether to fine tune : %s' % fine_tune_flag)
    print('whether to early stop : %s' % early_stop_flag)
    print('number of train samples : %d' % train_num_examples)
    print('number of train classes : %d' % FLAGS.num_class)
    print('eval every steps training %d' % FLAGS.eval_every_steps)
    print('train steps : %d' % train_steps)
    print('eval steps : %d' % eval_steps)
    print('batch size : %d' % batch_size)
    print('optimizer : %s' % FLAGS.optimizer)
    print('init_learning rate :%f' % FLAGS.init_learning_rate)
    print('lr deay policy :%s' % FLAGS.lr_decay)
    print('warm epochs : %d' % FLAGS.warmup_epochs)
    print('warm learning rate :%f' % FLAGS.warm_lr)
    print('---' * 20)

    global_steps = tf.train.get_or_create_global_step()

    lr = Configure_file.configure_lr(FLAGS.init_learning_rate, FLAGS.lr_decay, FLAGS.decay_steps,
                      FLAGS.decay_rate, global_steps, FLAGS.warm_lr, warm_steps)
    # (init_lr,decay_policy,decay_steps,decay_rate,global_steps,warm_lr=0.0001,warm_steps=0)
#    tf.summary.scalar('learning_rate', lr)
    opt = Configure_file.configure_optimizer(FLAGS.optimizer, lr)
    # (optimizer,learning_rate)

    train_dataset = train_data_generator()
    validation_dataset = validation_data_generator()
    iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,output_shapes = train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    images, labels, label_texts, filenames = iterator.get_next()
    
#    images = tf.Print(images,[filenames])
    images = tf.reshape(images, shape=[-1, height, width, 3])
    labels = tf.reshape(labels, [batch_size])

    from nets import nets_factory
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes = num_classes,
        weight_decay = 0.0004,
        is_training = True)
    logits, end_points = network_fn(images)
    pred_soft = tf.nn.softmax(logits)
    values, indices = tf.nn.top_k(pred_soft, 1)    
    if 'AuxLogits' in end_points:
        aux_cross_entropy = 0.4 * tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['AuxLogits'], labels=labels,name='aux_cross-entropy'))
    else:
        aux_cross_entropy = 0
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy'))
    loss = cross_entropy + aux_cross_entropy
    with tf.name_scope('accuracy'):
        top_1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(pred_soft, labels, 1), dtype=tf.float32), name='top_1')
        top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(pred_soft, labels, 5), dtype=tf.float32), name='top_5')
    '''
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('aux_cross_entropy', aux_cross_entropy)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('top1', top_1)
    tf.summary.scalar('top5', top_5)

    for i in tf.global_variables():
        tf.summary.histogram(i.name.replace(":", "_"), i)
    '''
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.minimize(loss, global_step=global_steps)

    #  recommended
    with tf.control_dependencies([train_step, loss, top_1, top_5]):  # , variables_averages_op,batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    with tf.control_dependencies([loss, top_1, top_5]):  # , variables_averages_op,batch_norm_updates_op]):
        validation_op = tf.no_op(name='validation_op')

    config = tf.ConfigProto()
    config.allow_soft_placement = True

    checkpoint_dir = FLAGS.checkpoint_dir
    summary_dir = FLAGS.summary_dir
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=1)
    
    checkpoint_basename = FLAGS.model_name + '.ckpt'
    hooks = [
#        tf.train.LoggingTensorHook(
#            tensors={'step': global_steps, 'loss': loss, 'lr': lr, 'top1': top_1, 'top5': top_5},
#            every_n_iter=FLAGS.display_every_steps),
#        tf.train.StopAtStepHook(num_steps=None, last_step=max_steps),
        tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=100, saver=saver,
                                 checkpoint_basename=checkpoint_basename),
#        tf.train.SummarySaverHook(save_steps=100, save_secs=None, output_dir=summary_dir, summary_writer=None,
#                              scaffold=None, summary_op=summary_op)
    ]

    if FLAGS.fine_tune == 1:
        exclusions = Configure_file.model_exclusions(FLAGS.model_name)
        print(exclusions)
        variables_to_restore = []
        for var in slim.get_model_variables():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
              variables_to_restore.append(var)

        pre_train_saver = tf.train.Saver(variables_to_restore)

        if tf.gfile.IsDirectory(FLAGS.pre_trained_model_ckpt_path):
#          print(tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*'))
          if tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt'):
            print('there is one ckpt file')
            ckpt_path = tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt')[0]
      	  elif tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt.*'):
            print('there is more than one ckpt files')
            ckpts = tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt.*')[0]
            ckpt_path = ckpts.rsplit('.',1)[0]
          # imagenet pretrained model
          elif tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt-*'):
            print('there is more than one ckpt files')
            ckpts = tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*.ckpt-*')[0]
            ckpt_path = ckpts.rsplit('.',1)[0]
          # pipline pretrained model
        else:
          ckpt_path = FLAGS.pre_trained_model_ckpt_path
          
#        ckpt_path = tf.gfile.Glob(FLAGS.pre_trained_model_ckpt_path + FLAGS.model_name + '*')
        
        print(ckpt_path)
        def load_pretrain(scaffold, sess):
            pre_train_saver.restore(sess, ckpt_path)
        scaffold = tf.train.Scaffold(init_fn=load_pretrain, summary_op=tf.summary.merge_all())
    else:
        scaffold = None
      
    print('start training')
    early_stop_param = {}
    early_stop_param['count'] = 0
    early_stop_param['last'] = 100000
    
    with tf.train.MonitoredTrainingSession(checkpoint_dir=None,
                                           config=config,
                                           scaffold=scaffold,
                                           hooks=hooks
                                           ) as mon_sess:
      if FLAGS.early_stop == 1:
        print('early stop')
        mon_sess._coordinated_creator.tf_sess.run(train_init_op)
        
        import early_stop
        global_step = 0
        while global_step < train_steps:
          if (global_step) % eval_every_steps == 0 and global_step > 0:
            print('start validating')
            mon_sess._coordinated_creator.tf_sess.run(validation_init_op)
            loss_list = []
            tt1 = []
            for i in range(eval_steps) :
              _, batch_loss, t1 = mon_sess.run([validation_op, loss, top_1])
              loss_list.append(batch_loss)
              tt1.append(t1)
            validation_loss = numpy.mean(numpy.asarray(loss_list))
            validation_top1 = numpy.mean(numpy.asarray(tt1))
            mon_sess._coordinated_creator.tf_sess.run(train_init_op)
            print('done validating, validation loss is %f , top1 is %f' % (validation_loss,validation_top1))
            global_step = global_step + 1
            early_stop_param = early_stop.early_stop(validation_loss,early_stop_param)
            if early_stop_param['count'] >= 3:
              print('process should stop')
              break
          if (global_step+1) % display_every_steps == 0 and global_step > 0:
            global_step, _, batch_loss, top1, top5 = mon_sess.run([global_steps, train_op, loss, top_1, top_5])
            print('global_step: %d, train_loss: %f, top1: %f, top5: %f' % (global_step, batch_loss, top1, top5))
          else: 
            global_step, _ = mon_sess.run([global_steps, train_op])
      else:
        print('no early stop')
        mon_sess._coordinated_creator.tf_sess.run(train_init_op)
        global_step = 0
        while global_step < train_steps:
          if (global_step+1) % display_every_steps == 0 and global_step > 0:
            global_step, _, batch_loss, top1, top5 = mon_sess.run([global_steps, train_op, loss, top_1, top_5])
            print('global_step: %d, train_loss: %f, top1: %f, top5: %f' % (global_step, batch_loss, top1, top5))
          else: 
            global_step, _ = mon_sess.run([global_steps, train_op])
          
	# Export model
	# WARNING(break-tutorial-inline-code): The following code snippet is
	# in-lined in tutorials, please update tutorial documents accordingly
	# whenever code changes.
#    mon_sess.graph._unsafe_unfinalize()
    export_path_base = FLAGS.export_path_base
    export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes(str('image_class'))
    )
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # Build the signature_def_map.

    classification_inputs = tf.saved_model.utils.build_tensor_info(images)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(indices)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

    classification_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
          tf.saved_model.signature_constants.CLASSIFY_INPUTS:classification_inputs,
        },
        outputs={
          tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes,
          tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:classification_outputs_scores
        },
        method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
    
    prediction_input = tf.saved_model.utils.build_tensor_info(images)
    prediction_output = tf.saved_model.utils.build_tensor_info(pred_soft)
    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': prediction_input},
        outputs={'scores': prediction_output},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, latest_checkpoint)
      legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
      builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:classification_signature,
          'predict_images':prediction_signature
        },
      strip_default_attrs=True
      )
      builder.save()
      print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()
