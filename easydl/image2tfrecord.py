
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import threading

lock = threading.Lock()
import random

tf.app.flags.DEFINE_string('data_dir', '/mnt/shared/easydl/train_data_dir', '''''')
tf.app.flags.DEFINE_string('subset', 'train', '''''')

tf.app.flags.DEFINE_integer('num_shard', 10, '''''')
tf.app.flags.DEFINE_integer('num_thread', 10, '''''')

tf.app.flags.DEFINE_string('output_dir', '/mnt/shared/easydl/train_data_dir_tfreocrd', '''''')

FLAGS = tf.app.flags.FLAGS


def find_image_files(data_dir):
    labels = []
    #human = []
    filenames = []
    #    labels_dictionary = {}
    label_index = 0

    for i in tf.gfile.ListDirectory(data_dir):
        print(i)
        print(data_dir + i)
        filename_one_label = []
        try:
          for j in list(['*.png','*.jpg', '*.jpeg','*.PNG','*.JPG', '*.JPEG']):#'*.bmp',
            #print(j)
            filename_one_label.extend(tf.gfile.Glob(data_dir + i + '/' + j))
            #print(tf.gfile.Glob(data_dir + i + '/' + j))
        except BaseException as e:
            print(e)
            print(i, 'is not a directory')
            continue

        filenames.extend(filename_one_label)
        #human.extend([i] * len(filename_one_label))
        labels.extend([int(i)] * len(filename_one_label))

        #        labels.extend([label_index] * len(filename_one_label))
        label_index += 1
        
    shuffle_index = list(range(len(filenames)))
    random.shuffle(shuffle_index)

    filenames = [filenames[i] for i in shuffle_index]
    labels = [labels[i] for i in shuffle_index]
    #human = [human[i] for i in shuffle_index]
    return filenames, labels#, human  # , labels_dictionary


def process_image(filename):
    image_data = tf.gfile.GFile(filename,'rb').read()

    #image_data = tf.read_file(filename)
#    print(image_data)
    tf.reset_default_graph()  
    graph = tf.Graph()        
    with graph.as_default() as g:   
        with tf.Session(graph=g) as session:
            #            image = session.run(tf.image.decode_jpeg(image_data,channels=3))
            image = session.run(tf.image.decode_image(image_data, channels = 3))
            height = image.shape[0]
            width = image.shape[1]
            assert image.shape[2] == 3
            assert len(image.shape) == 3
            
    return image_data, height, width


def convert_to_example(filename, image_buffer, label, height, width):
    #colorspace = 'RGB'
    channels = 3
    #image_format = 'JPEG'

    from build_imagenet_data import _int64_feature, _bytes_feature
    # os.path.pardir(filename)
    #    print(os.path.abspath(filename))
    #    print(os.path.basename(os.path.dirname(filename)))
    #    print(os.path.dirname(filename))
#    print(os.path.basename(os.path.dirname(filename)) + '/' + os.path.basename(filename))
#    print(os.path.join(os.path.basename(os.path.dirname(filename)), os.path.basename(filename)))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        #'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        #'image/class/text': _bytes_feature(human),
        #'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.join(os.path.basename(os.path.dirname(filename)), os.path.basename(filename))),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


def process_image_files_batch(thread_index, ranges, name, filenames, labels, num_shards):
    global num_example, num_corrupted
    num_threads = len(ranges)
    num_shards_per_batch = int(num_shards / num_threads)
    # batch==thread???

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        print(output_filename)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            #human = humans[i]
            try:
                image_buffer, height, width = process_image(filename)
            except Exception as e:
                print(e)
                print(filename + ' is abandoned')
                lock.acquire()
                try:
                    num_example = num_example - 1
                    num_corrupted = num_corrupted + 1
                finally:
                    lock.release()
                continue
            example = convert_to_example(filename, image_buffer, label, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
        writer.close()


def process_image_files(filenames, labels):
    #assert len(filenames) == len(human)
    assert len(filenames) == len(labels)

    num_shards = FLAGS.num_shard
    num_thread = FLAGS.num_thread
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].

    spacing = np.linspace(0, len(filenames), num_thread + 1).astype(np.int)
    print(spacing)
    ranges = []

    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    print(ranges)
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, FLAGS.subset, filenames, labels, num_shards)
        t = threading.Thread(target=process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)


def main(unused_argv):

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    #    filenames, labels, human, labels_dictionary = find_image_files(FLAGS.data_dir)
    filenames, labels = find_image_files(FLAGS.data_dir)
    num_examples = {}

    global num_example, num_corrupted

    num_example = len(filenames)
    num_corrupted = 0

    num_examples['the total number of upload samples'] = num_example

    process_image_files(filenames, labels)

    num_examples['the total number of available samples'] = num_example
    num_examples['the number of corrupted samples'] = num_corrupted

    print('start saving the look_up dictionary and num_samples dictionary')
    '''
    import json
    labels_dictionary_path = FLAGS.output_dir + 'labels_dictionary.json'
    jsObj = json.dumps(labels_dictionary)
    fileObject = open(labels_dictionary_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()
    '''
    import json
    num_examples_path = FLAGS.output_dir + 'num_examples.json'
    jsObj = json.dumps(num_examples)
    fileObject = open(num_examples_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()

    print('done saving')

    print('----' * 15)
    print('finished')
    print('----' * 15)


if __name__ == '__main__':
    tf.app.run()






