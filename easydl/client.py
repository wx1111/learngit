# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with gesture model.

Typical usage example:

    do_inference.py --base64_file=<file_path> --server=localhost:9000
"""

from __future__ import print_function

import time

# This is a placeholder for a Google-internal import.

import grpc
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.core.framework import tensor_shape_pb2, tensor_pb2, types_pb2

import base64
from io import BytesIO
from PIL import Image

tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('base64_file', '', 'base64 file path. ')
FLAGS = tf.app.flags.FLAGS

def load_image_and_preprocess(content):
  # load base64 file and convert to image
  byte_data = base64.b64decode(content)
  image_data = BytesIO(byte_data)
  image = Image.open(image_data)
  # resize
  #resized_image = image.resize((224,224))
  # to np
  #np_image = numpy.array(resized_image)
  # subtract mean
  #processed_image = np_image - mean
  #processed_image = reorder_image
  #print(processed_image)
  #return numpy.reshape(processed_image, (150528))
  return image

def do_inference(hostport, image_str):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    path: The full path of working directory for test data set.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  #t0 = time.clock()
  channel = grpc.insecure_channel(hostport)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'image_class'
  request.model_spec.signature_name = 'predict_images'
  #t1 = time.clock()
  image = load_image_and_preprocess(image_str)
  #t2 = time.clock()
  #request.inputs['input'].CopyFrom(
  #    tf.contrib.util.make_tensor_proto(image, shape=[1, 224, 224, 3]))
  # use this approach can speed up over 100 times
  dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1),
          tensor_shape_pb2.TensorShapeProto.Dim(size=-1),
          tensor_shape_pb2.TensorShapeProto.Dim(size=-1),
          tensor_shape_pb2.TensorShapeProto.Dim(size=3)]
  tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
  tensor_proto = tensor_pb2.TensorProto(
    dtype=types_pb2.DT_FLOAT,
    tensor_shape=tensor_shape_proto,
    float_val=image)
  request.inputs['images'].CopyFrom(tensor_proto)
  #t3 = time.clock()
  result = stub.Predict(request, 5.0)  # 5 seconds
  #t4 = time.clock()
  response = numpy.array(result.outputs['scores'].float_val)
  print(numpy.shape(response))
  #t5 = time.clock()
  prediction = numpy.argmax(response)
  #print((t1-t0),(t2-t1),(t3-t2),(t4-t3),(t5-t4))
  return prediction, response


def main(_):
  if not FLAGS.server:
    print('please specify server host:port')
    return
  if not FLAGS.base64_file:
    print('please specify base64 file path')
    return
  with open(FLAGS.base64_file) as image_file:
    pred, prob_list = do_inference(FLAGS.server, image_file.read())
  print('\nprediction: %d prob_list: %s' % (pred, prob_list))


if __name__ == '__main__':
  tf.app.run()
