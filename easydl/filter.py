import argparse
import os
import shutil
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--pb_input_dir', type=str, default = None)
parser.add_argument('--ck_input_dir', type=str, default = None)
parser.add_argument('--result_input_dir', type=str, default = None)
parser.add_argument('--deploy_dir', type=str, default = None)
parser.add_argument('--train_dir', type=str, default = None)
parser.add_argument('--data_output_dir', type=str, default = None)
#parser.add_argument('--look_up_dictionary_dir', type=str, default = None)

args = parser.parse_args()

pb_input_dir = args.pb_input_dir
ck_input_dir = args.ck_input_dir
result_input_dir = args.result_input_dir
pb_output_dir = args.deploy_dir
ck_output_dir = args.train_dir
data_output_dir = args.data_output_dir
'''
while os.listdir(pb_input_dir)[0].rsplit('/')[-1] != 'model_pb':
  pb_input_dir = pb_input_dir + os.listdir(pb_input_dir)[0]
  
#pb_input_dir = pb_input_dir + '/' + os.listdir(pb_input_dir)[0] + '/'
print(pb_input_dir)

while os.listdir(ck_input_dir)[0].rsplit('/')[-1] != 'checkpoint_dir':
  ck_input_dir = ck_input_dir + os.listdir(ck_input_dir)[0]
#ck_input_dir = ck_input_dir + os.listdir(ck_input_dir)[0] + '/'
print(ck_input_dir)

while os.listdir(result_input_dir)[0].rsplit('/')[-1] != 'output_dir':
  result_input_dir = result_input_dir + os.listdir(result_input_dir)[0]
#result_input_dir = result_input_dir + os.listdir(result_input_dir)[0] + '/'
print(result_input_dir)
'''

os.popen('cp -r ' + pb_input_dir + '/./. ' + pb_output_dir)
os.popen('cp -r ' + ck_input_dir + '/./. ' + ck_output_dir)
os.popen('rm ' + ck_output_dir + 'checkpoint')
os.popen('rm ' + ck_output_dir + 'events.out.tfevents.*')
os.popen('rm ' + ck_output_dir + 'graph.pbtxt')
os.popen('cp -r ' + result_input_dir + '/. ' + data_output_dir)

print(pb_input_dir)
print(ck_input_dir)
print(result_input_dir)
print(pb_output_dir)
print(ck_output_dir)
print(data_output_dir)

'''
print(model_input_dir)
print(result_input_dir)
print(os.listdir(model_input_dir))
print(os.listdir(result_input_dir))
for i in os.listdir(model_input_dir):
  old = model_input_dir + i
  new = model_output_dir
  print(old)
  print(new)
  shutil.copy(old,new)

for i in os.listdir(result_input_dir):
  old = result_input_dir + i
  new = data_output_dir
  print(old)
  print(new)
  shutil.copy(old,new)
'''
print('done')


