import sklearn
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import json
import argparse
import numpy

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--result_dir', type=str, default = None)
parser.add_argument('--output_dir', type=str, default = None)
#parser.add_argument('--look_up_dictionary_dir', type=str, default = None)

args = parser.parse_args()
result_dir = args.result_dir
output_dir = args.output_dir
#look_up_dictionary_dir = args.look_up_dictionary_dir
'''
labels_dictionary_path = look_up_dictionary_dir + 'labels_dictionary.json'
with open(labels_dictionary_path) as load_f:
  labels_dictionary = json.load(load_f)

labels_dictionary_list =[]
for i in labels_dictionary:
  labels_dictionary_list.append(labels_dictionary[i])
print(labels_dictionary_list)
'''
result_path = result_dir + 'result.json'
with open(result_path) as load_f:
  result = json.load(load_f)

files = result['files']

labels = list(str(result['labels'][1:-1]).split(', '))
l = []
for i in labels:
  try:
    l.append(int(i))
  except:
    continue
labels = l

logits = result['logits']
preds = [i.index(max(i)) for i in logits]

num_class = len(logits[0])
enc = OneHotEncoder(dtype=numpy.int)
X = []
for i in range(num_class):
  X.append([i])

labelsll = []
for i in labels:
  labelsll.append([i])
enc.fit(X)
labels_one_hot = enc.transform(labelsll).toarray()
#labels_one_hot = label_binarize(labels, classes=range(num_class))
#print(labels)
#print(labels_one_hot)

classif_per = {}
accuracy_score = metrics.accuracy_score(labels,preds)
classif_per['accuracy'] = accuracy_score
print(accuracy_score)

precision_score = metrics.precision_score(labels,preds,average='macro')
classif_per['precision'] = precision_score
print(precision_score)

f1_score = metrics.f1_score(labels,preds,average='macro')
classif_per['f1_score'] = f1_score
print(f1_score)

recall_score = metrics.recall_score(labels,preds,average='macro')
classif_per['recall_score'] = recall_score
print(recall_score)
import numpy
'''
roc_auc_score_list = []

for i in range(num_class):
  ll = numpy.asarray(labels) == i
  log = []
  for j in logits:
    log.append(j[i])
#  print(log)
  roc_auc_score_list.append(metrics.roc_auc_score(ll,log, average='macro'))
roc_auc_score_macro=numpy.mean(roc_auc_score_list)
'''
roc_auc_score_micro = metrics.roc_auc_score(labels_one_hot, logits, average='micro')
'''
try :
  roc_auc_score_macro = metrics.roc_auc_score(labels_one_hot, logits, average='macro')
except:
  roc_auc_score_macro = numpy.nan

classif_per['auc_macro'] = roc_auc_score_macro
'''
classif_per['auc_micro'] = roc_auc_score_micro
print(roc_auc_score_micro)
classification_report_dict = {}
classification_report = metrics.classification_report(labels,preds,labels=range(num_class))
classification_report = str(classification_report).split('\n')

for i in range(len(classification_report)):
  x = classification_report[i]
  x = str(x).split(' ')
  xx =[]
  for j in x:
    try:
      assert len(j)>0
      xx.append(j)
    except:
      continue
#  print(len(xx))
  if len(xx) == 4:
    classification_report_dict['evaluation_index'] = xx
  elif len(xx) == 7:
    classification_report_dict['avg_all'] = xx[3:]
  elif len(xx)>0:
    classification_report_dict[xx[0]]=xx[1:]
print(classification_report_dict)

classif_per['classification_report'] = classification_report_dict
#print('classification_report' + str(type(classification_report)))
confusion_matrix = metrics.confusion_matrix(labels,preds)
confusion_matrix_str = ''
#print(type(confusion_matrix)) <type 'numpy.ndarray'>
for i in confusion_matrix:
  #print(type(i)) <type 'numpy.ndarray'>
#  confusion_matrix_.append(list(i))
#  print(numpy.shape(i))
  for j in i:
    confusion_matrix_str = confusion_matrix_str + str(j) + '\t'
  confusion_matrix_str = confusion_matrix_str + '\n'
print(confusion_matrix_str)
'''
confusion_matrix_dic = {}
confusion_matrix_dic['confusion_matrix'] = confusion_matrix_
print(confusion_matrix_dic)
'''
classif_per_path = output_dir + 'result.json'
import json
jsObj = json.dumps(classif_per)
fileObject = open(classif_per_path, 'w')
fileObject.write(jsObj)
fileObject.close()

confusion_matrix_path = output_dir + 'confusion_matrix.txt'
confusion_matrix_file = open(confusion_matrix_path,'w')
confusion_matrix_file.write(confusion_matrix_str)
confusion_matrix_file.close()

correct_path =  output_dir + 'correct.txt'
error_path = output_dir + 'error.txt'
right = open(correct_path,'w')
error = open(error_path,'w')
for i in range(len(files)):
  f = files[i]
  l = labels[i]
  p = preds[i]
  if l == p:
    right.write(f + '\t' + str(l) + '\t' + str(p) + '\n')
  else:
    error.write(f + '\t' + str(l) + '\t' + str(p) + '\n')
right.close()
error.close()




