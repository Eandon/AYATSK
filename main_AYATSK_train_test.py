import torch
from math import log, ceil
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from skfuzzsys import TSK
from train_test import *

# hyper-parameters
train_size = 0.9
learning_rate = 0.001
num_fuzzy_set = 3
max_epoch = 300
batch_size = 1.0

# load dataset
dataset_name = r'SRBCT'
dataset = torch.load(r'datasets/{}.pt'.format(dataset_name))
sample, target = dataset.sample, dataset.target

# one-hot the label
target = torch.LongTensor(preprocessing.OneHotEncoder().fit_transform(target).toarray())

# split train-test samples
tra_sam, test_sam, tra_tar, test_tar = train_test_split(sample, target, train_size=train_size)

# preprocessing, linearly normalize the training and test samples into the interval [0,1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
tra_sam = torch.Tensor(min_max_scaler.fit_transform(tra_sam))
test_sam = torch.Tensor(min_max_scaler.transform(test_sam))

# No. samples, features, and classes
num_tra_sam, num_fea = tra_sam.shape
num_class = tra_tar.shape[1]

# init the model
myTSK = TSK(num_fea, num_class, num_fuzzy_set, mf='CEMF', tnorm='yager_simple', order='first')
myTSK.antecedent.k_cemf = 10
myTSK.antecedent.lambda_yager = -log(num_fea) / log(1 - 1 / myTSK.antecedent.k_cemf)

# training and test
train_mini_batch(tra_sam, myTSK, tra_tar, learning_rate, max_epoch,
                 batch_size=ceil(num_tra_sam * batch_size), loss_type='mse_loss_fun', optim_type='Adam')

tra_loss, tra_acc = test(tra_sam, myTSK, tra_tar)
test_loss, test_acc = test(test_sam, myTSK, test_tar)
print(fr'{dataset_name} dataset, training acc: {tra_acc:.4f}, test acc: {test_acc:.4f}')
