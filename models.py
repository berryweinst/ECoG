import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py as h5
from scipy import stats

import torch
from torch import FloatTensor
from torch.autograd import Variable
from torch.nn import Linear, Module, BatchNorm1d, Dropout, ReLU6
from torch.nn.functional import sigmoid, softmax

# from attention import ScaledDotProductAttention
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import Ridge
# from lightning.regression import CDRegressor
from ridge import *
from functions import *
import utils

zscore = lambda v: (v-v.mean(0))/v.std(0)

def Volatile(x):
    return Variable(x, volatile=True)


class Data(object):

    @staticmethod
    def create_minibatches(image_model, batch_size, electrode):
        ecog_data = pickle.load(open("data_dict_src.p", "rb"))
        batch_ecog_data = ecog_data[0:int(ecog_data.shape[0] / batch_size) * batch_size, :, :].reshape(
            (int(ecog_data.shape[0] / batch_size), ecog_data.shape[1], batch_size, ecog_data.shape[2]))
        f = h5.File("features.h5", "r")
        minibatches = []
        for idx, e in enumerate(batch_ecog_data):
            features = {}
            for layer in list(f[image_model]):
                features[layer] = FloatTensor(np.array(f[image_model][layer][idx * batch_size: idx * batch_size + batch_size]))
            target = FloatTensor(e[electrode, :, :])
            minibatches.append((features, target))
        return minibatches

    @staticmethod
    def create_train_test_data_for_reg(image_model, mode='full', hfile='features.h5'):
        ecog_data = pickle.load(open("data_dict_src.p", "rb"))
        f = h5.File(hfile, "r")
        dataset, layer = extract_last_dict_samples(f[image_model])
        # features = {}
        minibatches = []
        for idx, e in enumerate(ecog_data):
            # features = {}
            # for layer in list(f[image_model]):
            # features[layer] = np.array(dataset[idx])
            # if mode == 'full':
            #     features[layer] = features[layer].reshape(
            #         (features[layer].shape[0] * features[layer].shape[1] * features[layer].shape[2]))
            # if mode == 'voxel':
            #     features[layer] = features[layer].reshape(
            #         (features[layer].shape[0] * features[layer].shape[1], features[layer].shape[2]))
            # elif mode == 'avg':
            #     features[layer] = features[layer].reshape(
            #         (features[layer].shape[0] * features[layer].shape[1], features[layer].shape[2]))
            #     features[layer] = np.mean(features[layer], axis=0)
            # features[layer] = zscore(np.array(features[layer]))
            # ecog = e[electrode, :]
            dim = np.array(dataset[idx]).shape
            minibatches.append((zscore(np.array(dataset[idx]).reshape(
                    (dim[0] * dim[1] * dim[2]))), e))
        return minibatches, layer


class Fetures2ECoGTrans(Module):
    def __init__(self, features_dim, hidden_dim):
        super(Fetures2ECoGTrans, self).__init__()
        self.hidden_dim = hidden_dim
        self.features_dim = features_dim
        self.f1 = Linear(self.features_dim[1] * self.features_dim[2], 1)
        # self.d = Dropout()
        self.f2 = Linear(self.features_dim[3], 1)
        self.b = BatchNorm1d(self.features_dim[3], affine=False)
        self.attention = ScaledDotProductAttention(self.features_dim[3], return_weight=True)
        self.activation = ReLU6()


    def forward(self, tensor):
        w = 0
        tensor_flat = tensor.view(tensor.shape[0],
                                  tensor.shape[1] * tensor.shape[2], tensor.shape[3])
        # norm = tensor_flat.norm(p=2, dim=1, keepdim=True)
        tensor_q = tensor_flat.transpose(1, 2)
        tensor_q = self.f1(tensor_q)
        # tensor_q = self.b(tensor_q)
        tensor_q = self.activation(tensor_q)
        tensor_q = tensor_q.view(tensor_q.shape[0], tensor_q.shape[1])
        # tensor_lin = tensor_lin.view(tensor.shape[0], tensor.shape[1], tensor.shape[3])
        # w, tensor_att = self.attention(tensor_q, tensor_flat, tensor_flat)
        # tensor_att = tensor_att.view(tensor_att.shape[0], tensor_att.shape[2])
        # tensor_att = torch.sum(tensor_att, dim=1)
        # tensor_att = self.d(tensor_att)
        # norm = tensor_q.norm(p=2, dim=1, keepdim=True)
        tensor_lin = self.f2(tensor_q)
        # norm = tensor_lin.norm(p=2, dim=1, keepdim=True)
        # tensor_lin = self.b(tensor_lin)
        # tensor_lin = self.f2(tensor_lin)
        # tensor_lin_flat = tensor_lin.view((tensor_lin.shape[0] * tensor_lin.shape[1]))
        # tensor_bn = self.b(tensor_lin)
        # tensor_out = tensor_sm.view((tensor_lin.shape[0], tensor_lin.shape[1]))
        # return w, torch.div(tensor_lin, norm)
        return w, tensor_lin

