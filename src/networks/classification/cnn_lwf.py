import torch
import sys
import torch
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class Net(torch.nn.Module):

    def __init__(self, taskcla, args):
        super(Net, self).__init__()

        ncha = args.image_channel
        size = args.image_size

        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=2, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=False)

        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1000, bias=False)
        self.fc2 = torch.nn.Linear(1000, 1000, bias=False)

        torch.nn.init.xavier_normal(self.fc1.weight)
        torch.nn.init.xavier_normal(self.fc2.weight)

        self.args = args
        self.taskcla=taskcla

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(1000,args.nclasses)

        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n,_ in self.taskcla:
                self.last.append(torch.nn.Linear(1000,n))

        #My implementation about

        # pooled_out = []
        # for i, filter_size in enumerate(self.config.FILTER_SIZE):
        #     filter_shape = [filter_size, self.config.DIM, 1, self.config.NUMBER_OF_FEATURE_MAPS[i]]
        #     W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        #     # W = tf.get_variable(shape = filter_shape, initializer = tf.truncated_normal_initializer(stddev=0.001), name="W"+str(i))
        #     b = tf.Variable(tf.constant(0.1, shape=[self.config.NUMBER_OF_FEATURE_MAPS[i]]), name="b")
        #
        #     conv = tf.nn.conv2d(self.image_patches_reshaped, filter=W, strides=[1, 1, 1, 1], padding="VALID",
        #                         name="conv")
        #     # print(tf.Print(conv,[conv]))
        #     # conv = tf.squeeze(conv) # ( (BATCH_SIZE*WORDS), WINDOW_LEN-FILTER_SIZE + 1, NUMBER_OF_FEATURE_MAPS)
        #     # conv = tf.nn.bias_add(conv,b)
        #     # conv = tf.nn.relu(conv)
        #     # conv_non_linear = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") # ( (BATCH_SIZE*WORDS), WINDOW_LEN-FILTER_SIZE + 1, 1, NUMBER_OF_FEATURE_MAPS)
        #
        #     pooled = tf.nn.max_pool(conv, ksize=[1, (self.config.WINDOW_LEN - filter_size + 1), 1, 1],
        #                             strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC', name="pool")
        #     pooled = tf.squeeze(pooled)  # ( (BATCH_SIZE*WORDS), NUMBER_OF_FEATURE_MAPS)
        #     self.output = tf.reshape(pooled, (-1, tf.shape(self.word_ids)[1], self.config.NUMBER_OF_FEATURE_MAPS[i]))
        #     pooled_out.append(self.output)
        #
        #     self.h_pool = tf.concat(pooled_out, 2)

        # two  convolution layers, two max - pool layers, and a fully
        # connected layer with softmax output.The first convolution layer consisted of 10 0
        # feature maps with filter size 2. The second convolution layer had 50 feature maps
        # with filter size 3. The stride in each convolution layer is 1 as we wanted to tag
        # each word.A max-pooling layer followed each convolution layer.The pool size we use
        # in the max-pool layers was 2. We used regularization with dropout on the penultimate
        # layer with a constraint on L2-norms of the weight vectors, with 30 epochs.
        #
        # The output of each convolution layer was computed using a non-linear function;
        # in our case we used the hyperbolic tangent.

        self.last = torch.nn.Sequential()
        self.conv.add_module('conv1',torch.nn.Conv2d(ncha,10, kernel_size=3, stride=1))
        self.conv.add_module('tanh1', torch.nn.Tanh())

        self.conv.add_module('maxpool1', torch.nn.MaxPool2d(2))

        self.conv.add_module('conv2', torch.nn.Conv2d(10,50, kernel_size=2,stride=1))
        self.conv.add_module('tanh2', torch.nn.Tanh())
        self.conv.add_module('maxpool2', torch.nn.MaxPool2d(2))
        self.conv.add_module('drop2', torch.nn.Dropout(0.001))
        self.cov.add_module('fully_conected2', torch.nn.Linear(in_features=50,out_features=self.config.ntags))

        print('DIL CNN')

        return


    def forward(self,x):
        output_dict = {}

        h_list = []
        x_list = []
        # Gated
        x = self.padding(x)
        x_list.append(torch.mean(x, 0, True))
        con1 = self.drop1(self.relu(self.c1(x)))
        con1_p = self.maxpool(con1)

        con1_p = self.padding(con1_p)
        x_list.append(torch.mean(con1_p, 0, True))
        con2 = self.drop1(self.relu(self.c2(con1_p)))
        con2_p = self.maxpool(con2)

        con2_p = self.padding(con2_p)
        x_list.append(torch.mean(con2_p, 0, True))
        con3 = self.drop1(self.relu(self.c3(con2_p)))
        con3_p = self.maxpool(con3)

        h = con3_p.view(x.size(0), -1)
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))

        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i,_ in self.taskcla:
                y.append(self.last[t](h))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)
        output_dict['masks'] = None
        output_dict['x_list'] = x_list
        output_dict['h_list'] = h_list

        return output_dict

