
#!usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function
import sys

from models.ConvNet import * 
from layers import *
from data_utils import *
from utils import *
ctx = check_ctx()

from optim import *

# from fast_progress import master_bar, progress_bar

# importing MxNet >= 1.0
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd, gluon

import random

mx.random.seed(1)
random.seed(1)


def evaluate_accuracy(data_iterator, num_examples, batch_size, params, net, pool_type,pool_size,pool_stride):
    numerator = 0.
    denominator = 0.
    for batch_i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((data.shape[0],1,1,-1))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output, _ = net(data, params,pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
        print('Evaluating accuracy. (complete percent: %.2f/100' %(1.0 * batch_i / (num_examples//batch_size) * 100) +')' , end='')
        sys.stdout.write("\r")
    return (numerator / denominator).asscalar()


def Solver(train, test, Debug, batch_size, lr
          , smoothing_constant, num_fc1, num_fc2, num_outputs, epochs, SNR
          , sl, pool_type ,pool_size ,pool_stride, params_init=None, period=None):
    
    num_examples = train.shape[0]
    # 训练集数据类型转换
    y = nd.array(~train.sigma.isnull() +0)
    X = nd.array(Normolise(train.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
    print('Label for training:', y.shape)
    print('Dataset for training:', X.shape, end='\n\n')

    dataset_train = gluon.data.ArrayDataset(X, y)
    train_data = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True, last_batch='keep')

    y = nd.array(~test.sigma.isnull() +0)
    X = nd.array(Normolise(test.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
    print('Label for testing:', y.shape)
    print('Dataset for testing:', X.shape, end='\n\n')
    
    # 这里使用data模块来读取数据。创建测试数据。  (suffle)
    dataset_test = gluon.data.ArrayDataset(X, y)
    test_data = gluon.data.DataLoader(dataset_test, batch_size, shuffle=True, last_batch='keep')

    
    # Train
    loss_history = []
    loss_v_history = []
    moving_loss_history = []
    test_accuracy_history = []
    train_accuracy_history = []
    
#     assert period >= batch_size and period % batch_size == 0
    
    # Initializate parameters
    if params_init:
        print('Loading params...')
        params = params_init

        [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6] = params

        # random fc layers
        weight_scale = .01
        W7 = nd.random_normal(loc=0, scale=weight_scale, shape=(sl, num_fc1), ctx=ctx )
        W8 = nd.random_normal(loc=0, scale=weight_scale, shape=(num_fc1, num_fc2), ctx=ctx )        
        W9 = nd.random_normal(loc=0, scale=weight_scale, shape=(num_fc2, num_outputs), ctx=ctx )
        b7 = nd.random_normal(shape=num_fc1, scale=weight_scale, ctx=ctx)
        b8 = nd.random_normal(shape=num_fc2, scale=weight_scale, ctx=ctx)    
        b9 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)  

        params = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6]
        print('Random the FC1&2-layers...')

        vs = []
        sqrs = [] 
        for param in params:
            param.attach_grad()
            vs.append(param.zeros_like())
            sqrs.append(param.zeros_like())              
    else:
        params, vs, sqrs = init_params(num_fc1 = 64, num_fc2 = 64, num_outputs = 2, sl=sl)
        print('Initiate weights from random...')

    # Debug
    if Debug:
        print('Debuging...')
        if params_init:
            params = params_init
        else:
            params, vs, sqrs = init_params(num_fc1 = 64, num_fc2 = 64, num_outputs = 2, sl=sl)
        for data, _ in train_data:
            data = data.as_in_context(ctx).reshape((batch_size,1,1,-1))
            break
        print(pool_type, pool_size, pool_stride)
        _, _ = ConvNet(data, params, debug=Debug, pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        print()
    
#     total_loss = [Total_loss(train_data_10, params, batch_size, num_outputs)]
    
    t = 0
#   Epoch starts from 1.
    print('pool_type: ', pool_type)
    print('pool_size: ', pool_size)
    print('pool_stride: ', pool_stride)
    print('sl: ', sl)
    for epoch in range(1, epochs + 1):
        Epoch_loss = []
#         学习率自我衰减。
        if epoch > 2:
#             lr *= 0.1
            lr /= (1+0.01*epoch)
        for batch_i, ((data, label),(data_v, label_v)) in enumerate(zip(train_data, test_data)):
            data = data.as_in_context(ctx).reshape((data.shape[0],1,1,-1))
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, num_outputs)
            with autograd.record():
                output, _ = ConvNet(data, params, pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
#             print(output)
            # params = sgd(params, lr, batch_size)

#           Increment t before invoking adam.
            t += 1
            params, vs, sqrs = adam(params, vs, sqrs, lr, batch_size, t)

            data_v = data_v.as_in_context(ctx).reshape((data_v.shape[0],1,1,-1))
            label_v = label_v.as_in_context(ctx)
            label_v_one_hot = nd.one_hot(label_v, num_outputs)
            output_v, _ = ConvNet(data_v, params, pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
            loss_v = softmax_cross_entropy(output_v, label_v_one_hot)            
            
#             #########################
#              Keep a moving average of the losses
#             #########################
            curr_loss = nd.mean(loss).asscalar()
            curr_loss_v = nd.mean(loss_v).asscalar()
            moving_loss = (curr_loss if ((batch_i == 0) and (epoch-1 == 0))
                           else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

            loss_history.append(curr_loss)
            loss_v_history.append(curr_loss_v)
            moving_loss_history.append(moving_loss)
            Epoch_loss.append(curr_loss)
#             if batch_i * batch_size % period == 0:
#                 print('Curr_loss: ', curr_loss)
                
            print('Working on epoch %d. Curr_loss: %.5f (complete percent: %.2f/100' %(epoch, curr_loss*1.0, 1.0 * batch_i / (num_examples//batch_size) * 100) +')' , end='')
            sys.stdout.write("\r")
            # print('{"metric": "Training Loss for ALL", "value": %.5f}' %(curr_loss*1.0) )
            # print('{"metric": "Testing Loss for ALL", "value": %.5f}' %(curr_loss_v*1.0) )
#             print('{"metric": "Training Loss for SNR=%s", "value": %.5f}' %(str(SNR), curr_loss*1.0) )
#             print('{"metric": "Testing Loss for SNR=%s", "value": %.5f}' %(str(SNR), curr_loss_v*1.0) )
        train_accuracy = evaluate_accuracy(train_data, num_examples, batch_size, params, ConvNet,pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        test_accuracy = evaluate_accuracy(test_data, num_examples, batch_size, params, ConvNet,pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        test_accuracy_history.append(test_accuracy)
        train_accuracy_history.append(train_accuracy)


        print("Epoch %d, Moving_loss: %.6f, Epoch_loss(mean): %.6f, Train_acc %.4f, Test_acc %.4f" %
              (epoch, moving_loss, np.mean(Epoch_loss), train_accuracy, test_accuracy))
#         print('{"metric": "Train_acc. for SNR=%s in epoches", "value": %.4f}' %(str(SNR), train_accuracy) )
#         print('{"metric": "Test_acc. for SNR=%s in epoches", "value": %.4f}' %(str(SNR), test_accuracy) )
        yield (params, loss_history, loss_v_history, moving_loss_history, test_accuracy_history, train_accuracy_history)
    
    
    
def predict(data, net, params):

    X = nd.array(Normolise(data.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
    # num_examples = data.shape[0]
    # batch_size = [2**i for i in range(10) if num_examples%(2**i) == 0][-1]
    # print('Batch size = %s' %batch_size)
    data = nd.array(X).as_in_context(ctx).reshape((-1,1,1,8192))
    output, interlayer = net(data, params)
    prob = transform_softmax(output)[:,1].asnumpy().tolist()[0]
    return prob, interlayer



    
def predict_(data, net, params):

    X = nd.array(Normolise(data))
    # num_examples = data.shape[0]
    # batch_size = [2**i for i in range(10) if num_examples%(2**i) == 0][-1]
    # print('Batch size = %s' %batch_size)
    data = nd.array(X).as_in_context(ctx).reshape((-1,1,1,8192))
    output, _ = net(data, params)
    prob = transform_softmax(output)[:,1].asnumpy().tolist()[0]
    return prob, output



if __name__ == '__main__':
    print('CPU or GPU? : ', ctx)
    
    
    
class Solver_nd(object):
    
    def __init__(self, model, train, test, SNR, **kwargs):
        self.model = model
        self.num_channel = model.input_dim[0]
        self.train_ori = train
        self.test_ori = test
        self.peak_samppoint, self.peak_time = cal_peak_nd(train)
        self.train_shift_list = []
        self.test_shift_list = []        

        try:
            assert self.train_ori.shape == self.test_ori.shape
        except:
            print('self.train_ori.shape != self.test_ori.shape')
        
        self.SNR = SNR

#         self.update_rule = kwargs.pop('update_rule', 'sgd')
#         self.optim_config = kwargs.pop('optim_config', {})

        self.batch_size = kwargs.pop('batch_size', 256)
        self.lr_rate = kwargs.pop('lr_rate', 0.01)
        self.lr_decay = kwargs.pop('lr_decay', 0.01)
        self.num_epoch = kwargs.pop('num_epoch', 10)
        self.smoothing_constant = kwargs.pop('smoothing_constant', 0.01)
        
        self.save_checkpoints_address = kwargs.pop('save_checkpoints_address', './checkpoints/')
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.verbose = kwargs.pop('verbose', False)
        self.oldversion = kwargs.pop('oldversion', False)
#         self.print_every = kwargs.pop('print_every', 100)
        
    
        self.params = kwargs.pop('params', None)   # Transfer learning
        if self.params:  # 若有迁移学习
            self.params = self.params.copy()
            try:         # 考察导入的模型参数变量 与 导入模型的参数之间得到关系
                assert [np.allclose(p1.asnumpy(), p2.asnumpy()) for (_,p1), (_,p2) in zip(self.params.items(), model.params.items())]

            except:
                print('导入的模型参数与导入模型现默认参数有着相同的值~')
                raise
            self._reset_params_Transfer()
        else:
            self._reset_params()

        if len(kwargs) != 0:
            extra = ', '.join('"%s"' %k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)
            
#         if not hasattr(optim, self.update_rule):
#             raise ValueError('Unrecognized update rule: "%s"' % self.update_rule)
#         self.update_rule = getattr(optim, self.update_rule)
        
        if self.oldversion:
            self._reset_data_old()
        else:
            self._random_data()
            self._reset_data()
            print('SNR = %s' %self.SNR)
            print('Label for training:', self.y_train.shape)
            print('Label for testing:', self.y_test.shape)



    def _reset_data_old(self):
        try:
            assert self.train_ori.shape[1] == self.test_ori.shape[1]
        except:
            print('self.train_ori.shape[1] != self.test_ori.shape[1],',self.train_ori.shape[1],self.test_ori.shape[1])
        self.train_ori_size = self.train_ori.shape[0]
        self.test_ori_size = self.test_ori.shape[0]
        
        y = nd.array(~self.train_ori.sigma.isnull() +0)
        X = nd.array(Normolise(self.train_ori.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
        print('Label for training:', y.shape)
        print('Dataset for training:', X.shape, end='\n\n')

        dataset_train = gluon.data.ArrayDataset(X, y)
        self.train_data = gluon.data.DataLoader(dataset_train, self.batch_size, shuffle=True, last_batch='keep')

        y = nd.array(~self.test_ori.sigma.isnull() +0)
        X = nd.array(Normolise(self.test_ori.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
        print('Label for testing:', y.shape)
        print('Dataset for testing:', X.shape, end='\n\n')

        dataset_test = gluon.data.ArrayDataset(X, y)
        self.test_data = gluon.data.DataLoader(dataset_test, self.batch_size, shuffle=True, last_batch='keep')        


    def _random_data(self):
        
        self.train, train_shift_list = shuffle_data_nd(self.train_ori,self.peak_samppoint, self.peak_time, 10)
        self.test, test_shift_list = shuffle_data_nd(self.test_ori,self.peak_samppoint, self.peak_time, 10)

        self.train_shift_list.extend(train_shift_list.asnumpy().tolist())
        self.test_shift_list.extend(test_shift_list.asnumpy().tolist())
        self.train = self.train.reshape(self.train_ori.shape[0]*10,self.num_channel,-1)
        self.test = self.test.reshape(self.test_ori.shape[0]*10,self.num_channel,-1)
        
        
    def _reset_data(self):
        
        try:
            assert self.train.shape[1] == self.test.shape[1]
        except:
            print('self.train.shape[1] != self.test.shape[1]')
        
        self.train_size = self.train.shape[0]
        self.test_size = self.test.shape[0]
        noiseAll_size = self.train_size+self.test_size

        self.param_noise = Pre_zero(size = (noiseAll_size,) + (self.train.shape[1:]))

        self.y_train = nd.concat(nd.ones(shape = (self.train_size,), ctx = ctx), nd.zeros(shape = (self.train_size,), ctx = ctx) , dim = 0)
        self.y_test = nd.concat(nd.ones(shape = (self.test_size,), ctx = ctx), nd.zeros(shape = (self.test_size,), ctx = ctx) , dim = 0)


    def _reset_params_Transfer(self):
        self.epoch = 0
        self.best_test_acc = 0
        self.best_params = {}
        self.moving_loss = 0

        self.train_acc_history = []
        self.test_acc_history = []

        self.loss_history = []
        self.loss_v_history = []
        self.moving_loss_history = []
            
#         self.optim_configs = {}
#         for p in self.model.params:
#             d = {k: v for k, v in self.optim_config.items()}
#             self.optim_configs[p] = d    


        # Opt. for Adam ############
        self.vs = []
        self.sqrs = []
        
        # Transfer Learning ########
        self.model.init_params()
        for key, params in self.params.items():
            if params.shape[0] == self.model.flatten_dim:
                break
            self.model.params[key] = params.copy()

        # And assign space for gradients
        for param in self.model.params.values():
            param.attach_grad()
            self.vs.append(param.zeros_like())
            self.sqrs.append(param.zeros_like())        

    def _reset_params(self):
        self.epoch = 0
        self.best_test_acc = 0
        self.best_params = {}
        self.moving_loss = 0

        self.train_acc_history = []
        self.test_acc_history = []

        self.loss_history = []
        self.loss_v_history = []
        self.moving_loss_history = []
            
#         self.optim_configs = {}
#         for p in self.model.params:
#             d = {k: v for k, v in self.optim_config.items()}
#             self.optim_configs[p] = d    


        # Opt. for Adam ############
        self.vs = []
        self.sqrs = []

        # And assign space for gradients
        for param in self.model.params.values():
            param.attach_grad()
            self.vs.append(param.zeros_like())
            self.sqrs.append(param.zeros_like())

        


    def Training(self, Iterator = False):
        
        t = 0    
        try:
#             self.mb = master_bar(range(1, self.num_epoch + 1))
#             self.mb.names = ['loss', 'loss_var']
            
            # if self.oldversion: pass
            # else: self._reset_noise()
            for epoch in range(1, self.num_epoch + 1):
#             for epoch in self.mb:
                self.epoch = epoch
                self.lr_rate = lr_decay(self.lr_rate, epoch, self.lr_decay)

                self._random_data()

                if self.oldversion: pass
                else: self._reset_noise()

                self._iteration(t, epoch)
                
                self.train_acc_history.append(self.check_acc(self.train_data))
                val_acc = self.check_acc(self.test_data)
                self.test_acc_history.append(val_acc)


                if val_acc >= self.best_test_acc:
                    self.best_test_acc = val_acc
                    self.best_params = {}
                    self.best_params_epoch = 0
                    self.findabest = 0
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
                        self.best_params_epoch = epoch
                        self.findabest = 1

                if self.verbose:
                    print('{"metric": "Train_acc. for SNR=%s in epoches", "value": %.4f}' %(str(self.SNR), self.train_acc_history[-1]) )
                    print('{"metric": "Test_acc. for SNR=%s in epoches", "value": %.4f}' %(str(self.SNR), self.test_acc_history[-1]) )
                else:
                    print("Epoch {:d}, Moving_loss: {:.6f}, Epoch_loss(mean): {:.6f}, Train_acc {:.4f}, Test_acc {:.4f}(Best:{:.4f})".format(epoch, self.moving_loss_history[-1], np.mean(self.Epoch_loss), self.train_acc_history[-1], self.test_acc_history[-1], self.best_test_acc))
                    
                self._save_checkpoint()

#                     self.mb.first_bar.comment = f'first bar stat'
#                     self.mb.write(f'Finished loop {epoch}')
        
#                 if Iterator:
#                     yield self.loss_history, self.loss_v_history, self.moving_loss_history, self.train_acc_history, self.test_acc_history

        except KeyboardInterrupt as e:
            print(e)
            print('Early stoping at epoch=%s' %str(epoch))

        self.model.params = self.best_params
        print('Finished!')
    
    def _iteration(self, t, epoch):
        
        self.Epoch_loss = []

        
        for batch_i, ((data, label),(data_v, label_v)) in enumerate(zip(self.train_data, self.test_data,)):

            loss = self.loss(data, label, train = True)
            loss_v, _= self.loss(data_v, label_v, train = False)
#            print(loss)
            # Increment t before invoking adam.
            t += 1
            self.model.params, self.vs, self.sqrs = adam(self.model.params, self.vs, self.sqrs, self.lr_rate, self.batch_size, t)
        
            # Keep a moving average of the losses
            curr_loss = nd.mean(loss).asscalar()
            curr_loss_v = nd.mean(loss_v).asscalar()
            self.moving_loss = (curr_loss if ((batch_i == 0) and (epoch-1 == 0))
                           else (1 - self.smoothing_constant) * self.moving_loss + (self.smoothing_constant) * curr_loss)

            self.loss_history.append(curr_loss)
            self.loss_v_history.append(curr_loss_v)
            self.moving_loss_history.append(self.moving_loss)
            self.Epoch_loss.append(curr_loss)

            if self.verbose:
                print('{"metric": "Training Loss for ALL", "value": %.5f}' %(curr_loss*1.0) )
                print('{"metric": "Testing Loss for ALL", "value": %.5f}' %(curr_loss_v*1.0) )
                print('{"metric": "Training Loss for SNR=%s", "value": %.5f}' %(str(self.SNR), curr_loss*1.0) )
                print('{"metric": "Testing Loss for SNR=%s", "value": %.5f}' %(str(self.SNR), curr_loss_v*1.0) )            
            else:
                print('Working on epoch {:d}. Curr_loss: {:.5f} (complete percent: {:.2f}/100)'.format(epoch, curr_loss*1.0, 1.0 * batch_i / (self.train_size/self.batch_size) * 100/ 2) , end='')
                sys.stdout.write("\r")
                
#                 x = np.arange(1, len(self.loss_history)+1,1)
#                 graphs = [[x, self.loss_history], [x, self.loss_v_history]]
#                 x_bounds = [0, 12]
#                 y_bounds = [0, 1]
#                 self.mb.update_graph(graphs, x_bounds, y_bounds)
#                 self.mb.child.comment = f'second bar stat'



    def loss(self, data, label, train=True):
        data = data.as_in_context(ctx).reshape((data.shape[0],self.num_channel,1,-1))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, self.model.output_dim)
        
        if train:
            with autograd.record():
                output, _= self.model.network(X=data)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
            return loss
        else:
            output, _ = self.model.network(X=data)
            loss = softmax_cross_entropy(output, label_one_hot)
            return loss, output
        
        

    def gen_noise(self):
        
        if ctx == mx.gpu():
            noise, _ = TimeseriesFromPSD_nd(self.param_noise)
        elif ctx == mx.cpu():
            noise, _ = TimeseriesFromPSD(self.param_noise)
            noise = nd.array(noise)
            
        return noise

    

    def _reset_noise(self):
        
        # noise for mixing
        noise = self.gen_noise()

        try: sigma = self.train.max(axis = 2) / float(self.SNR) / nd.array(noise[:self.train_size].asnumpy().std(axis = 2,dtype='float64'),ctx=ctx)
        except: sigma = self.train.max(axis = -1) / float(self.SNR) / nd.array(noise[:self.train_size].asnumpy().std(axis = -1,dtype='float64'),ctx=ctx)            
        self.sigma = sigma
        signal_train = nd.divide(self.train, sigma.reshape((self.train_size,self.num_channel,-1)))
        data_train = signal_train + noise[:self.train_size]
        
        try: sigma = self.test.max(axis = 2) / float(self.SNR) / nd.array(noise[-self.test_size:].asnumpy().std(axis = 2,dtype='float64'),ctx=ctx)
        except: sigma = self.test.max(axis = -1) / float(self.SNR) / nd.array(noise[-self.test_size:].asnumpy().std(axis = -1,dtype='float64'),ctx=ctx)
        signal_test = nd.divide(self.test, sigma.reshape((self.test_size,self.num_channel,-1)))    
        data_test = signal_test + noise[-self.test_size:]
        

        # noise for pure conterpart
        noise = self.gen_noise()
        
        X_train = Normolise_nd(nd.concat(data_train, noise[:self.train_size], dim=0), self.num_channel)
        try: dataset_train = gluon.data.ArrayDataset(X_train, self.y_train)
        except: dataset_train = gluon.data.ArrayDataset(X_train, self.y_train)
        self.train_data = gluon.data.DataLoader(dataset_train, self.batch_size, shuffle=True, last_batch='keep')
        
        X_test = Normolise_nd(nd.concat(data_test, noise[-self.test_size:], dim=0), self.num_channel)
        try: dataset_test = gluon.data.ArrayDataset(X_test, self.y_test)
        except: dataset_test = gluon.data.ArrayDataset(X_test, self.y_test)
        self.test_data = gluon.data.DataLoader(dataset_test, self.batch_size, shuffle=True, last_batch='keep')
        

    
    def check_acc(self, data_iterator):
        numerator = 0.
        denominator = 0.
        for batch_i, (data, label) in enumerate(data_iterator):
            _, output = self.loss(data, label, train = False)
            predictions = nd.argmax(output, axis=1).as_in_context(ctx)
            numerator += nd.sum(predictions == label.as_in_context(ctx))
            denominator += data.shape[0]
            print('Evaluating accuracy. (complete percent: {:.2f}/100)'.format(1.0 * batch_i / (self.train_size/self.batch_size) * 100 /2)+' '*20, end='')
            sys.stdout.write("\r")        

        return (numerator / denominator).asscalar()


    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        
        checkpoint = {
#           'update_rule': self.update_rule,
          'lr_decay': nd.array([self.lr_decay]),
          'lr_rate': nd.array([self.lr_rate]),
#           'optim_config': self.optim_config,
          'batch_size': nd.array([self.batch_size]),
#           'num_train_samples': self.num_train_samples,
#           'num_val_samples': self.num_val_samples,
          'train_shift_list': nd.array(self.train_shift_list),
          'test_shift_list': nd.array(self.test_shift_list),
          'num_epoch': nd.array([self.num_epoch]),
          'epoch': nd.array([self.epoch]),
          'loss_history': nd.array(self.loss_history),
          'loss_v_history': nd.array(self.loss_v_history),
          'moving_loss_history': nd.array(self.moving_loss_history),
          'train_acc_history': nd.array(self.train_acc_history),
          'test_acc_history': nd.array(self.test_acc_history),
        }
        
        file_address = self.save_checkpoints_address
        # save the model modification
        if self.epoch == 1: 
            os.system('mkdir -p %s' %file_address)
            np.save(file_address+ '%s_structure_epoch.pkl' %(self.checkpoint_name) ,self.model.structure)    
        # save the best params
        if self.findabest:
            os.system('rm -rf '+file_address+'%s_best_params_epoch@*' %self.checkpoint_name)
            nd.save(file_address+'%s_best_params_epoch@%s.pkl' %(self.checkpoint_name, self.best_params_epoch) , self.best_params)
            self.findabest = 0
        # save all the parsms during the training
        nd.save(file_address+'%s_params_epoch@%s.pkl' %(self.checkpoint_name, self.epoch), self.model.params)
        # save the processing info. within the training
        nd.save(file_address+'%s_info.pkl' %(self.checkpoint_name), checkpoint)
    
    
    def predict_nd(self):

        self._random_data()
        
        if self.oldversion: pass
        else: self._reset_noise()
        
        prob_list = []
        label_list = []
        for batch_i, (data, label) in enumerate(self.test_data,):

            data = data.as_in_context(ctx).reshape((data.shape[0],self.num_channel,1,-1))
            label = label.as_in_context(ctx)
            label = nd.one_hot(label, self.model.output_dim).asnumpy()[:,1].tolist()
            output, _ = self.model.network(X=data)
            prob = transform_softmax(output)[:,1].asnumpy().tolist()
            prob_list.extend(prob)
            label_list.extend(label)
        return prob_list, label_list, output
