#!usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function
import sys

from models.ConvNet import * 
from layers import *
from data_utils import *
from data_noise import *
from utils import *
ctx = check_ctx()

from optim import *

# from fast_progress import master_bar, progress_bar

# importing MxNet >= 1.0
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd, gluon
from multiprocessing import cpu_count
CPU_COUNT = cpu_count()

import random

# mx.random.seed(1)
# random.seed(1)


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

     #     self.update_rule = kwargs.pop('update_rule', 'sgd')
     #     self.optim_config = kwargs.pop('optim_config', {})
        self.stacking_size = kwargs.pop('stacking_size', 512)
        self.batch_size = kwargs.pop('batch_size', 256)
        assert self.stacking_size >= self.batch_size, "Note: stacking_size should be equal or greater than batch_size!"
        assert self.stacking_size <= min(self.train_ori.shape[0],self.test_ori.shape[0]), "Note: stacking_size should be equal or lesser than the shape of data!"
        
        self.lr_rate = kwargs.pop('lr_rate', 0.01)
        self.lr_decay = kwargs.pop('lr_decay', 0.01)
        self.num_epoch = kwargs.pop('num_epoch', 10)
        self.smoothing_constant = kwargs.pop('smoothing_constant', 0.01)
        
        self.save_checkpoints_address = kwargs.pop('save_checkpoints_address', './checkpoints/')
        self.allparams = kwargs.pop('allparams', None)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.floydhub_verbose = kwargs.pop('floydhub_verbose', False)
        self.oldversion = kwargs.pop('oldversion', False)
    #   self.print_every = kwargs.pop('print_every', 100)
        
        self.rand_times = kwargs.pop('rand_times', 2)  # 随机化扩充波形数据样本的倍数
        self.params = kwargs.pop('params', None)   # Transfer learning
        self.RandMLP = kwargs.pop('RandMLP', None)
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
        
        # Create mini-batches of GW waveforms for training & testing  写在初始化里
        train_wf = gluon.data.ArrayDataset(self.train_ori)            
        self.train_wf_loader = gluon.data.DataLoader(train_wf, batch_size=self.stacking_size, shuffle=True, last_batch='keep')
        test_wf = gluon.data.ArrayDataset(self.test_ori)            
        self.test_wf_loader = gluon.data.DataLoader(test_wf, batch_size=self.stacking_size, shuffle=True, last_batch='keep')


#         if not hasattr(optim, self.update_rule):
#             raise ValueError('Unrecognized update rule: "%s"' % self.update_rule)
#         self.update_rule = getattr(optim, self.update_rule)
        
        # if self.oldversion:
        #     self._reset_data_old()
        # else:
        #     self._random_data() 
        #     self._reset_data()
        #     print('SNR = %s' %self.SNR)
        #     print('Label for training:', self.y_train.shape)
        #     print('Label for testing:', self.y_test.shape)



#     def _reset_data(self):
        
#         try:
#             assert self.train.shape[1] == self.test.shape[1]
#         except:
#             print('self.train.shape[1] != self.test.shape[1]')


# #         self.param_noise = Pre_zero(size = (noiseAll_size,) + (self.train.shape[1:]))





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

        # self.optim_configs = {}
        # for p in self.model.params:
        #     d = {k: v for k, v in self.optim_config.items()}
        #     self.optim_configs[p] = d    


        # Opt. for Adam ############
        self.vs = []
        self.sqrs = []
        
        # Transfer Learning ########
        self.model.init_params()
        for key, params in self.params.items():
            if (params.shape[0] == self.model.flatten_dim) and (self.RandMLP):
                break
            self.model.params[key] = params.copy()
            
        if self.RandMLP:
            print('(Transfer Learning) Random the MLP part!')
        else:
            print('(Transfer Learning) NOT random the MLP part!')
        print('------------')
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
            
        # self.optim_configs = {}
        # for p in self.model.params:
        #     d = {k: v for k, v in self.optim_config.items()}
        #     self.optim_configs[p] = d    


        # Opt. for Adam ############
        self.vs = []
        self.sqrs = []
        
        # And assign space for gradients
        for param in self.model.params.values():
            param.attach_grad()
            self.vs.append(param.zeros_like())
            self.sqrs.append(param.zeros_like())
        print('------------')



    def _random_data(self):
        

        train_, train_shift_list = shuffle_data_nd(self.train,self.peak_samppoint, self.peak_time, self.rand_times)
        test_, test_shift_list = shuffle_data_nd(self.test,self.peak_samppoint, self.peak_time, self.rand_times)
        # self.train_shift_list.extend(train_shift_list.asnumpy().tolist())
        # self.test_shift_list.extend(test_shift_list.asnumpy().tolist())
        self.train = train_.reshape(self.train.shape[0]*self.rand_times,self.num_channel,-1).as_in_context(ctx)
        self.test = test_.reshape(self.test.shape[0]*self.rand_times,self.num_channel,-1).as_in_context(ctx)


    def _gen_yshape(self):
        self.train_size = self.train.shape[0]
        self.test_size = self.test.shape[0]
        self.noiseAll_size = self.train_size+self.test_size
        self.y_train = nd.concat(nd.ones(shape = (self.train_size,), ctx = ctx), nd.zeros(shape = (self.train_size,), ctx = ctx) , dim = 0)
        self.y_test = nd.concat(nd.ones(shape = (self.test_size,), ctx = ctx), nd.zeros(shape = (self.test_size,), ctx = ctx) , dim = 0)        

    def gen_noise(self):
        self.b = nd.array(pre_fir().reshape((-1,1)), ctx=ctx)
        self.pp = pre_fftfilt(self.b, shape = (self.noiseAll_size, self.train.shape[-1]), nfft=None)

        if ctx == mx.gpu():
            noise = GenNoise_matlab_nd(shape = (self.noiseAll_size, self.train.shape[-1]), params = self.pp)
        else:
            raise
        return noise

    def _reset_noise(self):
        
        # noise for mixing
        noise = self.gen_noise().reshape(shape= (self.noiseAll_size,) + (self.train.shape[1:]))

        try: sigma = self.train.max(axis = 2) / float(self.SNR) / nd.array(noise[:self.train_size].asnumpy().std(axis = 2,dtype='float64'),ctx=ctx)
        except: sigma = self.train.max(axis = -1) / float(self.SNR) / nd.array(noise[:self.train_size].asnumpy().std(axis = -1,dtype='float64'),ctx=ctx)            
        signal_train = nd.divide(self.train, sigma.reshape((self.train_size,self.num_channel,-1)))
        data_train = signal_train + noise[:self.train_size]
        
        try: sigma = self.test.max(axis = 2) / float(self.SNR) / nd.array(noise[-self.test_size:].asnumpy().std(axis = 2,dtype='float64'),ctx=ctx)
        except: sigma = self.test.max(axis = -1) / float(self.SNR) / nd.array(noise[-self.test_size:].asnumpy().std(axis = -1,dtype='float64'),ctx=ctx)
        signal_test = nd.divide(self.test, sigma.reshape((self.test_size,self.num_channel,-1)))
        data_test = signal_test + noise[-self.test_size:]
        
        # noise for pure conterpart
        noise = self.gen_noise().reshape(shape= (self.noiseAll_size,) + (self.train.shape[1:]))

        X_train = Normolise_nd(nd.concat(data_train, noise[:self.train_size], dim=0), self.num_channel)
        try: dataset_train = gluon.data.ArrayDataset(X_train, self.y_train)
        except: dataset_train = gluon.data.ArrayDataset(X_train, self.y_train)
        self.train_data = gluon.data.DataLoader(dataset_train, self.batch_size, shuffle=True,  last_batch='keep')
        
        X_test = Normolise_nd(nd.concat(data_test, noise[-self.test_size:], dim=0), self.num_channel)
        try: dataset_test = gluon.data.ArrayDataset(X_test, self.y_test)
        except: dataset_test = gluon.data.ArrayDataset(X_test, self.y_test)
        self.test_data = gluon.data.DataLoader(dataset_test, self.batch_size, shuffle=True, last_batch='keep')


    def Training(self, Iterator = False):
        print('=============================')
        print('Now! Training for SNR = %s!' %self.SNR)

        t = 0    
        try:

            for epoch in range(1, self.num_epoch + 1):
                self.epoch = epoch
                self.lr_rate = lr_decay(self.lr_rate, epoch, self.lr_decay)

                # Stacking all the waveform
                for _, (self.train, self.test) in enumerate(zip(self.train_wf_loader, self.test_wf_loader)):
                    self._random_data()     # data in GPU (waveform)
                    self._gen_yshape()      # data in GPU (label)
                    self._reset_noise()     # data in GPU (samples)

                    self._iteration(t, epoch)


                train_acc = self.check_acc(self.train_data)
                test_acc = self.check_acc(self.test_data)

                self._history_log_epoch(train_acc, test_acc)  # acc history for training/testing
                self._bestparams_filter(epoch, test_acc)


                if self.floydhub_verbose:
                    print('{"metric": "Train_acc. for SNR=%s in epoches", "value": %.4f}' %(str(self.SNR), self.train_acc_history[-1]) )
                    print('{"metric": "Test_acc. for SNR=%s in epoches", "value": %.4f}' %(str(self.SNR), self.test_acc_history[-1]) )
                else:
                    print("Epoch {:d}, lr: {:.2e} Moving_loss: {:.6f}, Epoch_loss(mean): {:.6f}, Train_acc {:.4f}, Test_acc {:.4f}(Best:{:.4f})".format(epoch, self.lr_rate, self.moving_loss_history[-1], np.mean(self.Epoch_loss), self.train_acc_history[-1], self.test_acc_history[-1], self.best_test_acc))
                    
                self._save_checkpoint()

        except KeyboardInterrupt as e:
            print(e)
            print('Early stoping at epoch=%s' %str(epoch))

        self.model.params = self.best_params
        print('Finished!')

    
    def _iteration(self, t, epoch):
        
        self.Epoch_loss = []  # log for training loss in each epoch

        # training loop (with autograd and trainer steps, etc.)
        for batch_i, ((data, label), (data_v, label_v) ) in enumerate(zip(self.train_data, self.test_data)):
            loss = self.loss(data, label, train = True)
            # Increment t before invoking adam.
            t += 1
            self.model.params, self.vs, self.sqrs = adam(self.model.params, self.vs, self.sqrs, self.lr_rate, self.batch_size, t)

            # Keep a moving average of the losses
            curr_loss = nd.mean(loss).asscalar()
            self.moving_loss = (curr_loss if ((batch_i == 0) and (epoch-1 == 0))
                           else (1 - self.smoothing_constant) * self.moving_loss + (self.smoothing_constant) * curr_loss)

            # validation 
            # r = random.randint(0, self.test_size*2//self.batch_size)
            # (data_v, label_v) = self.test_data._dataset[ r ]

            loss_v, _= self.loss(data_v, label_v, train = False)
            curr_loss_v = nd.mean(loss_v).asscalar()

            self._history_log_iteration(curr_loss, curr_loss_v)

            if self.floydhub_verbose:
                print('{"metric": "Training Loss for ALL", "value": %.5f}' %(curr_loss*1.0) )
                print('{"metric": "Testing Loss for ALL", "value": %.5f}' %(curr_loss_v*1.0) )
                print('{"metric": "Training Loss for SNR=%s", "value": %.5f}' %(str(self.SNR), curr_loss*1.0) )
                print('{"metric": "Testing Loss for SNR=%s", "value": %.5f}' %(str(self.SNR), curr_loss_v*1.0) )            
            else:
                print('Working on epoch {:d}. Curr_loss: {:.5f} (complete percent: {:.2f}/100)'.format(epoch, curr_loss*1.0, 1.0 * batch_i / (self.train_size/self.batch_size) * 100/ 2) , end='')
                sys.stdout.write("\r")
                




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
        
        
    def _history_log_epoch(self, train_acc, test_acc):
        self.train_acc_history.append(train_acc)
        self.test_acc_history.append(test_acc)

    def _history_log_iteration(self,curr_loss, curr_loss_v):      
        self.loss_history.append(curr_loss)
        self.loss_v_history.append(curr_loss_v)
        self.moving_loss_history.append(self.moving_loss)
        self.Epoch_loss.append(curr_loss)

    def _bestparams_filter(self, epoch, test_acc):

        if test_acc >= self.best_test_acc:
            self.best_test_acc = test_acc
            self.best_params = {}
            self.best_params_epoch = 0
            self.findabest = 0
            for k, v in self.model.params.items():
                self.best_params[k] = v.copy()
                self.best_params_epoch = epoch
                self.findabest = 1
    


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
            nd.save(file_address+'%s_best_params_epoch@%s_%s.pkl' %(self.checkpoint_name, self.best_params_epoch, self.best_test_acc) , self.best_params)
            self.findabest = 0
        # save all the parsms during the training
        if self.allparams:
            nd.save(file_address+'%s_params_epoch@%s.pkl' %(self.checkpoint_name, self.epoch), self.model.params)
        # save the processing info. within the training
        nd.save(file_address+'%s_info.pkl' %(self.checkpoint_name), checkpoint)
    
    
    def predict_nd(self):
        prob_list = []
        label_list = []
        for _, (self.train, self.test) in enumerate(zip(self.train_wf_loader, self.test_wf_loader)):
            self._random_data()
            self._gen_yshape()
            if self.oldversion: pass
            else: self._reset_noise()
            

            for batch_i, (data, label) in enumerate(self.test_data,):

                data = data.as_in_context(ctx).reshape((data.shape[0],self.num_channel,1,-1))
                label = label.as_in_context(ctx)
                label = nd.one_hot(label, self.model.output_dim).asnumpy()[:,1].tolist()
                output, _ = self.model.network(X=data)
                prob = transform_softmax(output)[:,1].asnumpy().tolist()
                prob_list.extend(prob)
                label_list.extend(label)
        return prob_list, label_list, output


if __name__ == '__main__':
    print('CPU or GPU? : ', ctx)
    
    