#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   models.py
@Desciption     :   None
@Modify Time      @Author    @Version
------------      -------    --------
2019/9/26 13:38   Daic       1.0
'''
import time
import torch
import torch.nn as nn
import numpy as np
from untils import *
from dataset import *
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR

class MultiClsTrainer(object):
    def __init__(self, opt):
        self.opt = opt
        print(opt)
        self.save_path = os.path.join(self.opt.out_path,self.opt.train_name)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        self.train_bch = opt.train_bch
        self.val_bch = opt.val_bch
        self.num_worker  = opt.num_worker
        self.model = build_cls_model(self.opt.cnn,active = 'relu',class_num=self.opt.num_class,chanel_num=3)#relu
        #print(self.nmodel)
        self.model.cuda()
        #self.model = torch.nn.DataParallel(self.nmodel)

        self.val_dataset = ClsSteelDataset(self.opt, 'val')
        self.train_arguementation = get_training_augmentation()
        self.train_dataset = ClsSteelDataset(self.opt,'train',self.train_arguementation)
        print("Initial dataset success! The train length: %d  The val length: %d"%
              (len(self.train_dataset),len(self.val_dataset)))

        self.train_loader = DataLoader(dataset=self.train_dataset,shuffle=True ,
                                       batch_size=self.train_bch, num_workers=self.num_worker)
        self.val_loader   = DataLoader(dataset=self.val_dataset,shuffle=False ,
                                       batch_size=self.val_bch, num_workers=self.num_worker-4)

        self.loss = build_loss_function(self.opt)

        if self.opt.split_weights == 1:
            params = split_weights(self.model)
        else:
            params = self.model.parameters()

        self.optimizer = build_optimizer(params, self.opt)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.opt.decay_step, gamma=self.opt.decay_rate)


        self.max_epoch = self.opt.max_epoch
        self.max_score = 0.

        self.best_acc = 0.5

    def resum_load(self,path,mutigpu=False):
        net_data = torch.load(path)

        if mutigpu:
            for k, v in net_data.items():
                name = k[7:]
                net_data[name] = net_data.pop(k)
        self.model.load_state_dict(torch.load(net_data))
        print("Loading success from %s"%path)

    def lr_finder(self,epoch,save_path,slr=1e-8,elr=1):
        log_lrs = []
        log_loss = []
        all_iter = self.train_dataset.__len__() * epoch // self.train_bch
        global_iter = 0.
        _lr__ = slr
        mult = (elr / slr) ** (1 / all_iter)

        for epoch in range(epoch):
            for i, data in enumerate(self.train_loader):
                _lr__ *= mult
                set_lr(self.optimizer, _lr__)
                log_lrs.append(math.log10(_lr__))
                img = data[0].cuda()
                label = data[1].cuda()
                outputs = self.model(img)
                loss = self.loss(outputs, label)
                log_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                global_iter+=1
                print("%.1f / %.1f  loss:%.6f  lr%.8f"%(global_iter,all_iter,loss.item(),_lr__))
        logs = {'lr':log_lrs,'loss':log_loss}
        json.dump(logs,open(os.path.join(save_path,'find_lr.json'),'w'))
        return logs



    def eval_model(self):
        self.model.eval()
        pre = []
        gt = []

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                img = data[0].cuda()
                label = data[1]
                label = label.numpy().astype(int).tolist()
                outputs = self.model(img).data.cpu()#.squeeze().numpy()#b
                #print(outputs)

                #predicted = (outputs>0.5).astype(int).tolist()
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.squeeze().numpy().tolist()
                #print(predicted)
                #print(label)
                pre += predicted
                gt += label
        print(classification_report(gt,pre))
        accs = classification_report(gt,pre,output_dict=True)
        return accs

    def train_model(self):
        global_iter = 0
        running_loss = 0.
        item_num = 0
        start = time.time()
        for epoch in range(self.max_epoch):
            print('\nEpoch: {}'.format(epoch))
            for i, data in enumerate(self.train_loader):
                img = data[0].cuda()
                label = data[1].cuda()
                #print(label)
                item_num += label.size(0)
                outputs = self.model(img)
                loss = self.loss(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i% 10 == 0 and i>0:
                    print('[%d, %d]   loss: %.5f    lr:%.5f    time:%.2f' %
                          (epoch, i, running_loss / item_num,self.optimizer.param_groups[0]['lr'],
                           (time.time()-start))
                          )
                    running_loss = 0.
                    item_num = 0
                    start = time.time()

                global_iter+=1

            acc = self.eval_model()
            json.dump(acc, open(os.path.join(self.save_path, 'log_epoch' + str(epoch) + '.json'), 'w'))

            # print("epoch %d:"%(epoch))
            # print('macro avg',acc['macro avg'])
            # print('weighted avg', acc['weighted avg'])
            if self.best_acc<= acc["accuracy"]:
                torch.save(self.model.state_dict(),os.path.join(self.save_path,'model.pth'))
                print("saving model....")
                self.best_acc = acc["accuracy"]
            if epoch == (self.max_epoch-1):
                torch.save(self.model.state_dict(), os.path.join(self.save_path,'model_latest.pth'))

            self.scheduler.step()

    def test_model(self,test_info_path,test_image_folder,save_path,model_path=None):
        if model_path != None:
            self.resum_load(model_path)

        opt_test = self.opt
        opt_test.image_path = test_image_folder
        opt_test.input_json = test_info_path

        prediction = {}
        test_dataset = ClsSteelDataset(opt_test, 'test')
        test_loader = DataLoader(test_dataset,shuffle=False,
                                batch_size=1, num_workers=4)

        for i, data in enumerate(test_loader):
            img = data[0].cuda()
            name = data[2][0]
            outputs = self.model(img)
            outputs = outputs.data.cpu().numpy().tolist()
            prediction[name] = outputs
            if i%100 == 0:
                print('%d / %d'%(i,len(test_dataset)))

        json.dump(prediction,open(save_path,'w'))



