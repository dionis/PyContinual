import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils
from tqdm import tqdm, trange
sys.path.append("./approaches/base/")
from cnn_base import Appr as ApprBase

class Appr(ApprBase):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        print('CNN LWF NCL')


        return



    def train(self,t,train,valid,num_train_steps,train_data,valid_data):
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            self.train_epoch(t,train,iter_bar)
            clock1=time.time()
            train_loss,train_acc,train_f1_macro=self.eval(t,train)
            clock2=time.time()
            # print('time: ',float((clock1-clock0)*30*25))

            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.args.train_batch_size*(clock1-clock0)/len(train),1000*self.args.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                patience=self.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=self.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<self.lr_min:
                        print()
                        break
                    patience=self.lr_patience
                    self.optimizer=self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model,best_model)

        # Update old
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights
        return

    def train_epoch(self,t,data,iter_bar):
        self.model.train()

        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            images,targets= batch

            targets_old_dic = None
            if t > 0:
                targets_old_dict = self.model_old(images)
            # Forward current model
            output_dict=self.model.forward(images)
            if 'dil' in self.args.scenario:
                output = output_dict['y']
                output = outputs[t]
                targets_old =  targets_old_dict['y']
            elif 'til' in self.args.scenario:
                outputs = output_dict['y']
                output = outputs[t]
                targets_old = targets_old_dict['y']

            loss = self.criterion_lwf(t,output,targets,targets_old)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,data,test=None,trained_task=None):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        target_list = []
        pred_list = []

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                images,targets= batch
                real_b=images.size(0)

                # Forward
                output_dict = self.model.forward(images)
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs = output_dict['y']
                    if self.args.ent_id: #detected id
                        output_d= self.ent_id_detection(trained_task,images,t=t)
                        output = output_d['output']
                    else:
                        output = outputs[t]



                loss=self.criterion_lwf(t,output,targets)
                _,pred=output.max(1)
                hits=(pred==targets).float()

                target_list.append(targets)
                pred_list.append(pred)
                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b

            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')

        return total_loss/total_num,total_acc/total_num,f1


    def criterion_lwf(self,t,outputs,targets, targets_old):
        # Regularization for all previous tasks
        if self.args.eval_only:
            return self.ce(output,targets)
        else:
            loss_dist = 0

            # for t_old in range(0,t):
            #     loss_dist+=utils.cross_entropy(outputs[t_old],targets_old[t_old],exp=1/self.T)
            #
            # # Cross entropy loss
            # loss_ce=self.ce(outputs[t],targets)
            for t_old in range(0, t):
                loss_dist += utils.cross_entropy(outputs, targets_old[t_old], exp=1 / self.T)

            # Cross entropy loss
            loss_ce = self.ce(outputs, targets)

            # We could add the weight decay regularization mentioned in the paper. However, this might not be fair/comparable to other approaches
            print("Loss evaluation")
            return loss_ce + self.lamb * loss_dist

