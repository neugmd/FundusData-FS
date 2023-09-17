""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset
import xlwt
import pdb
from time import time
import datetime

class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
            '_step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr1' + str(args.meta_lr1) + '_lr2' + str(args.meta_lr2) + \
            '_batch' + str(args.num_batch) + '_maxepoch' + str(args.max_epoch) + \
            '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + \
            '_stepsize' + str(args.step_size) + '_' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load meta-train set
        self.trainset = Dataset('train', self.args)
        self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way, self.args.shot + self.args.train_query)
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=8, pin_memory=True)

        # Load meta-val set
        self.valset = Dataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, 600, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)
        
        # Build meta-transfer learning model
        self.model = MtlLearner(self.args)

        # Set optimizer 
        self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
            {'params': self.model.base_learner_4.parameters(), 'lr': self.args.meta_lr2},\
            {'params': self.model.base_learner_43.parameters(), 'lr': self.args.meta_lr2},\
            {'params': self.model.base_learner_42.parameters(), 'lr': self.args.meta_lr2},\
            {'params': self.model.base_learner_41.parameters(), 'lr': self.args.meta_lr2}\
            ], lr=self.args.meta_lr1)
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)        
        
        # load pretrained model without FC classifier
        self.model_dict = self.model.state_dict()
        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']
        else:
            pre_base_dir = osp.join(log_base_dir, 'pre')
            pre_save_path1 = '_'.join([args.dataset, args.model_type])
            pre_save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
                str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
            pre_save_path = pre_base_dir + '/' + pre_save_path1 + '_' + pre_save_path2
            pretrained_dict = torch.load(osp.join(pre_save_path, 'max_acc.pth'))['params']
        pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
        print(pretrained_dict.keys())
        self.model_dict.update(pretrained_dict)
        self.model.load_state_dict(self.model_dict)    

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
    
    def soft_cross_entrophy(self, logits, target, num_class=64, gauss_u = 0.0, gauss_sd = 0.01):
        #pdb.set_trace()
        target_prob = F.one_hot(target, num_classes=num_class).float()
        gauss_p = torch.normal(gauss_u, gauss_sd, size=(target_prob.shape[0], 1))
        gauss_p = torch.abs(gauss_p)
        gauss_p_r = gauss_p.repeat(1, target_prob.shape[1])

        if torch.cuda.is_available():
            gauss_p_r = gauss_p_r.type(torch.cuda.FloatTensor)
        else:
            gauss_p_r = gauss_p_r.type(torch.Tensor)

        delta_p = torch.mul(target_prob, gauss_p_r)
        reversed_delta_p = torch.mul(torch.mul(torch.sub(target_prob, 1.0), -1.0), torch.div(gauss_p_r, (target_prob.shape[1]-1)))
        target_p = target_prob - delta_p + reversed_delta_p

        logits_log_likelyhood = -F.log_softmax(logits, dim=1)
        nllloss = torch.sum(torch.mul(logits_log_likelyhood, target_p))/target_p.shape[0]
        return nllloss  
    
    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))           

    def train(self):
        """The function for the meta-train phase."""

        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc4'] = []
        trlog['train_acc43'] = []
        trlog['train_acc42'] = []
        trlog['train_acc41'] = []
        trlog['val_acc4'] = []
        trlog['val_acc43'] = []
        trlog['val_acc42'] = []
        trlog['val_acc41'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Generate the labels for train set of the episodes
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)
        label_shot_train = torch.cat([label_shot, label_shot], dim=0) 

	# Save acc to excel sheet
        book = xlwt.Workbook(encoding='utf-8',style_compression=0)
        sheet = book.add_sheet('meta_train',cell_overwrite_ok=True)
        sheet.write(0,0,'epoch')
        sheet.write(0,1,'Loss/meta train')
        sheet.write(0,2,'val_acc4')
        sheet.write(0,3,'val_acc43')
        sheet.write(0,4,'val_acc42')
        sheet.write(0,5,'val_acc41')
        sheet.write(0,6,'val_acc3')
			
        # Start meta-train
        for epoch in range(1, self.args.max_epoch + 1):
            
            # Set the model to train mode
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc4_averager = Averager()
            train_acc43_averager = Averager()
            train_acc42_averager = Averager()
            train_acc41_averager = Averager()

            # Generate the labels for test set of the episodes during meta-train updates
            label = torch.arange(self.args.way).repeat(self.args.train_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
            label = torch.cat([label, label], dim=0)

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    #data, _ = [_.cuda() for _ in batch]
                    pos1, pos2 = [_.cuda() for _ in batch[0]]
                else:
                    data = batch[0]
                #pdb.set_trace() 
                p = self.args.shot * self.args.way
                pos1_shot, pos1_query = pos1[:p], pos1[p:]
                pos2_shot, pos2_query = pos2[:p], pos2[p:]

                data = torch.cat([pos1, pos2], dim=0)
                data_shot = torch.cat([pos1_shot, pos2_shot], dim=0) 
                data_query = torch.cat([pos1_query, pos2_query], dim=0) 
                #data_shot, data_query = data[:p], data[p:]
                # Output logits for model
                logits_4, logits_43, logits_42, logits_41,logits_3, _ = self.model((data_shot, label_shot_train, data_query))
                # Calculate meta-train loss
                #loss = F.cross_entropy(logits, label)
                loss_4 = self.soft_cross_entrophy(logits_4, label, logits_4.shape[1])
                #pdb.set_trace()
                #loss_4 = F.cross_entropy(logits_4, label) 
                loss_43 = F.cross_entropy(logits_43, label)
                loss_42 = F.cross_entropy(logits_42, label)
                loss_41 = F.cross_entropy(logits_41, label) 
                loss_3 = F.cross_entropy(logits_3, label) 

                loss = loss_4*0.7 + loss_43*0.1  + loss_42*0.1  + loss_41*0.1 + loss_3*0.05
                
                # Calculate meta-train accuracy
                acc_4 = count_acc(logits_4, label)
                acc_43 = count_acc(logits_43, label)
                acc_42 = count_acc(logits_42, label)
                acc_41 = count_acc(logits_41, label)
                
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc_4', float(acc_4), global_count)
                writer.add_scalar('data/acc_43', float(acc_43), global_count)
                writer.add_scalar('data/acc_42', float(acc_42), global_count)
                writer.add_scalar('data/acc_41', float(acc_41), global_count) 
                
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc4={:.4f} Acc43={:.4f} Acc42={:.4f} Acc41={:.4f}'.format(epoch, loss.item(), acc_4, acc_43, acc_42, acc_41))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc4_averager.add(acc_4)
                train_acc43_averager.add(acc_43)
                train_acc42_averager.add(acc_42)
                train_acc41_averager.add(acc_41)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update learning rate
            self.lr_scheduler.step()
            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc4_averager = train_acc4_averager.item()
            train_acc43_averager = train_acc43_averager.item()
            train_acc42_averager = train_acc42_averager.item()
            train_acc41_averager = train_acc41_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc4_averager = Averager()
            val_acc43_averager = Averager()
            val_acc42_averager = Averager()
            val_acc41_averager = Averager()
            val_acc3_averager = Averager()

            # Generate the labels for test set of the episodes during meta-val for this epoch
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
            #label = torch.cat([label, label], dim=0)

            # Print previous information
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run meta-validation
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                    #pos1, pos2 = [_.cuda() for _ in batch[0]]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                #pos1_shot, pos1_query = pos1[:p], pos1[p:]
                #pos2_shot, pos2_query = pos2[:p], pos2[p:]
                data_shot, data_query = data[:p], data[p:]
                #data = torch.cat([pos1, pos2], dim=0)
                #data_shot = torch.cat([pos1_shot, pos2_shot], dim=0) 
                #data_query = torch.cat([pos1_query, pos2_query], dim=0) 

                logits_4, logits_43, logits_42, logits_41, logits_3, _ = self.model((data_shot, label_shot, data_query))

                loss_4 = F.cross_entropy(logits_4, label)
                loss_43 = F.cross_entropy(logits_43, label)
                loss_42 = F.cross_entropy(logits_42, label)
                loss_41 = F.cross_entropy(logits_41, label)
                loss_3 = F.cross_entropy(logits_3, label)
                
                loss = loss_4*0.7 + loss_43*0.1  + loss_42*0.1  + loss_41*0.1 + loss_3*0.05

                acc_4 = count_acc(logits_4, label)
                acc_43 = count_acc(logits_43, label)
                acc_42 = count_acc(logits_42, label)
                acc_41 = count_acc(logits_41, label)
                acc_3 = count_acc(logits_3, label)

                val_loss_averager.add(loss.item())
                val_acc4_averager.add(acc_4)
                val_acc43_averager.add(acc_43)
                val_acc42_averager.add(acc_42)
                val_acc41_averager.add(acc_41)
                val_acc3_averager.add(acc_3)

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc4_averager = val_acc4_averager.item()
            val_acc43_averager = val_acc43_averager.item()
            val_acc42_averager = val_acc42_averager.item()
            val_acc41_averager = val_acc41_averager.item()
            val_acc3_averager = val_acc3_averager.item()
            
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc4', float(val_acc4_averager), epoch)
            writer.add_scalar('data/val_acc43', float(val_acc43_averager), epoch)
            writer.add_scalar('data/val_acc42', float(val_acc42_averager), epoch)
            writer.add_scalar('data/val_acc41', float(val_acc41_averager), epoch)
            writer.add_scalar('data/val_acc3', float(val_acc3_averager), epoch)            
            
            # Save result to excel
            sheet.write(epoch+1,0,epoch)
            sheet.write(epoch+1,1,loss.item())
            sheet.write(epoch+1,2,acc_4)
            sheet.write(epoch+1,3,acc_43)
            sheet.write(epoch+1,4,acc_42)
            sheet.write(epoch+1,5,acc_41)
            sheet.write(epoch+1,6,acc_3)
            book.save(r'./meta_acc_loss.xls')
            
            print('Epoch {}, Val, Loss={:.4f} Acc4={:.4f} Acc43={:.4f} Acc42={:.4f} Acc41={:.4f} Acc3={:.4f}'.format(epoch, val_loss_averager, val_acc4_averager, val_acc43_averager, val_acc42_averager, val_acc41_averager, val_acc3_averager))

            # Update best saved model
            val_list = [val_acc4_averager, val_acc43_averager, val_acc42_averager, val_acc41_averager]
            val_acc_averager = max(val_list)
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc4'].append(train_acc4_averager)
            trlog['train_acc43'].append(train_acc43_averager)
            trlog['train_acc42'].append(train_acc42_averager)
            trlog['train_acc41'].append(train_acc41_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc4'].append(val_acc4_averager)
            trlog['val_acc43'].append(val_acc43_averager)
            trlog['val_acc42'].append(val_acc42_averager)
            trlog['val_acc41'].append(val_acc41_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))

        writer.close()

    def eval(self):
        """The function for the meta-eval phase."""
        # Load the logs
        trlog = torch.load(osp.join(self.args.save_path, 'trlog'))
        N_test = 100
        # Load meta-test set
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, N_test, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

        # Set test accuracy recorder
        test_acc_record = np.zeros((N_test,))
        test_acc_record43 = np.zeros((N_test,))
        test_acc_record42 = np.zeros((N_test,))
        test_acc_record41 = np.zeros((N_test,))
        test_acc_record_ave = np.zeros((N_test,))
        test_acc_record_weight = np.zeros((N_test,))
        test_acc_record_vote = np.zeros((N_test,))
        test_acc_record_merge = np.zeros((N_test,))

        # Load model for meta-test phase
        if self.args.eval_weights is not None:
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])
        # Set model to eval mode
        self.model.eval()

        # Set accuracy averager
        ave_acc = Averager()
        ave_acc43 = Averager()
        ave_acc42 = Averager()
        ave_acc41 = Averager()
        ave_acc_ave = Averager()
        ave_acc_weight = Averager()
        ave_acc_vote = Averager()
        ave_acc_merge = Averager()

        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)
        '''
        # Save acc to excel sheet
        book2 = xlwt.Workbook(encoding='utf-8',style_compression=0)
        sheet_ = book2.add_sheet('meta_test',cell_overwrite_ok=True)
        sheet_.write(0,0,'true label')
        sheet_.write(0,1,'p1')
        sheet_.write(0,2,'p2')
        sheet_.write(0,3,'p3')
        sheet_.write(0,4,'p4')
        sheet_.write(0,5,'p5')
        
        sheet_.write(0,6,'p1')
        sheet_.write(0,7,'p2')
        sheet_.write(0,8,'p3')
        sheet_.write(0,9,'p4')
        sheet_.write(0,10,'p5') 
        
        sheet_.write(0,11,'p1')
        sheet_.write(0,12,'p2')
        sheet_.write(0,13,'p3')
        sheet_.write(0,14,'p4')
        sheet_.write(0,15,'p5')
        
        sheet_.write(0,16,'p1')
        sheet_.write(0,17,'p2')
        sheet_.write(0,18,'p3')
        sheet_.write(0,19,'p4')
        sheet_.write(0,20,'p5')
        '''
        # Start meta-test
        begin = time()
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]
            # logits = self.model((data_shot, label_shot, data_query))
            logits_4, logits_43, logits_42, logits_41, logits_3, logits_merge = self.model((data_shot, label_shot, data_query))
            
            logits_ave = 0.25*(logits_4 + logits_43 + logits_42 + logits_41)
            
            #logits_weight = 0.27 * logits_4 + 0.245 * logits_43 + 0.26 * logits_42 + 0.225 * logits_41
            logits_weight = 0.7 * logits_4 + 0.15 * logits_43 + 0.1 * logits_42 + 0.05 * logits_41

            # acc = count_acc(logits, label)
            acc4 = count_acc(logits_4, label)
            acc43 = count_acc(logits_43, label)
            acc42 = count_acc(logits_42, label)
            acc41 = count_acc(logits_41, label)
            acc_ave = count_acc(logits_ave, label)
            acc_weight = count_acc(logits_weight, label)
            acc_merge = count_acc(logits_merge, label)            
            
            ave_acc.add(acc4)
            ave_acc43.add(acc43)
            ave_acc42.add(acc42)
            ave_acc41.add(acc41)
            ave_acc_ave.add(acc_ave)
            ave_acc_weight.add(acc_weight)
            ave_acc_merge.add(acc_merge)
            logits_numpy_4 = F.softmax(logits_4, dim=1).data.cpu().numpy()
            logits_numpy_43 = F.softmax(logits_43, dim=1).data.cpu().numpy()
            logits_numpy_42 = F.softmax(logits_42, dim=1).data.cpu().numpy()
            logits_numpy_41 = F.softmax(logits_41, dim=1).data.cpu().numpy()
            
            pred_4 = F.softmax(logits_4, dim=1).argmax(dim=1).data.cpu().numpy()
            pred_43 = F.softmax(logits_43, dim=1).argmax(dim=1).data.cpu().numpy()
            pred_42 = F.softmax(logits_42, dim=1).argmax(dim=1).data.cpu().numpy()
            pred_41 = F.softmax(logits_41, dim=1).argmax(dim=1).data.cpu().numpy()
           
            # voting
            #pdb.set_trace()
            pre_voting = []
            for ii in range(logits_4.shape[0]):
                pre_list = [pred_4[ii], pred_43[ii], pred_42[ii], pred_41[ii]]
                pre_frequency = [pre_list.count(0),pre_list.count(1),pre_list.count(2),pre_list.count(3)]
                pre_voting.append(pre_frequency.index(max(pre_frequency)))               
            # true number
            #pdb.set_trace()
            pre_voting_arr = np.asarray(pre_voting)
            true_num = np.sum(pre_voting_arr == label.data.cpu().numpy())
            acc_voting = true_num/logits_4.shape[0]          
            ave_acc_vote.add(acc_voting)
            '''
            label_numpy = label.data.cpu().numpy()
            
            # Save result to excel
            pro_len = np.size(logits_numpy_4,0)
            for t in range(pro_len):
                num = t + (i - 1) * pro_len
                sheet_.write(num,0,str(label_numpy[t]))
                
                sheet_.write(num,1,str(logits_numpy_4[t][0]))
                sheet_.write(num,2,str(logits_numpy_4[t][1]))
                sheet_.write(num,3,str(logits_numpy_4[t][2]))
                sheet_.write(num,4,str(logits_numpy_4[t][3]))
                sheet_.write(num,5,str(logits_numpy_4[t][4]))
                
                sheet_.write(num,6,str(logits_numpy_43[t][0]))
                sheet_.write(num,7,str(logits_numpy_43[t][1]))
                sheet_.write(num,8,str(logits_numpy_43[t][2]))
                sheet_.write(num,9,str(logits_numpy_43[t][3]))
                sheet_.write(num,10,str(logits_numpy_43[t][4]))
                
                sheet_.write(num,11,str(logits_numpy_42[t][0]))
                sheet_.write(num,12,str(logits_numpy_42[t][1]))
                sheet_.write(num,13,str(logits_numpy_42[t][2]))
                sheet_.write(num,14,str(logits_numpy_42[t][3]))
                sheet_.write(num,15,str(logits_numpy_42[t][4]))
                
                sheet_.write(num,16,str(logits_numpy_41[t][0]))
                sheet_.write(num,17,str(logits_numpy_41[t][1]))
                sheet_.write(num,18,str(logits_numpy_41[t][2]))
                sheet_.write(num,19,str(logits_numpy_41[t][3]))
                sheet_.write(num,20,str(logits_numpy_41[t][4]))
                              
                book2.save(r'./meta_test_acc.xls')
            '''   
            test_acc_record[i-1] = acc4
            test_acc_record43[i-1] = acc43
            test_acc_record42[i-1] = acc42
            test_acc_record41[i-1] = acc41
            #test_acc_record42[i-1] = acc42
            test_acc_record_ave[i-1] = acc_ave
            test_acc_record_weight[i-1] = acc_weight
            test_acc_record_vote[i-1] = acc_voting
            test_acc_record_merge[i-1] = acc_merge
            '''
            if i % 100 == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
            '''
        date_time = datetime.timedelta(seconds = int(time() - begin))
        date_time_epoch = datetime.timedelta(seconds = (time() - begin)/100)
        print("Took " + str(date_time))
        print("Took " + str(date_time_epoch) + " per task.")

        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        m43, pm43 = compute_confidence_interval(test_acc_record43)
        m42, pm42 = compute_confidence_interval(test_acc_record42)
        m41, pm41 = compute_confidence_interval(test_acc_record41)
        m_ave, pm_ave = compute_confidence_interval(test_acc_record_ave)
        m_weight, pm_weight = compute_confidence_interval(test_acc_record_weight)
        m_vote, pm_vote = compute_confidence_interval(test_acc_record_vote)
        m_merge, pm_merge = compute_confidence_interval(test_acc_record_merge)
        
        print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
        print('Test Acc43 {:.4f} + {:.4f}'.format(m43, pm43))
        print('Test Acc42 {:.4f} + {:.4f}'.format(m42, pm42))
        print('Test Acc41 {:.4f} + {:.4f}'.format(m41, pm41))
        print('Test Acc_ave {:.4f} + {:.4f}'.format(m_ave, pm_ave))
        print('Test Acc_weight {:.4f} + {:.4f}'.format(m_weight, pm_weight))
        print('Test Acc_vote {:.4f} + {:.4f}'.format(m_vote, pm_vote))
        print('Test Acc_merge {:.4f} + {:.4f}'.format(m_merge, pm_merge))
        
