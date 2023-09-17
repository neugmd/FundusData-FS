""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, ensure_path
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset
import xlwt
from losses import NTXentLoss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PreTrainer(object):
    """The class that contains the code for the pretrain phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
            str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args
        
        self.device = device

        # Load pretrain set
        self.trainset = Dataset('train', self.args, train_aug=True)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

        # Load meta-val set
        self.valset = Dataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, 600, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)

        # Set pretrain class number 
        num_class_pretrain = self.trainset.num_class
        
        # loss definition
        self.ntxent = NTXentLoss(self.device, self.args.pre_batch_size, temperature=0.5, use_cosine_similarity=True)
        
        # Build pretrain model
        self.model = MtlLearner(self.args, mode='pre', num_cls=num_class_pretrain)

        # Set optimizer 
        self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.pre_lr}, \
            {'params': self.model.pre_fc_4.parameters(), 'lr': self.args.pre_lr},\
            {'params': self.model.pre_fc_43.parameters(), 'lr': self.args.pre_lr},\
            {'params': self.model.pre_fc_42.parameters(), 'lr': self.args.pre_lr},\
            {'params': self.model.pre_fc_41.parameters(), 'lr': self.args.pre_lr},\
            {'params': self.model.pre_fc_3.parameters(), 'lr': self.args.pre_lr},\
            {'params': self.model.pre_fc_2.parameters(), 'lr': self.args.pre_lr},\
            {'params': self.model.pre_fc_1.parameters(), 'lr': self.args.pre_lr},\
            ], momentum=self.args.pre_custom_momentum, nesterov=True, weight_decay=self.args.pre_custom_weight_decay)
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, \
            gamma=self.args.pre_gamma)        
        
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
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))
        
    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc4'] = []
        trlog['train_acc43'] = []
        trlog['train_acc42'] = []
        trlog['train_acc41'] = []
        trlog['train_acc3'] = []
        trlog['train_acc2'] = []
        trlog['train_acc1'] = []
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
        
        '''
		    # Save acc to excel sheet
        book = xlwt.Workbook(encoding='utf-8',style_compression=0)
        sheet = book.add_sheet('pre',cell_overwrite_ok=True)
        sheet.write(0,0,'epoch')
        sheet.write(0,1,'Loss/pre')
        sheet.write(0,2,'val_acc4')
        sheet.write(0,3,'val_acc43')
        sheet.write(0,4,'val_acc42')
        sheet.write(0,5,'val_acc41')
        '''
			
        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
           
            # Set the model to train mode
            self.model.train()
            self.model.mode = 'pre'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc4_averager = Averager()
            train_acc43_averager = Averager()
            train_acc42_averager = Averager()
            train_acc41_averager = Averager()
            train_acc3_averager = Averager()
            train_acc2_averager = Averager()
            train_acc1_averager = Averager()
                
            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    # data, _ = [_.cuda() for _ in batch]
                    pos1, pos2 = [_.cuda() for _ in batch[0]]
                else:
                    data = batch[0]
                label = batch[1]
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)

                data = torch.cat([pos1, pos2], dim=0)
                label = torch.cat([label, label], dim=0)

                # Output logits for model
                logits_4_1, logits_43_1, logits_42_1, logits_41_1, logits_3_1, logits_2_1, logits_1_1, feat_4_1 = self.model(pos1)
                logits_4_2, logits_43_2, logits_42_2, logits_41_2, logits_3_2, logits_2_2, logits_1_2, feat_4_2 = self.model(pos2)

                logits_4 = torch.cat([logits_4_1, logits_4_2], dim=0)
                logits_43 = torch.cat([logits_43_1, logits_43_2], dim=0)
                logits_42 = torch.cat([logits_42_1, logits_42_2], dim=0)
                logits_41 = torch.cat([logits_41_2, logits_41_1], dim=0)
                logits_3 = torch.cat([logits_3_1, logits_3_2], dim=0)
                logits_2 = torch.cat([logits_2_1, logits_2_2], dim=0)
                logits_1 = torch.cat([logits_1_1, logits_1_2], dim=0)

                loss_ss = self.ntxent(feat_4_1, feat_4_2)

                # Calculate train loss
                #loss = F.cross_entropy(logits, label)
                loss_4 = self.soft_cross_entrophy(logits_4, label, logits_4.shape[1])
                loss_43 = F.cross_entropy(logits_43, label)
                loss_42 = F.cross_entropy(logits_42, label)
                loss_41 = F.cross_entropy(logits_41, label)
                loss_3 = F.cross_entropy(logits_3, label)
                loss_2 = F.cross_entropy(logits_2, label) 
                loss_1 = F.cross_entropy(logits_1, label)              

                #loss = loss_4*0.8 + loss_43*0.2  + loss_42*0.1  + loss_41*0.1 + loss_3*0.05 + loss_2*0.05 + loss_1*0.02
                loss = loss_4*0.7 + loss_43*0.1  + loss_42*0.1  + loss_41*0.1 + loss_3*0.05 + loss_ss
                # Calculate train accuracy
                acc_4 = count_acc(logits_4, label)
                acc_43 = count_acc(logits_43, label)
                acc_42 = count_acc(logits_42, label)
                acc_41 = count_acc(logits_41, label)
                acc_3 = count_acc(logits_3, label)
                acc_2 = count_acc(logits_2, label)
                acc_1 = count_acc(logits_1, label)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc_4', float(acc_4), global_count)
                writer.add_scalar('data/acc_43', float(acc_43), global_count)
                writer.add_scalar('data/acc_42', float(acc_42), global_count)
                writer.add_scalar('data/acc_41', float(acc_41), global_count)
                writer.add_scalar('data/acc_3', float(acc_3), global_count)
                writer.add_scalar('data/acc_2', float(acc_2), global_count)
                writer.add_scalar('data/acc_1', float(acc_1), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc4={:.4f} Acc43={:.4f} Acc42={:.4f} Acc41={:.4f} Acc3={:.4f} Acc2={:.4f} Acc1={:.4f}'.format(epoch, loss.item(), acc_4, acc_43, acc_42, acc_41, acc_3, acc_2, acc_1))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc4_averager.add(acc_4)
                train_acc43_averager.add(acc_43)
                train_acc42_averager.add(acc_42)
                train_acc41_averager.add(acc_41)
                train_acc3_averager.add(acc_3)
                train_acc2_averager.add(acc_2)
                train_acc1_averager.add(acc_1)

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
            train_acc3_averager = train_acc3_averager.item()
            train_acc2_averager = train_acc2_averager.item()
            train_acc1_averager = train_acc1_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()
            self.model.mode = 'preval'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc4_averager = Averager()
            val_acc43_averager = Averager()
            val_acc42_averager = Averager()
            val_acc41_averager = Averager()

            # Generate the labels for test 
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
              
            # Print previous information  
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run meta-validation
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                logits_4, logits_43, logits_42, logits_41,logits_3 = self.model((data_shot, label_shot, data_query))
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
                 
                val_loss_averager.add(loss.item())
                val_acc4_averager.add(acc_4)
                val_acc43_averager.add(acc_43)
                val_acc42_averager.add(acc_42)
                val_acc41_averager.add(acc_41)

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc4_averager = val_acc4_averager.item()
            val_acc43_averager = val_acc43_averager.item()
            val_acc42_averager = val_acc42_averager.item()
            val_acc41_averager = val_acc41_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc4', float(val_acc4_averager), epoch)
            writer.add_scalar('data/val_acc43', float(val_acc43_averager), epoch)
            writer.add_scalar('data/val_acc42', float(val_acc42_averager), epoch)
            writer.add_scalar('data/val_acc41', float(val_acc41_averager), epoch)   
            '''    
            # Save result to excel
            sheet.write(epoch+1,0,epoch)
            sheet.write(epoch+1,1,loss.item())
            sheet.write(epoch+1,2,acc_4)
            sheet.write(epoch+1,3,acc_43)
            sheet.write(epoch+1,4,acc_42)
            sheet.write(epoch+1,5,acc_41)
            book.save(r'./Val_acc_loss.xls')
            '''
            # Print loss and accuracy for this epoch
            print('Epoch {}, Val, Loss={:.4f} Acc12={:.4f} Acc11={:.4f} Acc10={:.4f} Acc9={:.4f}'.format(epoch, val_loss_averager, val_acc4_averager, val_acc43_averager, val_acc42_averager, val_acc41_averager))

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
            trlog['train_acc3'].append(train_acc3_averager)
            trlog['train_acc2'].append(train_acc2_averager)
            trlog['train_acc1'].append(train_acc1_averager)
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
        
