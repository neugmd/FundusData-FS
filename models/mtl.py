""" Model for meta-transfer learning. """
import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_mtl import ResNetMtl
import pdb
import math

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars
        
class MetaLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.z_dim, 1]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0] # 4*16=64 (16*16+16*4+16+4)
        fc1_w_softmax = F.softmax(fc1_w, dim=0)
        #pdb.set_trace()
        #net = fc1_w_softmax[0]*input_x[:,0:3] + fc1_w_softmax[1]*input_x[:,3:6] + fc1_w_softmax[2]*input_x[:,6:9] + fc1_w_softmax[3]*input_x[:,9:]
        #net = fc1_w_softmax[0]*input_x[:,0:2] + fc1_w_softmax[1]*input_x[:,2:4] + fc1_w_softmax[2]*input_x[:,4:6] + fc1_w_softmax[3]*input_x[:,6:]
        #net = fc1_w_softmax[0]*input_x[:,0:4] + fc1_w_softmax[1]*input_x[:,4:8] + fc1_w_softmax[2]*input_x[:,8:12] + fc1_w_softmax[3]*input_x[:,12:]
        net = fc1_w_softmax[0]*input_x[:,0:6] + fc1_w_softmax[1]*input_x[:,6:12] + fc1_w_softmax[2]*input_x[:,12:18] + fc1_w_softmax[3]*input_x[:,18:]
        return net

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        z_dim_4 = 512
        z_dim_43 = 512
        z_dim_42 = 512
        z_dim_41 = 512
        z_dim_3 = 256
        z_dim = 4
        self.base_learner_4 = BaseLearner(args, z_dim_4)
        self.base_learner_43 = BaseLearner(args, z_dim_43)
        self.base_learner_42 = BaseLearner(args, z_dim_42)
        self.base_learner_41 = BaseLearner(args, z_dim_41)
        self.base_learner_3 = BaseLearner(args, z_dim_3)
        self.meta_learner = MetaLearner(args, z_dim)

        if self.mode == 'meta':
            self.encoder = ResNetMtl()  
        else:
            self.encoder = ResNetMtl(mtl=False)  
            self.pre_fc_4 = nn.Linear(512, num_cls)
            self.pre_fc_43 = nn.Linear(512, num_cls)
            self.pre_fc_42 = nn.Linear(512, num_cls)
            self.pre_fc_41 = nn.Linear(512, num_cls)
            self.pre_fc_3 = nn.Linear(256, num_cls)
            self.pre_fc_2 = nn.Linear(128, num_cls)
            self.pre_fc_1 = nn.Linear(64, num_cls)

    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode=='pre':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):      
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        out4, out43, out42, out41, out3, out2, out1 = self.encoder(inp)
        return self.pre_fc_4(out4), self.pre_fc_43(out43), self.pre_fc_42(out42), self.pre_fc_41(out41), self.pre_fc_3(out3), self.pre_fc_2(out2), self.pre_fc_1(out1), out4

    def meta_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query_4, embedding_query_43, embedding_query_42, embedding_query_41, embedding_query_3, _2, _1  = self.encoder(data_query)
        embedding_shot_4, embedding_shot_43, embedding_shot_42, embedding_shot_41, embedding_shot_3, _2, _1 = self.encoder(data_shot)
        logits_4 = self.base_learner_4(embedding_shot_4)
        logits_43 = self.base_learner_43(embedding_shot_43)
        logits_42 = self.base_learner_42(embedding_shot_42)
        logits_41 = self.base_learner_41(embedding_shot_41)
        logits_3 = self.base_learner_3(embedding_shot_3)
        
        # fast_weights for meta-learner
        #pdb.set_trace() 
        logits_matrix = torch.cat((logits_4,logits_43,logits_42,logits_41), 1)
        logits_merge = self.meta_learner(logits_matrix)
        #pdb.set_trace()
        loss = F.cross_entropy(logits_merge, label_shot)
        grad = torch.autograd.grad(loss, self.meta_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.meta_learner.parameters())))
 
        loss_4 = F.cross_entropy(logits_4, label_shot)
        loss_43 = F.cross_entropy(logits_43, label_shot)
        loss_42 = F.cross_entropy(logits_42, label_shot)
        loss_41 = F.cross_entropy(logits_41, label_shot)
        loss_3 = F.cross_entropy(logits_3, label_shot)
        
        grad_4 = torch.autograd.grad(loss_4, self.base_learner_4.parameters())
        grad_43 = torch.autograd.grad(loss_43, self.base_learner_43.parameters())
        grad_42 = torch.autograd.grad(loss_42, self.base_learner_42.parameters())
        grad_41 = torch.autograd.grad(loss_41, self.base_learner_41.parameters())
        grad_3 = torch.autograd.grad(loss_3, self.base_learner_3.parameters())

        fast_weights_4 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_4, self.base_learner_4.parameters())))
        fast_weights_43 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_43, self.base_learner_43.parameters())))
        fast_weights_42 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_42, self.base_learner_42.parameters())))
        fast_weights_41 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_41, self.base_learner_41.parameters())))
        fast_weights_3 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_3, self.base_learner_3.parameters())))

        logits_q_4 = self.base_learner_4(embedding_query_4, fast_weights_4)
        logits_q_43 = self.base_learner_43(embedding_query_43, fast_weights_43)
        logits_q_42 = self.base_learner_42(embedding_query_42, fast_weights_42)
        logits_q_41 = self.base_learner_41(embedding_query_41, fast_weights_41)
        logits_q_3 = self.base_learner_3(embedding_query_3, fast_weights_3)

        for _ in range(1, self.update_step):
            logits_4 = self.base_learner_4(embedding_shot_4, fast_weights_4)
            logits_43 = self.base_learner_43(embedding_shot_43, fast_weights_43)
            logits_42 = self.base_learner_42(embedding_shot_42, fast_weights_42)
            logits_41 = self.base_learner_41(embedding_shot_41, fast_weights_41)
            logits_3 = self.base_learner_3(embedding_shot_3, fast_weights_3)

            loss_4 = F.cross_entropy(logits_4, label_shot)
            loss_43 = F.cross_entropy(logits_43, label_shot)
            loss_42 = F.cross_entropy(logits_42, label_shot)
            loss_41 = F.cross_entropy(logits_41, label_shot)
            loss_3 = F.cross_entropy(logits_3, label_shot)

            grad_4 = torch.autograd.grad(loss_4, fast_weights_4)
            grad_43 = torch.autograd.grad(loss_43, fast_weights_43)
            grad_42 = torch.autograd.grad(loss_42, fast_weights_42)
            grad_41 = torch.autograd.grad(loss_41, fast_weights_41)
            grad_3 = torch.autograd.grad(loss_3, fast_weights_3)

            fast_weights_4 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_4, fast_weights_4)))
            fast_weights_43 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_43, fast_weights_43)))
            fast_weights_42 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_42, fast_weights_42)))
            fast_weights_41 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_41, fast_weights_41)))
            fast_weights_3 = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_3, fast_weights_3)))

            logits_q_4 = self.base_learner_4(embedding_query_4, fast_weights_4)
            logits_q_43 = self.base_learner_43(embedding_query_43, fast_weights_43) 
            logits_q_42 = self.base_learner_42(embedding_query_42, fast_weights_42) 
            logits_q_41 = self.base_learner_41(embedding_query_41, fast_weights_41)
            logits_q_3 = self.base_learner_3(embedding_query_3, fast_weights_3) 

            # meta-learner
            # output from support set
            logits_matrix = torch.cat((logits_4,logits_43,logits_42,logits_41), 1)       # 4*16 =4*4*4     
            logits_merge = self.meta_learner(logits_matrix, fast_weights)   # 4*4
            loss = F.cross_entropy(logits_merge, label_shot)    
            grad = torch.autograd.grad(loss, self.meta_learner.parameters())
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.meta_learner.parameters())))
            # size: 60*16 (4 way * 15 query set)
            logits_matrix_query = torch.cat((logits_q_4,logits_q_43,logits_q_42,logits_q_41), 1)
            
            logits_q_merge = self.meta_learner(logits_matrix_query, fast_weights)
            
        return logits_q_4, logits_q_43, logits_q_42, logits_q_41, logits_q_3, logits_q_merge

    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query_4, embedding_query_43, embedding_query_42, embedding_query_41, embedding_query_3, _2, _1 = self.encoder(data_query)
        embedding_shot_4, embedding_shot_43, embedding_shot_42, embedding_shot_41, embedding_shot_3, _2, _1 = self.encoder(data_shot)
        logits_4 = self.base_learner_4(embedding_shot_4)
        logits_43 = self.base_learner_43(embedding_shot_43)
        logits_42 = self.base_learner_42(embedding_shot_42)
        logits_41 = self.base_learner_41(embedding_shot_41)
        logits_3 = self.base_learner_3(embedding_shot_3)

        loss_4 = F.cross_entropy(logits_4, label_shot)
        loss_43 = F.cross_entropy(logits_43, label_shot)
        loss_42 = F.cross_entropy(logits_42, label_shot)
        loss_41 = F.cross_entropy(logits_41, label_shot)
        loss_3 = F.cross_entropy(logits_3, label_shot)

        grad_4 = torch.autograd.grad(loss_4, self.base_learner_4.parameters())
        grad_43 = torch.autograd.grad(loss_43, self.base_learner_43.parameters())
        grad_42 = torch.autograd.grad(loss_42, self.base_learner_42.parameters())
        grad_41 = torch.autograd.grad(loss_41, self.base_learner_41.parameters())
        grad_3 = torch.autograd.grad(loss_3, self.base_learner_3.parameters())

        fast_weights_4 = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad_4, self.base_learner_4.parameters())))
        fast_weights_43 = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad_43, self.base_learner_43.parameters())))
        fast_weights_42 = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad_42, self.base_learner_42.parameters())))
        fast_weights_41 = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad_41, self.base_learner_41.parameters())))
        fast_weights_3 = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad_3, self.base_learner_3.parameters())))

        logits_q_4 = self.base_learner_4(embedding_query_4, fast_weights_4)
        logits_q_43 = self.base_learner_43(embedding_query_43, fast_weights_43)
        logits_q_42 = self.base_learner_42(embedding_query_42, fast_weights_42)
        logits_q_41 = self.base_learner_41(embedding_query_41, fast_weights_41)
        logits_q_3 = self.base_learner_3(embedding_query_3, fast_weights_3)

        for _ in range(1, 100):
            logits_4 = self.base_learner_4(embedding_shot_4, fast_weights_4)
            logits_43 = self.base_learner_43(embedding_shot_43, fast_weights_43)
            logits_42 = self.base_learner_42(embedding_shot_42, fast_weights_42)
            logits_41 = self.base_learner_41(embedding_shot_41, fast_weights_41)
            logits_3 = self.base_learner_3(embedding_shot_3, fast_weights_3)

            loss_4 = F.cross_entropy(logits_4, label_shot)
            loss_43 = F.cross_entropy(logits_43, label_shot)
            loss_42 = F.cross_entropy(logits_42, label_shot)
            loss_41 = F.cross_entropy(logits_41, label_shot)
            loss_3 = F.cross_entropy(logits_3, label_shot)

            grad_4 = torch.autograd.grad(loss_4, fast_weights_4)
            grad_43 = torch.autograd.grad(loss_43, fast_weights_43)
            grad_42 = torch.autograd.grad(loss_42, fast_weights_42)
            grad_41 = torch.autograd.grad(loss_41, fast_weights_41)
            grad_3 = torch.autograd.grad(loss_3, fast_weights_3)

            fast_weights_4 = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad_4, fast_weights_4)))
            fast_weights_43 = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad_43, fast_weights_43)))
            fast_weights_42 = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad_42, fast_weights_42)))
            fast_weights_41 = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad_41, fast_weights_41)))
            fast_weights_3 = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad_3, fast_weights_3)))

            logits_q_4 = self.base_learner_4(embedding_query_4, fast_weights_4)
            logits_q_43 = self.base_learner_43(embedding_query_43, fast_weights_43) 
            logits_q_42 = self.base_learner_42(embedding_query_42, fast_weights_42) 
            logits_q_41 = self.base_learner_41(embedding_query_41, fast_weights_41)
            logits_q_3 = self.base_learner_3(embedding_query_3, fast_weights_3)

            
        return logits_q_4, logits_q_43, logits_q_42, logits_q_41, logits_q_3
        
