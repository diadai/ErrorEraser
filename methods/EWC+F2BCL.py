import torch
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from copy import deepcopy
from nflows.flows.base import Flow
from nflows.transforms.permutations import RandomPermutation, ReversePermutation
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.nn.nets.myresnet import ResidualNet
from nflows.distributions.normal import StandardNormal
from torch.nn import functional as F
from utils.utils import myitem
from torch import optim
from utils.meter import Meter
import copy
from trainer import eval, loss_picker, optimizer_picker
from boundary_models import init_params as w_init
import time
eps = 1e-30
import os

class Manager(torch.nn.Module):
    def __init__(self,
                 arch,
                 taskcla,
                 args):
        super(Manager, self).__init__()
        self.arch = arch
        self.current_task = 0
        self.args = args
        self.class_incremental = self.args.class_incremental
        self.lr_patience = self.args.lr_patience
        self.lr_factor = self.args.lr_factor
        self.lr_min = self.args.lr_min
        self.fisher = {}
        self.params = {}
        self.lamb = 10000
        # self.lamb = 500
        self.ce = torch.nn.CrossEntropyLoss()
        # self.ce = torch.nn.NLLLoss()
        self.classify_criterion_noreduce = torch.nn.NLLLoss(reduction='none')
        # self.available_labels_past = [[0, 1], [2, 3],[4, 5], [6, 7], [8, 9]]
        self.available_labels_past = [[j + i * 2 for j in range(2)] for i in range(args.num_tasks)]
        if not self.class_incremental:
            self.num_all_class = sum(n_class for _, n_class in taskcla)
            self.all_classes = list(range(self.num_all_class))
            self.num_each_class = int((self.num_all_class / args.num_tasks))
            self.not_current_task_classes = self.all_classes


        self.softmax = torch.nn.Softmax(dim=1)
        self.xa_shape = [2048]
        self.num_classes = 10
        self.flow = self.get_1d_nflow_model(feature_dim=int(np.prod(self.xa_shape)), hidden_feature=512,
                                            context_feature=self.num_classes,
                                            num_layers=4)
        self.flow_optimizer = optim.Adam(
            self.flow.parameters(), lr=self.args.flow_lr,
            weight_decay=self.args.weight_decay, betas=(self.args.beta1, self.args.beta2),
        )
        parameters_fb = [a[1] for a in filter(lambda x: 'fc2' in x[0], self.named_parameters())]
        self.classifier_fb_optimizer = optim.Adam(
            parameters_fb, lr=self.args.lr, weight_decay=self.args.weight_decay,
            betas=(self.args.beta1, self.args.beta2),
        )
        self.opt = torch.optim.Adam(self.arch.parameters(), lr=0.0001, weight_decay=5e-4)

    def parameters(self):
        for param in self.arch.parameters():
            yield param
        if True:
            for param in self.flow.parameters():
                yield param

    def named_parameters(self):
        for name, param in self.arch.named_parameters():
            yield 'classifier.'+name, param
        if True:
            for name, param in self.flow.named_parameters():
                yield 'flow.'+name, param
    def Multi_Class_Cross_Entropy(self, logits, labels, task):
        if self.class_incremental == False and task>0:
            labels = labels+(task*self.num_each_class)
        ce = torch.nn.CrossEntropyLoss()
        loss = ce(logits, labels)
        return loss

    def calculate_fisher(self, train_dataloader, task):
        self.train()
        # self.arch.set_eval_mode(False)

        fisher = {}
        params = {}
        for n, p in self.named_parameters():
            fisher[n] = 0 * p.data
            params[n] = 0 * p.data

        for features, labels in train_dataloader:
            features, labels = features.to(self.args.device), labels.to(self.args.device)
            self.zero_grad()
            logits,_ = self.arch(features, task, middle_feature=False)
            # loss = self.ce(logits, labels)
            loss = self.Multi_Class_Cross_Entropy(logits, labels, task)
            loss.backward()

            for n, p in self.named_parameters():
                if p.grad is not None:
                    pg = p.grad.data.clone().pow(2)
                    fisher[n] += pg

        for n, p in self.named_parameters():
            if p.grad is not None:
                pd = p.data.clone()
                params[n] = pd

        self.zero_grad()
        return fisher, params

    def train_with_eval(self, train_dataloader, val_dataloader, task):
        lr = self.args.lr
        patience = 10
        best_loss = np.inf
        best_model = deepcopy(self.state_dict())
        cls_meter = Meter()
        self.available_labels = self.available_labels_past[task]
        for epoch in trange(self.args.epochs, leave=False):
            # torch.cuda.empty_cache()

            last_classifier = None
            last_flow = None

            if task !=0:
                last_flow = copy.deepcopy(self.flow)
                last_flow.cuda()
                last_flow.eval()

            for features, labels in train_dataloader:
                features, labels = features.to(self.args.device), labels.to(self.args.device)

                self.eval()
                self.flow.train()
                flow_result = self.train_a_batch_flow(features, labels, last_flow, task)
                cls_meter._update(flow_result, batch_size=self.args.train_batch_size)
                flow = self.flow
                flow.eval()

                self.train()
                # self.arch.set_eval_mode(False)
                # self.zero_grad()

                with torch.no_grad():
                    logits, _ = self.arch(features, task, middle_feature=False)

                    xa = self.arch.forward_to_xa(features)
                    xa = xa.reshape(xa.shape[0], -1)

                    y_one_hot = F.one_hot(labels, num_classes=self.num_classes).float() # 将标签转换为one-hot编码形式
                    log_prob, xa_u = flow.log_prob_and_noise(xa, y_one_hot)  # 计算在给定上下文中输入的对数概率和生成的噪声，用于正规化流模型
                    log_prob = log_prob.detach()
                    xa_u = xa_u.detach()
                    prob_mean = torch.exp(log_prob / xa.shape[1]).mean() + eps   # 计算平均概率，用于后续概率比较

                    flow_xa, label, _ = self.sample_from_flow(flow, self.available_labels, self.args.train_batch_size)   # 从流模型中采样，生成新的样本 # 从NF采样
                    if self.class_incremental and task > 0:
                        label = label - (task * self.args.each_classes)
                    flow_xa_prob = self.probability_in_current_task(xa_u, labels, prob_mean, flow_xa, label)   # 计算生成样本在局部数据分布上的概率
                    flow_xa_prob = flow_xa_prob.detach()
                    flow_xa_prob_mean = flow_xa_prob.mean()

                flow_xa = flow_xa.reshape(flow_xa.shape[0], *self.xa_shape)  # 调整生成样本的形状以匹配网络输入要求
                _, softmax_output_flow = self.arch(flow_xa, task, middle_feature=True)  # 对生成的样本进行分类预测
                c_loss_flow_generate = (self.classify_criterion_noreduce(torch.log(softmax_output_flow + eps),
                                                                         torch.Tensor(
                                                                                label).long().cuda()) * flow_xa_prob).mean() # 计算分类损失
                k_loss_flow_explore_forget = (1 - self.args.flow_explore_theta) * prob_mean + self.args.flow_explore_theta  # 结合探索和遗忘权重计算总损失

                c_loss_flow = (c_loss_flow_generate*k_loss_flow_explore_forget)*self.args.k_loss_flow  # 计算最终的流损失
                # 优化流模型
                self.classifier_fb_optimizer.zero_grad()
                c_loss_flow.backward()
                self.classifier_fb_optimizer.step()


                # 训练 EWC
                # loss = self.ce(logits, labels)
                self.zero_grad()
                logits, _ = self.arch(features, task, middle_feature=False)
                loss = self.Multi_Class_Cross_Entropy(logits, labels, task) #+ c_loss_flow

                if task != 0:
                    loss_ewc = 0
                    for t in range(task):
                        for n, p in self.named_parameters():
                            l = self.fisher[t][n]
                            l = l * (p - self.params[t][n]).pow(2)
                            loss_ewc += l.sum()
                    loss = loss + self.lamb * loss_ewc
                loss.backward()
                self.opt.step()

            val_loss, acc, mif1, maf1 = self.evaluation(val_dataloader, task, valid=True)
            print()
            print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}, c_loss_flow:{}'.format(epoch, val_loss, acc, mif1, maf1, c_loss_flow))
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.state_dict())
                # patience = self.lr_patience
                patience = 50
            else:
                patience -= 1
                if patience <= 0:
                    break
                    # lr /= self.lr_factor
                    # if lr < self.lr_min:
                    #     break
                    # patience = self.lr_patience
                    # self.opt = torch.optim.SGD(self.parameters(),lr=lr, momentum=0.9, weight_decay=5e-4)
        self.load_state_dict(deepcopy(best_model))
        fisher, params = self.calculate_fisher(train_dataloader, task)
        self.current_task = task
        self.fisher[self.current_task] = fisher
        self.params[self.current_task] = params

    @torch.no_grad()

    def evaluation(self, test_dataloader, task, valid=False):
        self.eval()
        total_prediction = np.array([])
        total_labels = np.array([])
        total_loss = 0
        all_features_h = []
        all_labels = []
        for features, labels in test_dataloader:
            features, labels = features.to(self.args.device), labels.to(self.args.device)
            logits, _ = self.arch(features, task, middle_feature=False)
            loss = self.ce(logits,labels)
            prob, prediction = torch.max(logits, dim=1)
            total_loss = total_loss + loss.cpu().item()
            total_labels = np.concatenate((total_labels, labels.cpu().numpy()), axis=0)
            total_prediction = np.concatenate((total_prediction, prediction.cpu().numpy()), axis=0)

        acc = accuracy_score(total_labels, total_prediction)
        mif1 = f1_score(total_labels, total_prediction, average='micro')
        maf1 = f1_score(total_labels, total_prediction, average='macro')
        if valid:
            return total_loss, round(acc*100, 2), round(mif1*100, 2), round(maf1*100, 2)
        return round(acc*100, 2), round(mif1*100, 2), round(maf1*100, 2)

    def get_1d_nflow_model(self,
                           feature_dim,
                           hidden_feature,
                           context_feature,
                           num_layers):
        transforms = []

        for l in range(num_layers):
            assert num_layers // 2 > 1

            if l < num_layers // 2:
                transforms.append(ReversePermutation(features=feature_dim))
            else:
                transforms.append(RandomPermutation(features=feature_dim))

            mask = (torch.arange(0, feature_dim) >= (feature_dim // 2)).float()

            net_func = lambda in_d, out_d: ResidualNet(in_features=in_d, out_features=out_d,
                                                       hidden_features=hidden_feature, context_features=context_feature,
                                                       num_blocks=2, activation=F.leaky_relu, dropout_probability=0)

            transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=net_func))

        transform = CompositeTransform(transforms)
        base_dist = StandardNormal(shape=[feature_dim])
        flow = Flow(transform, base_dist)
        return flow

    def train_a_batch_flow(self, features, labels, last_flow, task):
        xa = self.arch.forward_to_xa(features)
        xa = xa.reshape(xa.shape[0], -1)
        if not self.class_incremental and task > 0:
            labels = labels + (task * self.num_each_class)

        y_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        loss_data = -self.flow.log_prob(inputs=xa, context=y_one_hot).mean()

        if type(last_flow) != type(None):
            batch_size = features.shape[0]
            with torch.no_grad():
                available_labels_past = [label for sublist in self.available_labels_past[:task+1] for label in sublist]
                flow_xa, label, label_one_hot = self.sample_from_flow(last_flow, available_labels_past, self.args.train_batch_size)

            loss_last_flow = -self.flow.log_prob(inputs=flow_xa, context=label_one_hot).mean()
        else:
            loss_last_flow = 0
        # loss_last_flow = 0
        loss_last_flow = self.args.k_flow_lastflow * loss_last_flow
        loss = loss_data + loss_last_flow
        self.flow_optimizer.zero_grad()
        loss.backward()
        self.flow_optimizer.step()
        return {'flow_loss': loss_data.item(), 'flow_loss_last': myitem(loss_last_flow)}  # 返回当前数据损失和先前流模型损失

    def sample_from_flow(self, flow, labels, batch_size):
        label = np.random.choice(labels, batch_size)
        class_onehot = np.zeros((batch_size, self.num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        class_onehot = torch.Tensor(class_onehot).cuda()

        flow_xa = flow.sample(num_samples=1, context=class_onehot).squeeze(1)
        # flow_xa,_ = flow.sample_and_log_prob(num_samples=1, context=class_onehot)
        # squeeze(1)用于移除采样结果中多余的维度
        flow_xa = flow_xa.detach()
        return flow_xa, label, class_onehot

    def probability_in_current_task(self, xa_u, y, prob_mean, flow_xa, flow_label):
        flow_xa_label_set = set(flow_label)
        flow_xa_prob = torch.zeros([flow_xa.shape[0]], device=flow_xa.device)
        for flow_yi in flow_xa_label_set:
            if (y==flow_yi).sum()>0:
                xa_u_yi = xa_u[y==flow_yi]
                xa_u_yi_mean = torch.mean(xa_u_yi, dim=0, keepdim=True)
                xa_u_yi_var = torch.mean((xa_u_yi-xa_u_yi_mean)*(xa_u_yi-xa_u_yi_mean), dim=0, keepdim=True)

                flow_xa_yi = flow_xa[flow_label==flow_yi]
                prob_xa_yi_ = 1/np.sqrt(2*np.pi)*torch.pow(xa_u_yi_var+eps, -0.5)*torch.exp(-torch.pow(flow_xa_yi-xa_u_yi_mean, 2)*torch.pow(xa_u_yi_var+eps, -1)*0.5)
                prob_xa_yi = torch.mean(prob_xa_yi_, dim=1)
                flow_xa_prob[flow_label==flow_yi] = prob_xa_yi
            else:
                flow_xa_prob[flow_label==flow_yi] = prob_mean
        return flow_xa_prob

    def low_probability_in_original_data(self, train_dataloader):
        probabilities = []
        all_features = []
        all_labels = []
        for features, labels in train_dataloader:
            features, labels = features.to(self.args.device), labels.to(self.args.device)
            idx_labels_one = (labels == 0)

            features_one = features[idx_labels_one]
            labels_one = labels[idx_labels_one]

            y_one_hot = F.one_hot(labels_one, num_classes=self.num_classes).float()
            with torch.no_grad():
                xa = self.arch.forward_to_xa(features_one)
                xa = xa.reshape(xa.shape[0], -1)
                prob = self.flow.log_prob(xa, context=y_one_hot)
            probabilities.append(prob)
            all_features.append(features_one)
            all_labels.append(labels_one)


        all_probabilities = torch.cat(probabilities)
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        print("num of sampling:", len(all_probabilities) * self.args.forget_rate)


        threshold = torch.quantile(all_probabilities, self.args.forget_rate)


        low_prob_features = all_features[all_probabilities <= threshold]
        low_prob_labels = all_labels[all_probabilities <= threshold]
        return low_prob_features.cpu(), low_prob_labels.cpu()

    def boundary_expanding_xa(self, train_forget_loader, TASK):
        start = time.time()

        num_classes = 2
        n_filter2 = 1000

        original_classifier = self.arch.predict[TASK].to(self.args.device)
        widen_classifier = torch.nn.Linear(n_filter2, num_classes + 1)
        w_init(widen_classifier)
        self.arch.predict[TASK] = widen_classifier.to(self.args.device)  # 替换原有的分类层


        widen_classifier.weight.data[:num_classes] = original_classifier.weight.data.clone()
        widen_classifier.bias.data[:num_classes] = original_classifier.bias.data.clone()

        optimizer = optimizer_picker(self.args.optim_name, self.arch.parameters(), lr=0.00001, momentum=0.9)
        criterion = loss_picker('cross')

        for epoch in trange(self.args.forget_epoch, leave=False):  # self.args.epochs
            for features, labels in train_forget_loader:
                features, labels = features.to(self.args.device), labels.to(self.args.device)
                target_label = torch.full_like(labels, num_classes, device=self.args.device)

                self.arch.train()
                optimizer.zero_grad()

                logits = self.arch.forward(features, 100)
                widen_loss = criterion(logits, target_label)

                widen_loss.backward()
                optimizer.step()

        # 裁剪模型
        pruned_classifier = torch.nn.Linear(n_filter2, num_classes)
        pruned_classifier.weight.data = widen_classifier.weight.data[:num_classes].clone()
        pruned_classifier.bias.data = widen_classifier.bias.data[:num_classes].clone()
        self.arch.predict[TASK] = pruned_classifier
