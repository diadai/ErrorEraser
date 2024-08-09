import sys
import torch
import torch.nn.functional as F
import numpy as np
# Alext net from
# https://github.com/joansj/hat/blob/master/src/networks/alexnet.py

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class NET(torch.nn.Module):

    def __init__(self, shape, args, taskcla):
        super(NET,self).__init__()
        ncha, size = shape[0], shape[1]
        self.args = args
        self.softmax = torch.nn.Softmax(dim=1)

        self.conv1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()


        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*s*s,2048)
        self.fc2=torch.nn.Linear(2048,1000)

        # 定义分类头
        self.class_incremental = args.class_incremental
        if self.class_incremental:
            self.predict = torch.nn.ModuleList()
            for task, n_class in taskcla:
                self.predict.append(torch.nn.Linear(1000, n_class))
        elif self.class_incremental == 'yu':
            for task, n_class in taskcla:
                self.predict = torch.nn.Linear(1000, n_class)
                break
        else:
            self.num_all_class = sum(n_class for _, n_class in taskcla)
            self.predict = torch.nn.Linear(1000, self.num_all_class)
            self.all_classes = list(range(self.num_all_class))
            self.num_each_class = int((self.num_all_class / args.num_tasks))
            self.available_class = []
        self.is_eval_mode = False  # 默认为训练模式
        self.current_max_task = 0  # 跟踪已经训练完成的最大任务编号

        print('CNN')
        return

    # def forward(self, x):
    #     h = self.forward_to_xa(x)
    #     h = self.forward_from_xa(h)
    #     return h
    def set_eval_mode(self, mode=False):
        """ 设置模型为评估模式 """
        self.is_eval_mode = mode

    def forward(self, x, task, middle_feature=False):
        if middle_feature:
            h = self.forward_from_xa(x)
        else:
            h = self.forward_to_xa(x)
            h = self.forward_from_xa(h)

        if self.class_incremental and task != 100 and task!='t-sne':
            logits = self.predict[task](h)

        elif self.class_incremental and task == 100:
            logits = self.predict[1](h)
            return logits

        if task == 't-sne':
            return h

        classes_p = self.softmax(logits)
        return logits, classes_p

    # def apply_mask(self, logits, start, end):
    #     # 应用掩码使得只有指定范围的类别激活
    #     mask = torch.full_like(logits, float('-inf'))
    #     mask[:, start:end] = 0
    #     return logits + mask
    def forward_to_xa(self,x):
        h=self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h=self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h=self.maxpool(self.drop2(self.relu(self.conv3(h))))
        self.pool = h
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        # h=self.drop2(self.relu(self.fc2(h)))
        return h

    def forward_from_xa(self, xa):
        xb = F.leaky_relu(self.fc2(xa))
        return xb

    # def get_cam_feature(self, x):
    #     h = self.forward_to_xa(x)
    #     pooled_features = h
    #     return pooled_features, self.fc2.weight

    # def get_cam_feature(self, x):
    #     feature_map = self.pool
    #
    #     return feature_map, self.fc1.weight

    # def forward_to_xa(self, x):
    #     xa = F.leaky_relu(self.conv1(x))
    #     xa = F.leaky_relu(self.conv2(xa))
    #     xa = F.leaky_relu(self.conv3(xa))
    #     # xa = xa.view(xa.shape[0], (32//8)**2 * self.channel_size*4)
    #     xa = xa.view(x.size(0), -1)
    #     xa = F.leaky_relu(self.fc1(xa))
    #     # xa = F.leaky_relu(self.fc2(xa))
    #     return xa

