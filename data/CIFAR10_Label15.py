import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import copy
from torch.utils.data import TensorDataset

cf10_dir = './datasets/cf10/'
file_dir = './datasets/cf10_label/binary_cf10'

num_task = 5
each_task_classes = 2


def get(args, pc_valid=0.10):
    seed = args.seed
    data = {}
    taskcla = []

    size = [3, 32, 32]
    if not os.path.isdir(cf10_dir):
        os.makedirs(cf10_dir)
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

        mean = (0.4914, 0.4821, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

        dat = {}
        dat['train'] = datasets.CIFAR10(cf10_dir, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR10(cf10_dir, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for n in range(num_task):
            data[n] = {}
            data[n]['name'] = 'smnist'
            data[n]['ncla'] = each_task_classes
            data[n]['train'] = {'x': [], 'y': []}
            data[n]['test'] = {'x': [], 'y': []}
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                n = target.numpy()[0]
                nn = (n // each_task_classes)  # nn: task
                data[nn][s]['x'].append(image)  # 255
                # data[nn][s]['y'].append(n)
                data[nn][s]['y'].append(n % each_task_classes)

        # overlap
        # 0-10  5-15 10-20 20-30, 22-32, 30-40, 40-50

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(file_dir), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(file_dir), 'data' + str(t) + s + 'y.bin'))


    # Load binary files
    data = {}
    # ids=list(shuffle(np.arange(5),random_state=seed))
    ids = list(np.arange(num_task))
    ids = np.array([0, 1, 2, 3, 4])
    print('Task order =', ids)
    for i in range(5):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(file_dir), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(file_dir), 'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'CIFAR10-' + str(ids[i])

    # Validation
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    percent = args.precent
    task_idx = 0

    train_data = data[task_idx]['train']

    task_idx_labels_list = [label.item() for label in train_data['y']]
    task_idx_label = list(set(task_idx_labels_list))

    orig_class = task_idx_label[1]
    new_class = task_idx_label[0]

    orig_class_idxs = [idx for idx, label in enumerate(train_data['y']) if label == orig_class]

    np.random.seed(seed)
    shuffle_idxs = np.random.permutation(orig_class_idxs)
    num_change = int(len(shuffle_idxs) * percent)
    change_idxs = shuffle_idxs[:num_change]


    for idx in change_idxs:
        train_data['y'][idx] = new_class
    data[task_idx]['train'] = train_data

    return data, taskcla, size#retrain_data, orig_class#



