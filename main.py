from Utils import *
from init_parameters import init_parameters
from data.load_data import *
import importlib
from time import time
import pandas as pd
import numpy as np
CUDA_LAUNCH_BLOCKING=1

import torch
torch.cuda.empty_cache()


def main(args):

    f = open(f'./results/mis_cifar10_task1/{args.method}_cifar10_ori.txt'.format(args.method),'a')
    print('forget_rate:{}'.format(args.forget_rate),'epoch:{}'.format(args.epochs),'forget_epoch:{}'.format(args.forget_epoch),'precent:{}'.format(args.precent))
    f.write('forget_rate:{}'.format(args.forget_rate))
    f.write(' epoch:{}'.format(args.epochs))
    f.write(' forget_epoch:{}'.format(args.forget_epoch))
    f.write(' precent:{}\n'.format(args.precent))


    unshuffled_data, taskcla, size = load_dataset(args)

    print('Input size =', size, '\nTask info =', taskcla)
    arch = importlib.import_module(f'models.{args.arch}')
    arch = arch.NET(size, args, taskcla)
    manager = importlib.import_module(f'methods.{args.method}')
    manager = manager.Manager(arch, taskcla, args).to(args.device)

    results = pd.DataFrame([],columns=['stage','task','accuracy','micro-f1','macro-f1','seed'])

    index = np.arange(args.num_tasks)
    args.index = list(np.arange(5))
    print("args.index", args.index)
    data = {}
    for idx, i in enumerate(args.index):
        data[idx] = unshuffled_data[i]


    train_dataloaders, test_dataloaders, val_dataloaders = data2dataloaders(data, args)


    t0 = time()

    acc_dict = {}
    for task in range(args.num_tasks):

        print('Train task:{}'.format(task))
        manager.train_with_eval(train_dataloaders[task], val_dataloaders[task], task)
        if task==0:
              low_prob_features, low_prob_labels = manager.low_probability_in_original_data(train_dataloaders[task])

        if task == 0:
            train_unlearn_dataloaders = get_unlearn_loader2(low_prob_features, args, low_prob_labels)
            # manager.forget_shrink(train_unlearn_dataloaders)
            # manager.boundary_expanding(train_unlearn_dataloaders)
            manager.boundary_expanding_xa(train_unlearn_dataloaders, 1)

        for previous in range(task+1):
            acc, mif1, maf1 = manager.evaluation(test_dataloaders[previous], previous)
            if task not in acc_dict:
                acc_dict[task] = {}
            acc_dict[task][previous] = acc

            print('Stage:{} Task:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(task, previous, acc, mif1, maf1))
            # 打印当前任务和已训练任务的评估结果
            # writer.add_scalar(f'{args.method}/{previous}/acc',acc,task)
            # writer.add_scalar(f'{args.method}/{previous}/mif1',mif1,task)
            # writer.add_scalar(f'{args.method}/{previous}/maf1',maf1,previous)
            # 将评估结果写入 SummaryWriter 对象中
            results.loc[len(results.index)] = [task,previous,acc,mif1,maf1,args.seed]

    t1 = time()
    args.time = t1 - t0
    # save_results(results,args)
    each_last_values = [list(values.values())[-1] for values in acc_dict.values()]
    average_last_value = sum(each_last_values) / len(each_last_values)

    last_dict = {k: v for k, v in sorted(acc_dict.items())}
    values = list(last_dict.values())[-1]

    average_acc = sum(values.values()) / len(values)


    all_forget = 0
    all_task_acc = 0
    all_task_for = 0
    for inner_key2, inner_value2 in values.items():
        max_forget = 0
        each_task_acc = 0
        each_task_for = 0
        num = 0
        for key, value in acc_dict.items():
            for inner_key, inner_value in value.items():
                if inner_key == inner_key2:
                    forget = inner_value - inner_value2
                    each_task_acc += inner_value
                    each_task_for += forget
                    num += 1
                    if forget > max_forget:
                        max_forget = forget
        all_task_acc+=(each_task_acc/num)
        all_forget += max_forget
        all_task_for += (each_task_for / num)
    average_forget = all_forget/ (len(values)-1)
    print("average_acc:", average_acc)
    print("average_for:", average_forget)
    print("avr_each_acc:", all_task_acc/len(values))
    print("avr_each_for:", all_task_for/len(values))
    print("new_task_acc:", average_last_value)
    print("acc_dict:", acc_dict)

    # Comprehensive_indicators = (all_task_acc) / (all_task_for * len(values))
    # Comprehensive_indicators2 = (all_task_acc) / (average_forget * len(values))
    Comprehensive_indicators4 = (all_task_acc * all_task_for) / (all_task_for + all_task_acc)
    Comprehensive_indicators3 = (average_acc * all_task_acc) / (all_task_for + average_acc + all_task_acc)
    Comprehensive_indicators5 = (average_acc * all_task_acc) / (average_forget + average_acc + all_task_acc)
    Comprehensive_indicators6 = (average_last_value * all_task_acc) / (
                average_forget + average_last_value + all_task_acc)

    Comprehensive_indicators7 = (average_last_value * all_task_acc * average_acc * average_forget) / ((all_task_for + average_last_value + all_task_acc + average_acc + average_forget) * len(
        values) ** 3)

    # print("Comprehensive_indicators:", Comprehensive_indicators)
    # print("Comprehensive_indicators2:", Comprehensive_indicators2)
    print("Comprehensive_indicators3:", Comprehensive_indicators3)
    print("Comprehensive_indicators4:", Comprehensive_indicators4)
    print("Comprehensive_indicators5:", Comprehensive_indicators5)
    print("Comprehensive_indicators6:", Comprehensive_indicators6)
    print("Comprehensive_indicators7:", Comprehensive_indicators7)

    f.write(" acc_dict:{}\n".format(acc_dict))
    f.write(" average acc:{}\n".format(average_acc))
    f.write(" average for:{}\n".format(average_forget))
    f.write(" avr each acc:{}\n".format(all_task_acc/len(values)))
    f.write(" avr_each_for:{}\n".format(all_task_for/len(values)))
    f.write(" new_task_acc:{}\n".format(average_last_value))
    # f.write(" Comprehensive_indicators:{}\n".format(Comprehensive_indicators))
    # f.write(" Comprehensive_indicators2:{}\n".format(Comprehensive_indicators2))
    f.write(" Comprehensive_indicators3:{}\n".format(Comprehensive_indicators3))
    f.write(" Comprehensive_indicators4:{}\n".format(Comprehensive_indicators4))
    f.write(" Comprehensive_indicators5:{}\n".format(Comprehensive_indicators5))
    f.write(" Comprehensive_indicators6:{}\n".format(Comprehensive_indicators6))
    f.write(" Comprehensive_indicators7:{}\n".format(Comprehensive_indicators7))
    f.write("******************************************************")
    f.close()


def joint_training(args):
    args.class_incremental = True
    train_dataloaders, test_dataloaders, val_dataloaders, taskcla, size = load_joint_data(args)
    arch = importlib.import_module(f'models.{args.arch}')
    arch = arch.NET(size, args)
    manager = importlib.import_module(f'methods.Finetune')
    manager = manager.Manager(arch, taskcla, args).to(args.device)

    results = pd.DataFrame([],columns=['stage','task','accuracy','micro-f1','macro-f1','seed'])
    index = np.arange(args.num_tasks)
    np.random.shuffle(index)
    args.index = index

    t0 = time()
    manager.train_with_eval(train_dataloaders[0], val_dataloaders[0], 0)
    acc, mif1, maf1 = manager.evaluation(test_dataloaders[0], 0)
    print('Stage:{} Task:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(0, 0, acc, mif1, maf1))
    results.loc[len(results.index)] = [0,0,acc,mif1,maf1,args.seed]
    t1 = time()
    args.time = t1 - t0
    # save_results(results,args)

if __name__ == '__main__':
    args = init_parameters()

    args.device = 'cuda:{}'.format(str(args.gpu_id)) if torch.cuda.is_available() else 'cpu'
    args.class_incremental = True
    if args.dataset == 'PMNIST':
        args.class_incremental = False
        args.num_tasks = 10
    elif args.dataset == 'CIFAR100':
        args.num_tasks = 10
    elif args.dataset == 'CIFAR10':
        args.num_tasks = 2
    elif args.dataset == 'CIFAR10_label':
        args.num_tasks = 2
    elif args.dataset == 'CIFAR10_Label15':
        args.num_tasks = 5
    elif args.dataset == 'SplitMNIST':
        args.num_tasks = 2
    elif args.dataset == 'SplitMNIST_label':
        args.num_tasks = 5
    elif args.dataset == 'CIFAR10_gauss':
        args.num_tasks = 2
    elif args.dataset == 'mixture':
        args.num_tasks = 3
    set_seed(args.seed)
    args.each_classes = 2
    if args.method == 'Joint':
        joint_training(args)
    else:
        for args.method in ["EWC+F2BCL"]:
            for args.precent in [0.5]:#,
                for args.forget_rate in [0.01]:#,0.01, 0.05, 0.1, 0.15
                    for args.epochs in [100]:#50, 100
                #     for args.forget_epoch in [4, 5, 6]:
                        main(args)
        # for args.method in ['TAT','MAS','Lwf','NF_EWC_0504','DERPP','HAT']:
        #     main(args)
