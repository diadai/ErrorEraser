import argparse

def init_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['CIFAR100','SplitMNIST','PMNIST','CIFAR10', 'CIFAR10_gauss', 'mixture','SplitMNIST_label', 'CIFAR10_label', 'CIFAR10_Label15'], default='CIFAR10_Label15')
    parser.add_argument('--arch', '--architecture', type=str, choices=['ResNet', 'CNN0422'], default='CNN0422')
    parser.add_argument('--method', type=str, choices=['Finetune','Joint','CR','EWC2','HAT','GEM','MAS','DERPP','TAT','Prompt', 'HAT2', 'EWC','EWC+F2BCL'], default='EWC+F2BCL')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--train_unlearn_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_min', type=float, default=1e-9) #default=1e-5
    parser.add_argument('--lr_patience', type=int, default=6)
    parser.add_argument('--lr_factor', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./save_models')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--warm', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--flow_lr', type=float, default=1e-4)
    parser.add_argument("--flow_explore_theta", type=float, default=0.2)
    parser.add_argument("--k_flow_lastflow", type=float, default=0.2)
    parser.add_argument("--k_loss_flow", type=float, default=0.1)
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')
    parser.add_argument("--threshord", type=float, default=0)
    parser.add_argument("--forget_rate", type=float, default=0.1)
    parser.add_argument("--forget_epoch", type=int, default=4)
    parser.add_argument("--precent", type=float, default=0.3)
    args = parser.parse_args()
    return args

#  python main.py --method EWC2 --dataset SplitMNIST --lr 0.05 --lr_factor 3 --lr_min 1e-4 --lr_patience 5