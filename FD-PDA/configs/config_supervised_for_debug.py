import  argparse
import os

dataset = 'CRACK500'
# Choose from: CRACK500, CFD, DeepCrack, + CrackTree, CrackLS315, CRKWH200, GAPS384, TUT, AEL (for test only)
model = 'nnUNet_2'
# nnUNet_2 is our method
# crackformer, crackmer, DeepCrack, CTCrackseg
epoch = 10
bs = 1
lr = 0.001
img_size = 256

data_path = os.path.join('../dataset', dataset)
root_path = '..'
# exp_name = f'{dataset}_trained_{model}_for_debug'
exp_name = f'{dataset}_check'

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default=root_path, help='Name of Experiment')
parser.add_argument('--data_path', type=str,
                    default=data_path, help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default=dataset, help='Name of Dataset')
parser.add_argument('--exp', type=str,
                    default=exp_name, help='experiment_name')
parser.add_argument('--model', type=str,
                    default=model, help='model_name')
# nnUNet_2 is our method

parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--epoch_num', type=int, default=epoch,
                    help='epochs')
parser.add_argument('--batch_size', type=int, default=bs,
                    help='batch_size per gpu')
# 5 is for CTCrackseg, 8 is for nnUNet, 16 is for crackformer
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
# for crackmer, lr=0.01 is the best, other models are 0.001
parser.add_argument('--weight_decay', type=float,  default=0.0001,
                    help='weight decay')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--img_size', type=int,  default=img_size)
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=1,
                    help='output channel of network')
parser.add_argument('--loss_weight', type=tuple,  default=(1, 0),
                    help='weight between bce loss and dice loss( [0] for bce, [1] for dice )')

parser.add_argument('--labeled_num', type=int, default=50,
                    help='labeled data count')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
