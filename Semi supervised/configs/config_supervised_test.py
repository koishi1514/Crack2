import  argparse
import os

dataset = 'CRACK500'
# CRACK500, CFD
data_path = os.path.join('../dataset', dataset)
root_path = '..'

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default=root_path, help='Name of Experiment')
parser.add_argument('--data_path', type=str,
                    default=data_path, help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default=dataset, help='Name of Dataset')
parser.add_argument('--exp', type=str,
                    default='test_1class', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')

parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--epoch_num', type=int, default=10,
                    help='epochs')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=1,
                    help='output channel of network')


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
