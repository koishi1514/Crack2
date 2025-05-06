import  argparse
import os

dataset = 'CAMUS'
root_path = os.path.join('../data', dataset)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=root_path, help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default=dataset, help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='CAMUS/ST++ and Mean_Teacher test1', help='experiment_name')
parser.add_argument('--output', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--epoch_num', type=int, default=5,
                    help='epochs')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=10,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=400,
                    help='labeled ratio')
# 只考虑es和ed的话 labeled_num 是固定的

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# parser.add_argument('--data-root', type=str, required=True)
# parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'], default='pascal')
# parser.add_argument('--batch-size', type=int, default=16)
# parser.add_argument('--lr', type=float, default=None)
# parser.add_argument('--epochs', type=int, default=None)
# parser.add_argument('--crop-size', type=int, default=None)
# parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
# parser.add_argument('--output', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
#                     default='deeplabv3plus')
#
# semi-supervised settings
# parser.add_argument('--labeled-id-path', type=str, required=True)
# parser.add_argument('--unlabeled-id-path', type=str, required=True)
# parser.add_argument('--pseudo-mask-path', type=str, required=True)
#
# parser.add_argument('--save-path', type=str, required=True)

# arguments for ST++
# parser.add_argument('--reliable-id-path', default="../data/CAMUS", type=str)
parser.add_argument('--plus', dest='plus', default=True, action='store_true',
                    help='whether to use ST++')

args = parser.parse_args()
