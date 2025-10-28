import os.path

from networks.efficientunet import Effi_UNet
from networks.enet import ENet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS, UNet_URPC, UNet_CCT
import argparse
from networks.config import get_config
from networks.nnunet import initialize_network
from networks.nnunet_1 import initialize_network as initialize_network1
from networks.nnunet_2 import initialize_network as initialize_network2
from networks.nnunet_3 import initialize_network as initialize_network3
from networks.crackformerII import crackformer
from networks.crackmer.Net import crackmer
from networks.CTCrackseg.TransMUNet import TransMUNet
from networks.SimCrack.consnet import ConsNet
from networks.DTrCNet.CTCNet import CTCNet
from networks.DeeplabV3.modeling import deeplabv3plus_resnet50
from networks.deepcrack import DeepCrack
from networks.TransUNet.vit_seg_modeling import VisionTransformer
from networks.segment_anything import sam_model_registry
from networks.delta.sam_adapter_lora_image_encoder import LoRA_Adapter_Sam


def net_factory(net_type="unet", in_chns=1, class_num=3, args=None):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "enet":
        net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()

    elif net_type == 'TransUNet':
        from networks.TransUNet.vit_seg_modeling import CONFIGS
        vit_config = CONFIGS['R50-ViT-B_16']
        vit_config.n_classes = class_num
        vit_config.n_skip = 3
        vit_config.patches.grid = (int(args.patch_size[0] / 16), int(args.patch_size[1] / 16))

        net = VisionTransformer(config = vit_config,img_size=args.patch_size,num_classes=class_num ).cuda()

    elif net_type == "nnUNet":
        net = initialize_network(threeD=False, num_classes=class_num).cuda()
    elif net_type == "nnUNet_1":
        net = initialize_network1(threeD=False, num_classes=class_num).cuda()
    elif net_type == "nnUNet_2":
        net = initialize_network2(threeD=False, num_classes=class_num).cuda()
    elif net_type == "nnUNet_3":
        net = initialize_network3(threeD=False, num_classes=class_num).cuda()
    elif net_type == "crackformer":
        net = crackformer(in_channels=in_chns, final_hidden_dims=64, num_classes=class_num).cuda()
    elif net_type == "crackmer":
        # hard to train, not recommended
        net = crackmer(in_channels=in_chns, final_hidden_dims=64, num_classes=class_num).cuda()
    elif net_type == "CTCrackseg":
        net = TransMUNet(n_classes=class_num).cuda()
    elif net_type == "SimCrack":
        net = ConsNet(n_channels=in_chns, n_classes=class_num, att=False, consistent_features=False, img_size=256).cuda()
    elif net_type == "DTrCNet":
        net = CTCNet(n_channels=in_chns, n_classes=class_num).cuda()
    elif net_type == "DeeplabV3+":
        net = deeplabv3plus_resnet50(num_classes=class_num, pretrained_backbone=False).cuda()
    elif net_type == "DeepCrack":
        net = DeepCrack(num_classes=class_num).cuda()
    elif net_type == "CrackSAM":
        sam, img_emb_size = sam_model_registry['vit_h'](image_size = args.img_size, num_classes = args.num_classes,
                                                        checkpoint = r'./checkpoints/sam_vit_h_4b8939.pth',
                                                        pixel_mean = [0, 0, 0], pixel_std = [1, 1, 1])
        net = LoRA_Adapter_Sam(sam, 32, 4).cuda()

    else:
        net = None
    return net
