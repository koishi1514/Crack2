from networks.efficientunet import Effi_UNet
from networks.enet import ENet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS, UNet_URPC, UNet_CCT
import argparse
from networks.config import get_config
from networks.nnunet import initialize_network
from networks.nnunet_1 import initialize_network as initialize_network1
from networks.crackformerII import crackformer
from networks.crackmer.Net import crackmer
from networks.CTCrackseg.TransMUNet import TransMUNet
from networks.SimCrack.consnet import ConsNet
from networks.DTrCNet.CTCNet import CTCNet
# from models.decoder import build


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
    elif net_type == "nnUNet":
        net = initialize_network(threeD=False, num_classes=class_num).cuda()
    elif net_type == "nnUNet_1":
        net = initialize_network1(threeD=False, num_classes=class_num).cuda()
    elif net_type == "crackformer":
        net = crackformer(in_channels=in_chns, final_hidden_dims=64, num_classes=class_num).cuda()
    # elif net_type == "SCSegamba":
    #     net, _ = build(args=None)
    #     net = net.cuda()
    elif net_type == "crackmer":
        # 难以训练，暂时弃用
        net = crackmer(in_channels=in_chns, final_hidden_dims=64, num_classes=class_num).cuda()
    elif net_type == "CTCrackseg":
        net = TransMUNet(n_classes=class_num).cuda()
    elif net_type == "SimCrack":
        net = ConsNet(n_channels=in_chns, n_classes=class_num, att=False, consistent_features=False, img_size=256).cuda()
    elif net_type == "DTrCNet":
        net = CTCNet(n_channels=in_chns, n_classes=class_num).cuda()
    else:
        net = None
    return net
