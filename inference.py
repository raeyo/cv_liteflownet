import argparse
import cv2
import numpy as np
from numpy.core.fromnumeric import trace
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import Resize

import neptune.new as neptune

from datasets import MpiSintel
from utils.flow_utils import flow2img
from model import LiteFlowNet

from tools import LiteFlowNetLoss, visualize_result

parser = argparse.ArgumentParser()
'''Dataset arguments'''
parser.add_argument('--root', 
    default='/data2/raeyo/optical_flow/MPI-Sintel/training', help='data root of MPI Sintel training dataset')
#region reference: https://github.com/NVIDIA/flownet2-pytorch
parser.add_argument('--crop_size', type=int, nargs='+', 
default = [384, 768], help="Spatial dimension to crop training samples for training")
parser.add_argument('--inference_size', type=int, nargs='+', 
default = [384, 768], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
parser.add_argument('--gpu', default="0", help='gpu id')
#endregion

parser.add_argument('--log_dir', 
    default='/data2/raeyo/results/liteflownet_MSR_stage', help='logging dir')
  
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

# log_dir = "/data2/raeyo/results/liteflownet_MSR_stage"
log_dir = args.log_dir

'''load dataset'''
data_root = args.root
test_dataset = MpiSintel(args, is_cropped=False, root=data_root, split="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)


'''load Model'''
model = LiteFlowNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
state_dict = torch.load(os.path.join(log_dir, "weights/best.pkl"))
model.load_state_dict(state_dict)

criterion = LiteFlowNetLoss()

with torch.no_grad():
  model = model.eval()
  losses = []
  for itr, data in enumerate(tqdm(test_loader)):
    input_image1 = data[0][:, :, 0, :, :].cuda() # --> B, 3, 384, 1024
    input_image2 = data[0][:, :, 1, :, :].cuda() # --> B, 3, 384, 1024
    target_flow = data[1].cuda() # --> B, 2, 384, 1024
    pred_flows = model(input_image1, input_image2)

    loss = criterion(pred_flows, target_flow)
    losses.append(loss.item())

    save_path = os.path.join(log_dir, "pred_results", "{}.png".format(itr))
    pred_flow = pred_flows[5][2]
    visualize_result(input_image1[0].cpu(), input_image2[0].cpu(), pred_flow[0].cpu(), save_path, gt=target_flow[0].cpu())      
  
  test_loss = np.mean(losses)
  print("test loss | {:<7.3f}".format(test_loss))
  
  
  
  
    
