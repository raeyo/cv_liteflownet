import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Resize
import imageio

from utils.flow_utils import flow2img

def visualize_result(image1, image2, flow, save_path, gt=None):
  image1 = np.uint8(image1.permute(1, 2, 0))
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
  image1 = cv2.putText(image1.copy(), "Image 1", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

  image2 = np.uint8(image2.permute(1, 2, 0))
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
  image2 = cv2.putText(image2.copy(), "Image 2", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

  flow = flow.permute(1, 2, 0).detach().numpy()
  flow = flow2img(flow)
  flow = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)
  flow = cv2.putText(flow.copy(), "Flow", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

  if gt is None:
    cv2.imwrite(save_path, np.vstack([image1, image2, flow]))
  else:
    gt = gt.permute(1, 2, 0).numpy()
    gt = flow2img(gt)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = cv2.putText(gt.copy(), "GT", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, np.vstack([image1, image2, flow, gt]))

class LiteFlowNetLoss(nn.Module):
  def __init__(self):
    super(LiteFlowNetLoss, self).__init__()
  
  def L2(self, output, target):
    # reference: https://github.com/NVIDIA/flownet2-pytorch/blob/master/losses.py
    lossvalue = torch.norm(output-target,p=2,dim=1).mean() 
    return lossvalue
  
  def forward(self, pred_flows, target_flow, train_stage=-1):
    height = target_flow.size(2)
    width = target_flow.size(3)
    if train_stage < 0:
      # Test last pred result
      pred_level_idx = [5] # level
      pred_unit_idx = [2]
    
    elif train_stage < 3:
      # Train M, S unit for level6
      pred_level_idx = [0] # level
      pred_unit_idx = [0, 1] # Unit (M, S, R)
      
    elif train_stage < 5:
      # Train M, S, R unit for level6
      pred_level_idx = [0] # level
      pred_unit_idx = [0, 1, 2] # unit
      
    elif train_stage < 8:
      # Train M,S,R for level 6 ~ 5
      pred_level_idx = [0, 1] # level
      pred_unit_idx = [0, 1, 2] # unit
    
    elif train_stage < 11:
      # Train M,S,R for level 6 ~ 4
      pred_level_idx = [0, 1, 2] # level
      pred_unit_idx = [0, 1, 2] # unit
    
    elif train_stage < 14:
      # Train M,S,R for level 6 ~ 3
      pred_level_idx = [0, 1, 2, 3] # level
      pred_unit_idx = [0, 1, 2] # unit
    
    elif train_stage < 17:
      # Train M,S,R for level 6 ~ 2
      pred_level_idx = [0, 1, 2, 3, 4] # level
      pred_unit_idx = [0, 1, 2] # unit
    else:
      # Train Whole Network
      pred_level_idx = [0, 1, 2, 3, 4, 5] # level
      pred_unit_idx = [0, 1, 2] # unit
    loss = 0.0
    for level_idx in pred_level_idx:
      rate = 2**(5-level_idx)
      target = Resize((height//(rate), width//(rate)))(target_flow)
      for unit_idx in pred_unit_idx:
        loss += self.L2(pred_flows[level_idx][unit_idx], target)
    
    return loss
    
def save_image2gif(image_list, save_path, fps=10):
  imageio.mimsave(save_path, image_list, fps=fps)


