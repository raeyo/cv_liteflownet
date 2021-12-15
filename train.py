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
torch.set_num_threads(10)
if __name__=="__main__":
  parser = argparse.ArgumentParser()

  '''Dataset arguments'''
  parser.add_argument('--root', 
    default='/data2/raeyo/optical_flow/MPI-Sintel/training', help='data root of MPI Sintel training dataset')
  
  #region reference: https://github.com/NVIDIA/flownet2-pytorch
  parser.add_argument('--crop_size', type=int, nargs='+', 
    default = [384, 768], help="Spatial dimension to crop training samples for training")
  parser.add_argument('--inference_size', type=int, nargs='+', 
    default = [384, 768], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
  #endregion
  
  '''train arguments'''
  parser.add_argument('--batch_size', type=int, 
    default = 4, help='batch size for training')
  parser.add_argument('--maximum_epoch', type=int, 
    default = 200, help='maximum epoch')
  parser.add_argument('--test_interval', type=int, 
    default = 1, help='test interval(epoch)')
  parser.add_argument('--gpu', default="0", help='gpu id')


  '''logging argument'''
  parser.add_argument('--log_dir', 
    default='/data2/raeyo/results/liteflownet_MSR_final', help='logging dir')
  parser.add_argument('--neptune', 
    action='store_true', help='neptune logging')
  
  args = parser.parse_args()
  
  '''CUDA Device setting'''
  os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID" 
  os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

  '''logging'''
  if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
    os.mkdir(os.path.join(args.log_dir, "weights"))
    os.mkdir(os.path.join(args.log_dir, "pred"))
  if args.neptune:
    nlogger = neptune.init(
      project="raeyo/liteflownet",
      api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0OTA1Mzk4OS04MWI4LTQ5YjctYTViZi1iZDEyNjFlOWJmMzAifQ==",
      )
  
  '''load dataset'''
  data_root = args.root
  
  
  train_dataset = MpiSintel(args, is_cropped=True, root=data_root, split="train")
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
  
  test_dataset = MpiSintel(args, is_cropped=False, root=data_root, split="test")
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

  print("Train Data: {}".format(len(train_dataset)))
  print("Test Data: {}".format(len(test_dataset)))
  '''data example'''
  images, flow = train_dataset[0]
  
  image1 = images[:, 0, :, :]
  image2 = images[:, 1, :, :]
  visualize_result(image1, image2, flow, "sample_data.png")

  '''load Model'''
  model = LiteFlowNet()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  '''optimizer'''
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

  '''Loss'''
  criterion = LiteFlowNetLoss()

  best_loss = np.inf

  for epoch in range(args.maximum_epoch):
    '''train loop'''
    optimizer.zero_grad()
    losses = []
    for itr, data in enumerate(tqdm(train_loader)):
      input_image1 = data[0][:, :, 0, :, :].cuda() # --> B, 3, 384, 1024
      input_image2 = data[0][:, :, 1, :, :].cuda() # --> B, 3, 384, 1024
      target_flow = data[1].cuda() # --> B, 2, 384, 1024

      pred_flows = model(input_image1, input_image2)
      
      loss = criterion(pred_flows, target_flow, train_stage=(epoch//10))
      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      losses.append(loss.item())
      
    train_loss = np.mean(losses)
    print("[{}/{}] train loss | {:<7.3f}".format(epoch, args.maximum_epoch, train_loss))
    if args.neptune:
      nlogger["train/loss"].log(train_loss)
    '''evaluation loop'''   
    if epoch % args.test_interval == 0:
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
        
        test_loss = np.mean(losses)
        print("[{}/{}] test loss | {:<7.3f}".format(epoch, args.maximum_epoch, test_loss))
        if args.neptune:
          nlogger["test/loss"].log(test_loss)
        if test_loss < best_loss:
          best_loss = train_loss
          best_epoch = epoch
          torch.save(model.state_dict(), os.path.join(args.log_dir, "weights/best.pkl"))
        save_path = os.path.join(args.log_dir, "pred", "{}_{}.png".format(epoch, itr))
        
        pred_flow = pred_flows[5][2] # pred flow of last level
        visualize_result(input_image1[0].cpu(), input_image2[0].cpu(), pred_flow[0].cpu(), save_path, gt=target_flow[0].cpu())