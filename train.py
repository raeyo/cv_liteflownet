import argparse
import cv2
import numpy as np
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import neptune.new as neptune

from datasets import MpiSintel
from losses import L2
from utils.flow_utils import flow2img

from model import LiteFlowNet

def visualize_result(image1, image2, flow, save_path):
  image1 = np.uint8(image1.permute(1, 2, 0))
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
  image1 = cv2.putText(image1.copy(), "Image 1", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

  image2 = np.uint8(image2.permute(1, 2, 0))
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
  image2 = cv2.putText(image2.copy(), "Image 2", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

  flow = flow.permute(1, 2, 0).numpy()
  flow = flow2img(flow)
  flow = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)
  flow = cv2.putText(flow.copy(), "Flow", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

  cv2.imwrite(save_path, np.vstack([image1, image2, flow]))


if __name__=="__main__":
  parser = argparse.ArgumentParser()

  '''Dataset arguments'''
  parser.add_argument('--crop_size', type=int, nargs='+', 
    default = [384, 768], help="Spatial dimension to crop training samples for training")
  parser.add_argument('--inference_size', type=int, nargs='+', 
    default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')

  '''train arguments'''
  parser.add_argument('--batch_size', type=int, 
    default = 4, help='batch size for training')
  parser.add_argument('--maximum_epoch', type=int, 
    default = 100, help='maximum epoch')
  parser.add_argument('--test_interval', type=int, 
    default = 5, help='test interval(epoch) while training')
  parser.add_argument('--gpu', default="0", help='gpu id')


  '''logging argument'''
  parser.add_argument('--log_dir', 
    default='/data2/raeyo/results/liteflownet_M0', help='logging dir')
  parser.add_argument('--neptune', 
    action='store_true', help='neptune logging')

  

  args = parser.parse_args()
  if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
    os.mkdir(os.path.join(args.log_dir, "weights"))
    os.mkdir(os.path.join(args.log_dir, "pred"))
  
  os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID" 
  os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

  '''logging'''
  if args.neptune:
    nlogger = neptune.init(
      project="raeyo/liteflownet",
      api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0OTA1Mzk4OS04MWI4LTQ5YjctYTViZi1iZDEyNjFlOWJmMzAifQ==",
      )

    

  '''load dataset'''
  train_root = "/data2/raeyo/optical_flow/MPI-Sintel/training"
  test_root = "/data2/raeyo/optical_flow/MPI-Sintel/training"

  train_dataset = MpiSintel(args, is_cropped=True, root=train_root)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

  test_dataset = MpiSintel(args, is_cropped=True, dstype='final', root=test_root)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

  '''data example'''
  images, flow = train_dataset[0]

  image1 = images[0][:, 0, :, :]
  image2 = images[0][:, 1, :, :]
  flow = flow[0]
  visualize_result(image1, image2, flow, "sample_data.png")

  '''load Model'''
  model = LiteFlowNet()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  '''optimizer'''
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

  '''Loss'''
  criterion = L2()

  best_loss = np.inf

  for epoch in range(args.maximum_epoch):
    optimizer.zero_grad()
    '''train loop'''
    losses = []
    
    for itr, data in enumerate(tqdm(train_loader)):
      input_image1 = data[0][0][:, :, 0, :, :].cuda() # --> B, 3, 384, 1024
      input_image2 = data[0][0][:, :, 1, :, :].cuda() # --> B, 3, 384, 1024
      target_flow = data[1][0].cuda() # --> B, 2, 384, 1024

      pred_flow = model(input_image1, input_image2)
      
      loss = criterion(pred_flow, target_flow)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      losses.append(loss.item())
      if args.neptune:
        nlogger["train/loss"].log(loss.item())
      
    train_loss = np.mean(losses)
    print("[{}/{}] train loss | {:<7.3f}".format(epoch, args.maximum_epoch, train_loss))

    '''evaluation loop'''   
    if epoch % args.test_interval == 0:
      with torch.no_grad():
        model = model.eval()
        losses = []
        
        for itr, data in tqdm(enumerate(test_loader)):
          input_image1 = data[0][0][:, :, 0, :, :].cuda() # --> B, 3, 384, 1024
          input_image2 = data[0][0][:, :, 1, :, :].cuda() # --> B, 3, 384, 1024
          target_flow = data[1][0].cuda() # --> B, 2, 384, 1024
          
          pred_flow = model(input_image1, input_image2)
    
          loss, epe = criterion(pred_flow, target_flow)
          losses.append(loss.item())
          
        save_path = os.path.join(args.log_dir, "pred", "{}_{}.png".format(epoch, itr))
        visualize_result(input_image1[0].cpu(), input_image2[0].cpu(), pred_flow[0].cpu(), save_path)
        test_loss = np.mean(losses)
        print("[{}/{}] test loss | {:<7.3f}".format(epoch, args.maximum_epoch, test_loss))
        
        torch.save(model.state_dict(), os.path.join(args.log_dir, "weights/epoch{}.pkl".format(epoch)))
        if test_loss < best_loss:
          best_loss = test_loss
          torch.save(model.state_dict(), os.path.join(args.log_dir, "weights/best.pkl"))
        if args.neptune:
          nlogger["test/loss"].log(loss.item())
            
