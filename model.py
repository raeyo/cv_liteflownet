'''
reference: 
https://github.com/NVIDIA/flownet2-pytorch
https://github.com/sniklaus/pytorch-liteflownet

'''
import torch
import torch.nn as nn
import sys
from torchvision.transforms import Resize

#reference: https://github.com/NVIDIA/flownet2-pytorch
def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    nn.BatchNorm2d(out_channels),
    nn.LeakyReLU(0.1,inplace=True)
  )
#reference: https://github.com/NVIDIA/flownet2-pytorch
def deconv(in_channels, out_channels):
  return nn.Sequential(
    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
    nn.LeakyReLU(0.1,inplace=True)
  )
#reference: https://github.com/sniklaus/pytorch-liteflownet
backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 
                             1.0 - (1.0 / tenFlow.shape[3]), 
                             tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end
#reference: https://github.com/sniklaus/pytorch-liteflownet
try:
	from .correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python


class NetC(nn.Module):
  def __init__(self, feature_dims):
    super(NetC, self).__init__()
    self.conv_list = []
    in_channels = 3
    for level, out_channels in enumerate(feature_dims):
      level += 1
      if level == 1:
        self.conv_list.append(nn.Sequential(
          conv(in_channels, out_channels, 7, 1, 3)
        ))
        in_channels = out_channels
      elif level == 2:
        self.conv_list.append(nn.Sequential(
          conv(in_channels, out_channels, 3, 2, 1),
          conv(out_channels, out_channels, 3, 1, 1),
          conv(out_channels, out_channels, 3, 1, 1)
        ))
        in_channels = out_channels
      elif level in [3, 4]:
        self.conv_list.append(nn.Sequential(
          conv(in_channels, out_channels, 3, 2, 1),
          conv(out_channels, out_channels, 3, 1, 1)
        ))
        in_channels = out_channels
      else: # 5, 6
        self.conv_list.append(conv(in_channels, out_channels, 3, 2, 1))
        in_channels = out_channels

    self.conv_list = nn.ModuleList(self.conv_list)
  
  def forward(self, x):
    features = []
    for conv_layer in self.conv_list:
      x = conv_layer(x)
      features.append(x)
    return features

class NetE(nn.Module):
  def __init__(self, feature_dim=128):
    super(NetE, self).__init__()
    # Matching Unit
    self.M_upconv = deconv(2, 2)

    self.M_fwarp = backwarp
    self.M_corr = correlation.FunctionCorrelation
    
    self.M_corr_conv = nn.Conv2d(feature_dim*2, 49, 1, 1)
    
    self.M_conv = nn.Sequential(
      conv(49, feature_dim, 3, 1),
      conv(feature_dim, feature_dim//2, 3, 1),
      conv(feature_dim//2, feature_dim//4, 3, 1),
      conv(feature_dim//4, 2, 3, 1)
    )
    
    # Subpixel Refinement Unit
    self.S_fwarp = backwarp

    self.S_conv = nn.Sequential(
      conv(feature_dim*2+2, feature_dim, 3, 1),
      conv(feature_dim, feature_dim//2, 3, 1),
      conv(feature_dim//2, feature_dim//4, 3, 1),
      conv(feature_dim//4, 2, 3, 1)
    )
    
    # Regulariation Unit
    self.R_rgbwarp = backwarp #reference
    
    self.R_conv = nn.Sequential(
      conv(feature_dim+3, feature_dim, 3, 1),
      conv(feature_dim, feature_dim, 3, 1),
      
      conv(feature_dim, feature_dim//2, 3, 1),
      conv(feature_dim//2, feature_dim//2, 3, 1),
      
      conv(feature_dim//2, feature_dim//4, 3, 1),
      conv(feature_dim//4, feature_dim//4, 3, 1),
      
      conv(feature_dim//4, 9, 3, 1), # conv dist R
    )
    self.unfold = torch.nn.Unfold(kernel_size=3, padding=1)
        
  def forward(self, f1, f2, prev_flow, im1, im2):
    # Matching Unit
    if prev_flow is None:
      prev_flow = 0.0 #reference: https://github.com/sniklaus/pytorch-liteflownet
      warped_f2 = f2
    else:
      prev_flow = self.M_upconv(prev_flow)
      # feature warping
      warped_f2 = self.M_fwarp(f2, prev_flow)
      
    # cost = self.M_corr_conv(torch.cat((f1, f2), dim=1))
    cost = self.M_corr(f1, warped_f2, intStride=1)
    delta_flow = self.M_conv(cost)
    
    pred_flow = prev_flow + delta_flow # elementwise-sum

    # Subpixel Refinement
    warped_f2 = self.S_fwarp(f2, pred_flow)
    delta_flow = self.S_conv(torch.cat([f1, warped_f2, pred_flow], dim=1))
    
    pred_flow = pred_flow + delta_flow

    # Regulariation Unit
    
    # remove mean of flow field
    rm_flow = pred_flow.view(pred_flow.size(0), pred_flow.size(1), -1)
    rm_flow = rm_flow - rm_flow.mean(dim=2, keepdim=True)
    
    # Image warp
    warped_im2 = self.R_rgbwarp(im2, pred_flow)
    
    # Brightness error  == occlusion probability
    O = torch.norm(warped_im2-im1, p=2, dim=1, keepdim=True)
    
    # distance metric
    D = self.R_conv(torch.cat([f1, rm_flow.view_as(pred_flow), O], dim=1)) # B, 9, H, W
    
    # filter of f-lcon
    # reference: https://github.com/sniklaus/pytorch-liteflownet
    D = -D**2
    g = torch.nn.functional.softmax(D, dim=1) # B, 9, H, W
     
    u_flow = pred_flow[:, 0:1, :, :]
    v_flow = pred_flow[:, 1:2, :, :]
    u_flow = torch.sum(self.unfold(u_flow) * g.view(g.size(0), g.size(1), -1), dim=1).view_as(u_flow)
    v_flow = torch.sum(self.unfold(v_flow) * g.view(g.size(0), g.size(1), -1), dim=1).view_as(v_flow)
    
    pred_flow = torch.cat([u_flow, v_flow], dim=1)
    
    return pred_flow

class LiteFlowNet(nn.Module):
  
  def __init__(self, feature_dims=[32, 32, 64, 96, 128, 192]):
    super(LiteFlowNet, self).__init__()
    self.feature_descriptor = NetC(feature_dims)
  
    self.flow_estimator = []
    for dim in feature_dims:
      self.flow_estimator.append(NetE(dim))
    self.flow_estimator = nn.ModuleList(self.flow_estimator)
  

  def forward(self, I1, I2):
    '''Pyramidal Feature Extraction'''
    pyramid_features1 = self.feature_descriptor(I1)
    pyramid_features2 = self.feature_descriptor(I2)
    
    height = I1.size(2)
    width = I1.size(3)
    
    
    '''Cascaded Flow Inference and Flow Regularization'''
    # pyramid features: [F1, F2, F3, F4, F5, F6]
    # feature dim:      3 -> [32, 32, 64, 96, 128, 192]
    # image size(height): 384 -> [384, 192, 96, 48, 24, 12]
    flow = None
    for level in range(6):
      level = 5 - level
      im1 = Resize((height//(2**level), width//(2**level)))(I1)
      im2 = Resize((height//(2**level), width//(2**level)))(I2)

      f1 = pyramid_features1[level]
      f2 = pyramid_features2[level]
      
      flow = self.flow_estimator[level](f1, f2, flow, im1, im2)

    return flow