import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

img = Image.open('./cat.jpg')
fig = plt.figure()
plt.imshow(img)

transfrom = Compose([Resize((224,224)), ToTensor()]) #this is the pipline for resizing to imagenet size 
x= transfrom(img)
x.shape
x = x.unsqueeze(0) # unsqueeze is used for to add dimention lilke when i want to concatinate x=(2,3) and y=(2,) so I will
# add y.unsqueeze(0) to make y = (2,0)
x.shape


# Note After checking out the original implementation, I found out that 
# the authors are using a Conv2d layer instead of a Linear one for performance
# gain. This is obtained by using a kernel_size and stride equal to the `patch_size`. 
# Intuitively, the convolution operation is applied to each patch individually. So, we have to first 
# apply the conv layer and then flat the resulting images.

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
    
PatchEmbedding()(x).shape