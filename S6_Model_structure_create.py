import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from S6.BatchNorm_GhostBatchNorm import GhostBatchNorm

class Net(nn.Module):
    def __init__(self,gbatnor):
        super(Net,self).__init__()
        self.gbatnor=gbatnor
        print("Create the instance of the Net class with GBN = {}".format(gbatnor))
        # Input Block
        if gbatnor:      
            self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),nn.ReLU(),GhostBatchNorm(8,2)) 
        else:
            self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),nn.ReLU(),nn.BatchNorm2d(8)) 
        # output_size = 26
        # CONVOLUTION BLOCK 1
        if gbatnor:
            self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),nn.ReLU(),GhostBatchNorm(16,2))
        else:
            self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),nn.ReLU(),nn.BatchNorm2d(16))
        # output_size = 24
        # TRANSITION BLOCK 1
        
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12
        if gbatnor:
            self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),nn.ReLU(),GhostBatchNorm(8,2)) 
        else:
            self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),nn.ReLU(),nn.BatchNorm2d(8)) 
        # output_size = 12

        # CONVOLUTION BLOCK 2
        if gbatnor:
            self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),nn.ReLU(),GhostBatchNorm(16,2)) 
        else:
            self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),nn.ReLU(),nn.BatchNorm2d(16)) 
        # output_size = 10
        
        if gbatnor:
            self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),nn.ReLU(),GhostBatchNorm(32,2)) 
        else:
            self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),nn.ReLU(),nn.BatchNorm2d(32)) 
        # output_size = 8

        # OUTPUT BLOCK
        if gbatnor:
            self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),nn.ReLU(),GhostBatchNorm(10,2))
        else:
            self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),nn.ReLU(),nn.BatchNorm2d(10))
            
        # output_size = 8        
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=8)) # output_size = 1

    # defines the strcuture of the class
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
def create_model(device,gbatnor):
    model = Net(gbatnor).to(device)
    summary(model, input_size=(1, 28, 28))
    return model
