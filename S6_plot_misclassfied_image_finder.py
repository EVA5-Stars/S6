import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from S6.S6_data_loader import init_train_test_loader
from torch.optim.lr_scheduler import StepLR
train_loader, test_loader = init_train_test_loader()
from S6.S6_train_test_function import train
from S6.S6_train_test_function import test

def misclassified_image_finder(model, model_path, device, train_loader, image_num, msg):
    
    data_iter = iter(test_loader)
    figure = plt.figure()

    plt.title('Misclassified Images: With {}'.format(msg))
   
    for _i in range(image_num):
          data, target = data_iter.next()

          model.load_state_dict(torch.load(model_path)) 
          model.eval()

          data, target = data.to(device), target.to(device)

          output = model(data)
          pred = output.argmax(dim=1, keepdim=True) 

          for a in range(256):
              if(pred[a]!=target[a]):
                  
                  plt.subplot(5,5,_i+1)
                  plt.axis('off')
                  plt.imshow(data[a].cpu().numpy().squeeze(),cmap='gray_r')
