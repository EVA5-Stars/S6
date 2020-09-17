import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
from torchvision import datasets, transforms

def init_train_test_loader(batch_size = 128):

    print("\n Initialize train and test loader with Batch Size:{}".format(batch_size))

    torch.manual_seed(22)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([                            
                            transforms.RandomRotation((-10.0,10.0), fill=(1,)),                          
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=batch_size, shuffle=True, **kwargs)
       
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader
