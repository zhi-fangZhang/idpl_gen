
from config import extract_args
from utils2.cifar10 import load_cifar10
from utils2.cifar100 import load_cifar100
from utils2.fmnist import load_fmnist
from utils2.kmnist import load_kmnist
from utils2.mnist import load_mnist
import torch
import cv2
import numpy as np
from torchvision import utils as vutils
from torchvision import transforms



args=extract_args()
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else 'cpu')

def save_image_tensor(input_tensor: torch.Tensor, filename):
    unloader = transforms.ToPILImage()
    image = input_tensor.cpu().clone() # clone the tensor
    image = unloader(image)
    image.save(filename)


def train():
    print(args.ds)
    if args.ds == "cifar10":
        train_loader, valid_loader, test_loader, dim, K = load_cifar10(args.ds, batch_size=args.bs, device=device)
    if args.ds == "cifar100":
        train_loader, valid_loader, test_loader, dim, K = load_cifar100(args.ds, batch_size=args.bs, device=device)
    if args.ds == "fmnist":
        train_loader, valid_loader, test_loader, dim, K = load_fmnist(args.ds, batch_size=args.bs, device=device)
    if args.ds == "kmnist":
        train_loader, valid_loader, test_loader, dim, K = load_kmnist(args.ds, batch_size=args.bs, device=device)
    if args.ds == "mnist":
        train_loader, valid_loader, test_loader, dim, K = load_mnist(args.ds, batch_size=args.bs, device=device)
    

    
    for features, features_w, features_s, targets, trues, indexes in train_loader:
        for ind,f in enumerate(features):
            save_image_tensor(f,'vis/{}.png'.format(ind))
        print(targets)
        print(trues)
        break






if __name__ == '__main__':
    train()