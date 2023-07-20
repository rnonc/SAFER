import torch
import matplotlib.pyplot as plt
import torchvision
import random,math

class data_augm:
    def __init__(self,resolution):
        self.H_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.Jitter = torchvision.transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.1,0.1))
        self.resize = torchvision.transforms.Resize((resolution,resolution),antialias=True)
    def transform(self,x):
        x = self.H_flip(x)
        x = self.Jitter(x)
        x = self.resize(x)
        x = x/255
        #x = (x-torch.mean(x))/torch.std(x)
        return x

class data_adapt:
    def __init__(self,resolution):
        self.resize = torchvision.transforms.Resize((resolution,resolution),antialias=True)
    def transform(self,x):
        x = self.resize(x)
        x = x/255
        #x = (x-torch.mean(x))/torch.std(x)
        return x

def KLDivergence(mu,log_sigma,mu_2=0,log_sigma_2=0):
    return torch.sum(0.5*(log_sigma_2-log_sigma + (torch.exp(log_sigma) + (mu-mu_2.detach())**2)/(torch.exp(log_sigma_2))-1))

def ELBOLoss(target,pred,mu,log_sigma,mu_2=0,log_sigma_2=0,std=0.1):
    kl = KLDivergence(mu,log_sigma,mu_2,log_sigma_2)
    evidence = -((target-pred)/std)**2
    return -(evidence-kl)

def CCCLoss(output,target):
    cov_corr = torch.mean((output-target)**2,dim=1)
    cov_decorr = (torch.mean(output,dim=1)-torch.mean(target,dim=1))**2 +torch.var(output,dim=1)+torch.var(target,dim=1)
    return torch.sum(1-cov_corr/cov_decorr)