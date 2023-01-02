import torch
from torchmin import minimize,minimize_constr

def pairwise_distances(x, y, power=2, sum_dim=2):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n,m,d)
    y = y.unsqueeze(0).expand(n,m,d)
    dist = torch.pow(x-y, power).sum(sum_dim)
    return dist

def StandardScaler(x,with_std=False):
    mean = x.mean(0, keepdim=True)
    std = x.std(0, unbiased=False, keepdim=True)
    x = x- mean
    if with_std:
        x /= (std + 1e-10)
    return x

def H_Distance(FX,y,l,sigma=None,lamda=1e-2,device=torch.device('cpu')):   
    if sigma is None:
        pairwise_dist = torch.cdist(FX,FX,p=2)**2 
        sigma = torch.median(pairwise_dist[pairwise_dist!=0])  
    domain_label = torch.unique(l)
    target_domain_idx = len(domain_label)-1
    FXt,yt = FX[l==target_domain_idx],y[l==target_domain_idx]
    nt = len(yt)
    div = 0.0
    for dl in domain_label[1:]:
        FXs,ys = FX[l==dl],y[l==dl]
        ns = len(ys)
        FXst,yst = torch.cat((FXs,FXt),dim=0),torch.cat((ys,yt),dim=0)
        FXst_norm = torch.sum(FXst ** 2, axis = -1)
        Kst = torch.exp(-(FXst_norm[:,None] + FXst_norm[None,:] - 2 * torch.matmul(FXst, FXst.t())) / sigma) * (yst[:,None]==yst)
        def Obj(theta):
            """
            Approximation of Hellinger distance
            """
            div = 2. - (torch.mean(torch.exp(-torch.matmul(Kst[:ns],theta))) + torch.mean(torch.exp(torch.matmul(Kst[ns:],theta))))
            reg = lamda * torch.matmul(theta,theta) 
            return -div + reg
    
        theta_0 = torch.zeros(ns+nt, device=device)
        result = minimize(Obj,theta_0,method='l-bfgs')
        theta_hat = result.x
        div = div + 2. - (torch.mean(torch.exp(-torch.matmul(Kst[:ns],theta_hat))) + torch.mean(torch.exp(torch.matmul(Kst[ns:],theta_hat))))
    return div     