import torch.nn.functional as F
import torch

from meddlr.metrics.functional.image import mae

def pdl(x, y, num_projections=1000):
    '''Projected Distribution Loss (https://arxiv.org/abs/2012.09289)
    x.shape = B,M,N,...
    modified from https://github.com/saurabh-kataria/projected-distribution-loss
    '''
    def rand_projections(dim, device=torch.device('cpu'), num_projections=1000):
        projections = torch.randn((dim,num_projections), device=device)
        projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=0, keepdim=True))    # columns are unit length normalized
        return projections


    x = x.reshape(x.shape[0], -1, x.shape[1])   # B,N,M, where N = H x W
    y = y.reshape(y.shape[0], -1, y.shape[1])
    W = rand_projections(x.shape[-1], device=x.device, num_projections=num_projections)
    e_x = torch.matmul(x,W) # multiplication via broad-casting
    e_y = torch.matmul(y,W)
    loss = 0
    for ii in range(e_x.shape[2]):
        loss = loss + mae(torch.sort(e_x[:,:,ii],dim=1)[0] , torch.sort(e_y[:,:,ii],dim=1)[0])    # if this gives issues; try Huber loss later
    
    return loss


def dists_distance(x, y):
    ''' Distance function inspired by the Deep Image Structure and 
    Texture Similarity (DISTS) Metric (https://arxiv.org/abs/2004.07728).
    Modifed from: https://github.com/dingkeyan93/DISTS
    '''

    c1 = 1e-6
    c2 = 1e-6
    
    x_mean = x.mean([2, 3], keepdim=True)
    y_mean = y.mean([2, 3], keepdim=True)
    S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
    dist1 = (S1).sum(1, keepdim=True)
    
    x_var = ((x - x_mean)**2).mean([2, 3], keepdim=True)
    y_var = ((y - y_mean)**2).mean([2, 3], keepdim=True)
    xy_cov = (x * y).mean([2, 3], keepdim=True) - x_mean * y_mean
    S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
    dist2 = (S2).sum(1, keepdim=True)

    score = 1 - (dist1 + dist2)
    return score
