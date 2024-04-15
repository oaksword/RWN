import numpy as np
import torch
import torch.nn as nn
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

class DNN(nn.Module):
    
    def __init__(self, n_in, n_out, hs):
        
        super(DNN, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(n_in, hs),
                                nn.ReLU(),
                                nn.Linear(hs, hs // 2),
                                nn.ReLU(),
                                nn.Linear(hs // 2, hs // 4),
                                nn.ReLU(),
                                nn.Linear(hs // 4, hs // 2),
                                nn.ReLU(),
                                nn.Linear(hs // 2, n_out))
        
    def forward(self, x):
        
        return self.net(x)
    
def rf_graph(x):
        
    G = np.zeros((len(x), len(x)))
    
    for i in range(len(x)):
        
        G[i, np.where(x == x[i])[0]] = 1

    nodes = Counter(x)
    nodes_num = np.array([nodes[i] for i in x])
        
    return G, G / nodes_num.reshape(-1, 1)

    
def get_rfweight(rf, x):
    
    n = x.shape[0]
    
    leaf = rf.apply(x)
    ntrees = leaf.shape[1]
    G_unnorm = np.zeros((n, n))
    G_norm = np.zeros((n, n))
    
    for i in range(ntrees):
        
        tmp1, tmp2 = rf_graph(leaf[:, i])
        G_unnorm += tmp1
        G_norm += tmp2
    
    return G_unnorm / ntrees, G_norm / ntrees
    

def get_derivative_matrix(A, B, device='cpu'):
    
    n = A.shape[0]
    p = A.shape[1]
    
    C = torch.zeros(n, n * p).to(device)
    
    row_idx = torch.arange(n).repeat((p, 1)).T.reshape(-1)
    col_idx = torch.arange(n * p)
    
    C[row_idx, col_idx] = A.reshape(-1)
    
    mB = torch.tile(B.T, (n, 1)) - torch.tile(B.reshape(-1, 1), (1, n))
    mB = mB.to(device)
    
    return C @ mB


class RWN():
    
    def __init__(self, hs=64, device='cpu'):
        
        self.hs = hs
        self.device = device
        
    
    def fit(self, x, y, weight, tau=0.5, d=False, batch_size=100, n_iter=500, lr=1e-3, tol=1e-5, verbose=True):
        
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        weight = torch.FloatTensor(weight)
        
        n = x.shape[0]
        p = x.shape[1]
        
        self.fnet = DNN(p, 1, self.hs).to(self.device)
        self.dnet = DNN(p, p, self.hs).to(self.device)
        optimizer = torch.optim.Adam([
            {'params': self.fnet.parameters()},
            {'params': self.dnet.parameters()}
        ], lr = lr)
        
        mse_loss = nn.MSELoss()
        
        self.loss_count = []
        last_loss = np.inf
        flag = 0

        for i_iter in range(n_iter):
            
            csample = np.random.permutation(n)[:batch_size]
            tmp_x = x[csample].to(self.device)
            tmp_y = y[csample].to(self.device)
            tmp_w = weight[np.tile(csample, (batch_size, 1)).T.ravel(),
                           np.tile(csample, (1, batch_size))].reshape(batch_size, -1).T.to(self.device)
            tmp_w = tmp_w if tau is None else tmp_w - torch.diag(torch.diag(tmp_w))

            tmp_fx = self.fnet(tmp_x)
            tmp_my = torch.tile(tmp_y, (batch_size, 1))
            tmp_mfx = torch.tile(tmp_fx, (1, batch_size))
            
            if d:
                
                tmp_dfx = self.dnet(tmp_x)
                tmp_mdfx = get_derivative_matrix(tmp_dfx, tmp_x, self.device)
                
            else:
                
                tmp_mdfx = 0
            
            if tau is None:
                
                loss = torch.mean((tmp_my - tmp_mfx - tmp_mdfx) ** 2 * tmp_w)
            
            else:
                
                loss1 = mse_loss(tmp_y, tmp_fx.ravel())
                loss2 = torch.mean((tmp_my - tmp_mfx - tmp_mdfx) ** 2 * tmp_w) * n / (n - 1)
                loss = tau * loss1 + (1 - tau) * loss2
            
            self.loss_count.append(loss.data.cpu().tolist())

            if (np.abs(last_loss - loss.data.cpu().numpy()) <= tol) & (i_iter >= 100):
                
                if verbose:

                    print('Algorithm converges for RWN model at iter {}, loss: {}.'.format(i_iter, self.loss_count[-1]))
                    
                flag = 1
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.data.cpu().numpy()

        if (flag == 0) & verbose:

            print('Algorithm may not converge for RWN model, loss: {}.'.format(self.loss_count[-1]))
            
    
    def predict(self, x_new):
        
        x_new = torch.FloatTensor(x_new)
        x_new = x_new.reshape(-1, 1) if x_new.ndim == 1 else x_new
        
        return self.fnet.cpu()(x_new).data.numpy().ravel()
    
    def predict_derivative(self, x_new):
        
        x_new = torch.FloatTensor(x_new)
        x_new = x_new.reshape(-1, 1) if x_new.ndim == 1 else x_new
        
        return self.dnet.cpu()(x_new).data.numpy()