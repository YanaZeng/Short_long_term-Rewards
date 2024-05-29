# -*- coding: utf-8 -*-
# import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
# import torch.nn.functional as F
# from math import sqrt
# import pdb
# import time
# from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, input_size, bias = False)
        self.linear_2 = torch.nn.Linear(input_size, input_size // 2, bias = False)
        self.linear_3 = torch.nn.Linear(input_size // 2, 1, bias = False)    
    
    def forward(self, x):
        
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)        
        x = self.sigmoid(x)
        
        return torch.squeeze(x)    
    

    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, 1, bias = True)
    
    def forward(self, x):
        
        x = self.linear_1(x)     
        x = self.sigmoid(x)
        
        return torch.squeeze(x)      
    
    def fit(self, x, y, num_epoch=1000, lr=0.01, lamb=0, tol=1e-4, batch_size = 20, verbose=True):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                
                sub_x = torch.Tensor(x[selected_idx])
                sub_y = torch.Tensor(y[selected_idx])
                
                pred = self.forward(sub_x)

                loss = nn.MSELoss()(pred, sub_y)
                
                optimizer.zero_grad()                
                loss.backward()
                optimizer.step()
                
                #epoch_loss += xent_loss.detach().numpy()
                epoch_loss += loss.detach().numpy()
                           
            if  epoch_loss > last_loss - tol:
                if early_stop > 5:
                    print("[IPS_model] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[IPS_model] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        x = torch.Tensor(x)
        x = self.forward(x)
        return x.detach().cpu().numpy()

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x)) 

class OR_model_YS(nn.Module):
    def __init__(self, input_size):
        super(OR_model_YS, self).__init__()
        self.input_size = input_size
        self.model = MLP(input_size = self.input_size)
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, a, s, y, e, r, est_r_1, est_r_0, mu0, mu1, bar_mu0, bar_mu1, tilde_mu0, tilde_mu1, lambda_sy, thr = -5, stop = 10, panelty = 500, num_epoch=1000, lr=0.01, lamb=0, tol=1e-4, batch_size = 20, verbose=True):
    # def fit(self, x, mu0, mu1, c, rho, harm_num, thr = -5, stop = 10, panelty = 500, num_epoch=1000, lr=0.01, lamb=0, tol=1e-4, batch_size = 20, verbose=True):
    # r is an indicator
    # est_r_1 is estimated probability P(R=1|X=x, A=1, S=s).  value
#         batch_size = len(x)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size
        early_stop = 0
        y[np.isnan(y)] = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = torch.Tensor(x[selected_idx])  #matrix
                sub_a = torch.Tensor(a.reshape(-1,1)[selected_idx])  #matrix
                sub_s = torch.Tensor(s[selected_idx])  #matrix
                sub_y = torch.Tensor(y[selected_idx])  #matrix
                #
                sub_e = torch.Tensor(e.reshape(-1,1)[selected_idx])   #
                sub_r = torch.Tensor(r[selected_idx])  #matrix
                sub_est_r1 = torch.Tensor(est_r_1.reshape(-1,1)[selected_idx])  #matrix
                sub_est_r0 = torch.Tensor(est_r_0.reshape(-1,1)[selected_idx])  #matrix
                # 
                sub_mu0 = torch.Tensor(mu0.reshape(-1,1)[selected_idx])  #vector
                sub_mu1 = torch.Tensor(mu1.reshape(-1,1)[selected_idx])  #vector
                sub_bar_mu0 = torch.Tensor(bar_mu0.reshape(-1,1)[selected_idx])   #vector
                sub_bar_mu1 = torch.Tensor(bar_mu1.reshape(-1,1)[selected_idx])   #vector
                sub_tilde_mu0 = torch.Tensor(tilde_mu0.reshape(-1,1)[selected_idx])  #vector
                sub_tilde_mu1 = torch.Tensor(tilde_mu1.reshape(-1,1)[selected_idx]) #vector
                                
                pred = self.model.forward(sub_x)  #Pred: predicted π(pi). how to obtain pi?? forwerdNN
                # print(pred)

                # estimated_Vs = torch.nanmean(pred.reshape(-1,1)*sub_mu1 + (1-pred.reshape(-1,1))*sub_mu0) #Eq.4 #- estimated_Vs \
                loss_S = torch.mean( - pred.reshape(-1,1)*sub_mu1 - (1-pred.reshape(-1,1))*sub_mu0 \
                                    - pred.reshape(-1,1)*sub_a*(sub_s-sub_mu1)/sub_e \
                                    - (1-pred.reshape(-1,1))*(1-sub_a)*(sub_s-sub_mu0)/(1-sub_e) )

                # estimated_Vy = torch.nanmean(pred.reshape(-1,1)*sub_tilde_mu1 + (1-pred.reshape(-1,1))*sub_tilde_mu0) #Proposition 4.3 - estimated_Vy
                # loss_Y = torch.nanmean(pred.reshape(-1,1)*sub_bar_mu1 + (1-pred.reshape(-1,1))*sub_bar_mu0 - estimated_Vy \
                # + pred.reshape(-1,1)*sub_a*sub_r*(sub_y-sub_tilde_mu1)/(sub_e*sub_est_r1) + pred.reshape(-1,1)*sub_a*(sub_tilde_mu1-sub_bar_mu1)/sub_e \
                # + (1-pred.reshape(-1,1))*(1-sub_a)*sub_r*(sub_y-sub_tilde_mu0)/((1-sub_e)*sub_est_r0) + (1-pred.reshape(-1,1))*(1-sub_a)*(sub_tilde_mu0-sub_bar_mu0)/(1-sub_e) )
                loss_Y = torch.nanmean( - pred.reshape(-1,1)*sub_bar_mu1 - (1-pred.reshape(-1,1))*sub_bar_mu0  \
                - pred.reshape(-1,1)*sub_a*(sub_tilde_mu1-sub_bar_mu1)/sub_e \
                - (1-pred.reshape(-1,1))*(1-sub_a)*(sub_tilde_mu0-sub_bar_mu0)/(1-sub_e) \
                - pred.reshape(-1,1)*sub_a*sub_r*(sub_y-sub_tilde_mu1)/(sub_e*sub_est_r1) \
                - (1-pred.reshape(-1,1))*(1-sub_a)*sub_r*(sub_y-sub_tilde_mu0)/((1-sub_e)*sub_est_r0)  )

                # eq4: loss_S + lambda*loss_Y
                loss = (1-lambda_sy)*loss_S + lambda_sy*loss_Y
                # print(loss_S, loss_Y)
                            
                optimizer.zero_grad()                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.detach().numpy()

            if epoch > 0.5 * (num_epoch):    
                # print("[OR_model] epoch:{}, xent:{}".format(epoch, epoch_loss))
                break   
            if  epoch_loss > last_loss - tol:
                if early_stop > stop:
                    # print("[OR_model] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            # if epoch % 10 == 0 and verbose:
            #     print("[OR_model] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        x = torch.Tensor(x)
        pred = self.model.forward(x)
        pred = pred.detach().numpy().flatten()
        pred = np.random.binomial(1, pred)
        return pred    



if __name__ == "__main__":

    trials = 50
    missing_set = [0.1]
    train_data = np.load('data/jobs_DW_bin.new.10.train.npz')
    train_data.files
    x_tr = train_data['x'][:, :, 0]  # 2570*17
    x_tr = (x_tr - np.mean(x_tr, axis = 0))/np.std(x_tr, axis = 0)
    N = x_tr.shape[0]

    mumu = [0] #controls y_tr's mean
    sigma = [0.5]
    allT = [10]
    for tt in range(len(allT)):
        T = allT[tt]
        for m in range(len(missing_set)): #
            for mm in range(len(mumu)):
                for ss in range(len(sigma)):
                    sub_mumu = mumu[mm]  
                    sub_sigma = sigma[ss]
                    # randomly missing long-term effects
                    missing_ratio = missing_set[m]
                    Reward_ours_all, Policy_ours_all, Welfare_ours_all = np.zeros((trials,9)), np.zeros((trials,9)), np.zeros((trials,9)) #3 methods

                    for j in range(1,trials+1):
                        
                        print("The", j, "th experiment:")
                        ################## prepare training data & estimands needed ################
                        # generate or upload the data
                        out_treat = np.loadtxt('data/NEWS/test/'+str(sub_mumu)+'_var'+str(sub_sigma)+'_beta_2_var1_T'+str(T)+'_' + str(j) + '.txt', delimiter=',')                        
                        a_tr = out_treat[:, 0] # treatment
                        s_tr = np.reshape(out_treat[:, 1], (N, 1)) # short-term effects
                        y_tr = np.reshape(out_treat[:, 2], (N, 1)) # long-term effects
                        R_tr = np.reshape(out_treat[:, 3], (N, 1)) # missing indicator: 0 missing; 1 not missing
                        s0_tr = np.reshape(out_treat[:, 4], (N, 1)) # below are groundtruth
                        s1_tr = np.reshape(out_treat[:, 5], (N, 1)) # 
                        y0_tr = np.reshape(out_treat[:, 6], (N, 1)) # 
                        y1_tr = np.reshape(out_treat[:, 7], (N, 1)) # 

                        missing_index = np.random.choice(N,round(N*missing_ratio))
                        y_tr[missing_index] = np.nan 
                        R_tr[missing_index] = 0

                        # construct propensity score
                        clf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
                        clf.fit(x_tr, a_tr)
                        p_tr = clf.predict_proba(x_tr)[:, 1] # size: N
                        p_tr = np.clip(p_tr, 0.1, 0.9)

                        # construct est_r_1
                        clf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
                        clf.fit(np.hstack((x_tr, s_tr, a_tr.reshape(-1,1))), np.squeeze(R_tr))
                        est_r_0 = clf.predict_proba(np.hstack((x_tr, s_tr, np.zeros([N,1]))))[:, 1] # size: N
                        est_r_0 = np.clip(est_r_0, 0.1, 0.9)
                        est_r_1 = clf.predict_proba(np.hstack((x_tr, s_tr, np.ones([N,1]))))[:, 1] # size: N
                        est_r_1 = np.clip(est_r_1, 0.1, 0.9)

                        # construct mu0, mu1
                        clf = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42) #logisticregression as well
                        clf.fit(x_tr[a_tr == 0], s_tr[a_tr == 0].ravel())
                        mu0_or = clf.predict(x_tr)
                        clf = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
                        clf.fit(x_tr[a_tr == 1], s_tr[a_tr == 1].ravel())
                        mu1_or = clf.predict(x_tr)

                        # construct bar_mu0, bar_mu1 
                        bar_index0, bar_index1 = [], []
                        for ii in range(N):
                            if R_tr[ii] == 1 and a_tr[ii] == 0:
                                bar_index0.append(ii)
                            if R_tr[ii] == 1 and a_tr[ii] == 1:
                                bar_index1.append(ii)
                        clf = MLPRegressor(hidden_layer_sizes=(100,), max_iter=100000, random_state=42)
                        clf.fit(x_tr[bar_index0], y_tr[bar_index0].ravel())
                        bar_mu0_or = clf.predict(x_tr)
                        clf = MLPRegressor(hidden_layer_sizes=(100,), max_iter=100000, random_state=42)
                        clf.fit(x_tr[bar_index1], y_tr[bar_index1].ravel())
                        bar_mu1_or = clf.predict(x_tr)

                        # construct tilde_mu0, tilde_mu1 
                        xs_tr = np.concatenate((x_tr, s_tr), axis=1)
                        clf = MLPRegressor(hidden_layer_sizes=(100,), max_iter=100000, random_state=42)
                        clf.fit(xs_tr[bar_index0], y_tr[bar_index0].ravel())
                        tilde_mu0_or = clf.predict(xs_tr)
                        clf = MLPRegressor(hidden_layer_sizes=(100,), max_iter=100000, random_state=42)
                        clf.fit(xs_tr[bar_index1], y_tr[bar_index1].ravel())
                        tilde_mu1_or = clf.predict(xs_tr)

                        #groundtruth of the optimal policy 
                        optimal_0, optimal_1, optimal_2 = np.full([N,1], np.nan), np.full([N,1], np.nan), np.full([N,1], np.nan)
                        optimal_0 = np.where(s1_tr-s0_tr >= 0, 1, 0) #lambda=0
                        optimal_1 = np.where(s1_tr-s0_tr+y1_tr-y0_tr >= 0, 1, 0) #lambda=0.5
                        optimal_2 = np.where(y1_tr-y0_tr >= 0, 1, 0)  #lambda=1
                        Welfare_0, Welfare_1, Welfare_2 = [], [], []
                        Welfare_0 = s1_tr - s0_tr
                        Welfare_1 = s1_tr + 0.5*y1_tr - s0_tr - 0.5*y0_tr
                        Welfare_2 = y1_tr - y0_tr


                        ################### 1: lambda_sy=0.5 ###################
                        OR_ours = OR_model_YS(input_size = x_tr.shape[1])
                        lambda_sy = 0.5
                        OR_ours.fit(x_tr, a_tr, s_tr, y_tr, p_tr, R_tr, est_r_1, est_r_0, mu1_or, mu0_or, bar_mu1_or, bar_mu0_or, tilde_mu1_or, tilde_mu0_or, lambda_sy, stop = 10, lr = 0.01, panelty = 10, num_epoch = 100, tol = 0.05, batch_size = len(a_tr), lamb = 1e-1)
                        pred_ours = OR_ours.predict(x_tr) #actions with estimated policy

                        # evaluate according to the reward defined (overall)
                        Reward_ours_all[j-1,0]= np.sum( pred_ours.reshape(-1,1) * s1_tr + (1-pred_ours.reshape(-1,1)) * s0_tr)
                        Reward_ours_all[j-1,3]= np.sum( pred_ours.reshape(-1,1)*(s1_tr-s0_tr+(y1_tr-y0_tr)) + s0_tr + y0_tr) 
                        Reward_ours_all[j-1,6]= np.sum( pred_ours.reshape(-1,1) * y1_tr + (1-pred_ours.reshape(-1,1)) * y0_tr)
                        Policy_ours_all[j-1,0] = np.sum((optimal_0-pred_ours.reshape(-1,1))**2)
                        Policy_ours_all[j-1,3] = np.sum((optimal_1-pred_ours.reshape(-1,1))**2)
                        Policy_ours_all[j-1,6] = np.sum((optimal_2-pred_ours.reshape(-1,1))**2)
                        Welfare_ours_all[j-1,0] = sum(Welfare_0[pred_ours == 1])
                        Welfare_ours_all[j-1,3] = sum(Welfare_1[pred_ours == 1])
                        Welfare_ours_all[j-1,6] = sum(Welfare_2[pred_ours == 1])


                        #################### 2: lambda_sy=0 ###################
                        OR_ours0 = OR_model_YS(input_size = x_tr.shape[1])
                        lambda_sy = 0
                        OR_ours0.fit(x_tr, a_tr, s_tr, y_tr, p_tr, R_tr, est_r_1, est_r_0, mu1_or, mu0_or, bar_mu1_or, bar_mu0_or, tilde_mu1_or, tilde_mu0_or, lambda_sy, stop = 10, lr = 0.01, panelty = 10, num_epoch = 100, tol = 0.05, batch_size = len(a_tr), lamb = 1e-1)
                        pred_ours0 = OR_ours0.predict(x_tr) #actions with estimated policy

                        # evaluate according to the reward defined (overall)
                        Reward_ours_all[j-1,1]= np.sum( pred_ours0.reshape(-1,1) * s1_tr + (1-pred_ours0.reshape(-1,1)) * s0_tr)
                        Reward_ours_all[j-1,4]= np.sum( pred_ours0.reshape(-1,1) * (s1_tr-s0_tr+(y1_tr-y0_tr)) + s0_tr + y0_tr)
                        Reward_ours_all[j-1,7]= np.sum( pred_ours0.reshape(-1,1) * y1_tr + (1-pred_ours0.reshape(-1,1)) * y0_tr)
                        Policy_ours_all[j-1,1] = np.sum((optimal_0-pred_ours0.reshape(-1,1))**2)
                        Policy_ours_all[j-1,4] = np.sum((optimal_1-pred_ours0.reshape(-1,1))**2)
                        Policy_ours_all[j-1,7] = np.sum((optimal_2-pred_ours0.reshape(-1,1))**2)
                        Welfare_ours_all[j-1,1] = sum(Welfare_0[pred_ours0 == 1])
                        Welfare_ours_all[j-1,4] = sum(Welfare_1[pred_ours0 == 1])
                        Welfare_ours_all[j-1,7] = sum(Welfare_2[pred_ours0 == 1])


                        #################### 3: lambda_sy=infinity ###################
                        OR_ours_inf = OR_model_YS(input_size = x_tr.shape[1])
                        lambda_sy = 1
                        OR_ours_inf.fit(x_tr, a_tr, s_tr, y_tr, p_tr, R_tr, est_r_1, est_r_0, mu1_or, mu0_or, bar_mu1_or, bar_mu0_or, tilde_mu1_or, tilde_mu0_or, lambda_sy, stop = 10, lr = 0.01, panelty = 10, num_epoch = 100, tol = 0.05, batch_size = len(a_tr), lamb = 1e-1)
                        pred_ours_inf = OR_ours_inf.predict(x_tr) #actions with estimated policy

                        # evaluate according to the reward defined (overall)
                        Reward_ours_all[j-1,2]= np.sum( pred_ours_inf.reshape(-1,1) * s1_tr + (1-pred_ours_inf.reshape(-1,1)) * s0_tr)
                        Reward_ours_all[j-1,5]= np.sum( pred_ours_inf.reshape(-1,1) * (s1_tr-s0_tr+(y1_tr-y0_tr)) + s0_tr + y0_tr) 
                        Reward_ours_all[j-1,8]= np.sum( pred_ours_inf.reshape(-1,1) * y1_tr + (1-pred_ours_inf.reshape(-1,1)) * y0_tr)
                        Policy_ours_all[j-1,2] = np.sum((optimal_0-pred_ours_inf.reshape(-1,1))**2)
                        Policy_ours_all[j-1,5] = np.sum((optimal_1-pred_ours_inf.reshape(-1,1))**2)
                        Policy_ours_all[j-1,8] = np.sum((optimal_2-pred_ours_inf.reshape(-1,1))**2)
                        Welfare_ours_all[j-1,2] = sum(Welfare_0[pred_ours_inf == 1])
                        Welfare_ours_all[j-1,5] = sum(Welfare_1[pred_ours_inf == 1])
                        Welfare_ours_all[j-1,8] = sum(Welfare_2[pred_ours_inf == 1])
                    
                        print("Time:", T)
                        print("missing ratio:", missing_set[m])
                        print("sub_mumu:", sub_mumu)
                        print("sub_sigma:", sub_sigma)
                        print("Reward mean:", Reward_ours_all.mean(axis=0))
                        print("Policy MSE mean:", Policy_ours_all.mean(axis=0))
                        print("Welfare mean:", Welfare_ours_all.mean(axis=0))
    print()
    print('Done.')