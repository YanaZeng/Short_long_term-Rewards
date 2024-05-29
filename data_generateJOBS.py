import numpy as np
# import math

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))




N = 2570
cov_dim = 17
values = [0,1,2,3,4]
probabilities=[0.8,0.05,0.05,0.05,0.05]
R = np.ones(N) #missing_indicator variable

mumu = [0] #controls y_tr's mean
sigma = [0.5]
allT = [10]


'''Data generation follow Bayesian NonParameteric modeling'''
for tt in range(len(allT)):
    T = allT[tt]
    for mm in range(len(mumu)):
        for ss in range(len(sigma)): 
            sub_mumu = mumu[mm]
            sub_sigma = sigma[ss]

            for i in range(1,51):
                train_data = np.load('data/jobs_DW_bin.new.10.train.npz')
                train_data.files
                x_tr = train_data['x'][:, :, 0]  # 2570*17
                t_tr = train_data['t'][:, 0]  # 237个“1” /2570
                x_tr = (x_tr - np.mean(x_tr, axis = 0))/np.std(x_tr, axis = 0)
                
                # generate short-term effects
                w0 = np.clip(np.random.normal(0, 1, x_tr.shape[1]), -1, 1) #[-1,1]
                w1 = 2 * np.random.random(x_tr.shape[1]) - 1 #[-1,1]
                s0_tr = np.random.binomial(1, sigmoid(np.sum(w0*x_tr, axis = 1) + np.random.normal(0, 1, x_tr.shape[0])))
                s1_tr = np.random.binomial(1, sigmoid(np.sum(w1*x_tr, axis = 1) + np.random.normal(2, 1, x_tr.shape[0])))
                # print(sum(s0_tr))
                s_tr = np.where(t_tr == 0, s0_tr, s1_tr)

                # generate long-term effects
                y0_tr_full, y1_tr_full = np.zeros((N,T)), np.zeros((N,T)) # all time steps' effects
                y0_tr_full[:,0] = s0_tr
                y1_tr_full[:,0] = s1_tr
                for t in range(1,T): #time series genrating the data: Long-term outcomes
                    beta1 = np.random.choice(values, cov_dim, p=probabilities) #[0,4]
                    beta2 = np.clip(4 * np.random.normal(0, 1, x_tr.shape[1]), 0, 4)  # [0,4]
                    if t == 1:
                        C1 = 0.02
                        y0_tr_full[:,t] = np.random.binomial(1, sigmoid(np.sum(beta1*x_tr, axis = 1)+C1*y0_tr_full[:, 0])) + np.random.normal(0, 1, x_tr.shape[0])
                        y1_tr_full[:,t] = np.random.binomial(1, sigmoid(np.sum(beta2*x_tr, axis = 1)+C1*y1_tr_full[:, 0])) + np.random.normal(sub_mumu, sub_sigma, x_tr.shape[0])
                    else:
                        y0_tr_full[:,t] = np.random.binomial(1, sigmoid(np.sum(beta1*x_tr, axis = 1)+C1/t*np.sum(y0_tr_full[:, 0:t-1],axis=1))) + np.random.normal(0, 1, x_tr.shape[0])
                        y1_tr_full[:,t] = np.random.binomial(1, sigmoid(np.sum(beta2*x_tr, axis = 1)+C1/t*np.sum(y1_tr_full[:, 0:t-1],axis=1))) + np.random.normal(sub_mumu, sub_sigma, x_tr.shape[0])   
                #choose the last time step to be data
                y0_tr, y1_tr = y0_tr_full[:,T-1], y1_tr_full[:,T-1] 
                
                y_tr = np.where(t_tr == 0, y0_tr, y1_tr)

                # concatenate the data
                data = np.concatenate((t_tr.reshape(-1,1), s_tr.reshape(-1,1), y_tr.reshape(-1,1), R.reshape(-1,1), s0_tr.reshape(-1,1), s1_tr.reshape(-1,1), y0_tr.reshape(-1,1), y1_tr.reshape(-1,1)),axis=1) # data: N*4
                np.savetxt('data/NEWS/test/'+str(sub_mumu)+'_var'+str(sub_sigma)+'_beta_2_var1_T'+str(T)+'_'+str(i)+'.txt', data, delimiter=',', fmt='%.2f')
print("Done ")