import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

N = 747
values = [0,1,2,3,4]
probabilities=[0.5,0.2,0.15,0.1,0.05]
C1 = 0.02 
allT = [10]

R = np.ones(N) #missing_indicator variable

mumu = [2] 
sigma = [0.5] 


'''Data generation follow Bayesian NonParameteric modeling'''
for tt in range(len(allT)):
    T = allT[tt]
    for mm in range(len(mumu)):
        for ss in range(len(sigma)): 
            sub_mumu = mumu[mm]
            sub_sigma = sigma[ss]
            for i in range(1,51):

                if i%10 == 0:
                    file_name = np.loadtxt('data/IHDP/csv/ihdp_npci_'+str(10)+'.csv',delimiter=',')
                else:
                    file_name = np.loadtxt('data/IHDP/csv/ihdp_npci_'+str(i%10)+'.csv',delimiter=',')
                t_tr = file_name[:,0]
                x_tr = file_name[:,5:]  #covariates
                x_tr = (x_tr - np.mean(x_tr, axis = 0))/np.std(x_tr, axis = 0)
                
                # generate short-term effects
                w0 = np.clip(np.random.normal(0, 1, x_tr.shape[1]), -1, 1) #[-1,1]
                w1 = 2 * np.random.random(x_tr.shape[1]) - 1 #[-1,1]
                s0_tr = np.random.binomial(1, sigmoid(np.sum(w0*x_tr, axis = 1) + np.random.normal(1, 1, x_tr.shape[0])))
                s1_tr = np.random.binomial(1, sigmoid(np.sum(w1*x_tr, axis = 1) + np.random.normal(3, 1, x_tr.shape[0])))
                # print(sum(s0_tr))
                s_tr = np.where(t_tr == 0, s0_tr, s1_tr)

                # generate long-term effects
                y0_tr_full, y1_tr_full = np.zeros((N,T)), np.zeros((N,T)) # all time steps' effects

                y0_tr_full[:,0] = s0_tr
                y1_tr_full[:,0] = s1_tr
                #generate uncorrelated and correlated data
                for t in range(1,T): #time series genrating the data: Long-term outcomes
                    beta1 = np.random.choice(values, 25, p=probabilities) #[0,4]
                    beta2 = np.clip(4 * np.random.normal(0, 1, x_tr.shape[1]), 0, 4)   # [0,4]
                    y0_tr_full[:,t] = np.random.normal(np.dot(x_tr,beta1), 1, N) + C1 * np.sum(y0_tr_full[:, 0:t-1],axis=1) # all last time steps
                    y1_tr_full[:,t] = np.random.normal(np.dot(x_tr,beta2)+sub_mumu, sub_sigma, N) + C1 * np.sum(y1_tr_full[:, 0:t-1],axis=1)
                       
                #choose the last time step to be data
                y0_tr, y1_tr = y0_tr_full[:,T-1], y1_tr_full[:,T-1]
                y_tr = np.where(t_tr == 0, y0_tr, y1_tr)

                # concatenate the data
                data = np.concatenate((t_tr.reshape(-1,1), s_tr.reshape(-1,1), y_tr.reshape(-1,1), R.reshape(-1,1), s0_tr.reshape(-1,1), s1_tr.reshape(-1,1), y0_tr.reshape(-1,1), y1_tr.reshape(-1,1)),axis=1) # data: N*4
                np.savetxt('data/IHDP/'+str(sub_mumu)+'_var'+str(sub_sigma)+'_beta_3_var1_T'+str(T)+'_'+str(i)+'.txt', data, delimiter=',', fmt='%.2f')
print("Finished generating the data!! ")