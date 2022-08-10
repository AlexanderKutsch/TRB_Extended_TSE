import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import linregress
import copy
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


############################
# NN 
############################

class Net(nn.Module):
    def __init__(self,hparams):
        super(Net, self).__init__()
        self.hparams = hparams
        neurons = self.hparams['neurons']
        modules = []
        # 1. first layer
        hl = nn.Linear(hparams['input_size'], neurons) 
        if hparams['inclKaiming']==1:
            torch.nn.init.kaiming_normal_(hl.weight)
        torch.nn.init.zeros_(hl.bias)
        modules.append(hl)
        # 2. other layers
        for l in range(self.hparams['hidden_layers']-1):
            hl = nn.Linear(neurons, neurons)
            if hparams['inclKaiming']==1:
                torch.nn.init.kaiming_normal_(hl.weight)
            torch.nn.init.zeros_(hl.bias)
            modules.append(hparams['activation_function'])
            modules.append(hl)
        # 3. last layer
        self.oupt = nn.Linear(neurons,hparams['output_size'])
        torch.nn.init.kaiming_normal_(self.oupt.weight)
        torch.nn.init.zeros_(self.oupt.bias)
        modules.append(self.oupt)
        # 4. gather all
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"])

############################
# MLR 
############################

class LinearNet(nn.Module):
    def __init__(self,hparams):
        super(LinearNet, self).__init__()
        self.hparams = hparams
        self.linear = torch.nn.Linear(hparams['input_size'],hparams['output_size'])
        self.model = nn.Sequential(self.linear)
    def forward(self, x):
        return self.model(x)
    def configure_optimizer(self,lr):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

############################
# Dataloader 
############################

class myDataLoader(torch.utils.data.Dataset):
    def __init__(self, x, y):
        # remove nans i.e. time in between experiments
        mask = ~np.isnan(x).any(axis=1) & ~np.isnan(y).any(axis=1)
        x = x[mask]
        y = y[mask]
        self.idcs = np.where(mask)[0].tolist()
        # convert to torch
        self.x_data = torch.tensor(x).to(dtype=torch.float32)
        self.y_data = torch.tensor(y).to(dtype=torch.float32) # .flatten() # flatten if only one output
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        xs = self.x_data[idx,:] 
        ys = self.y_data[idx,:]
        return (xs, ys)       
    def getall(self):
        return (self.x_data, self.y_data)  
    def getIdcs(self):
        return self.idcs     

############################
# NNFD LOSS FUNCTIONS
############################    

# NNFD loss
def loss_function_NNFD(y_train,preds,f,FDparams,ax=0,scaleFDLoss=1):
    # values used for each FDType
    # D --> mid, -, slopeL, slopeR
    # T --> all
    # G --> mid, -, slopeL, slopeR --> indirectly for kjam
    FDType,mid,mid2,slopeL,slopeR,qdist = FDparams
    
    if FDType =='D':   # triangle
        loss = FDLossNNFD_D(y_train,preds,f,mid,slopeL,slopeR,scaleFDLoss,ax,dist=qdist)
    elif FDType =='D_Extended': # triangle with enriched info
        loss = FDLossNNFD_D_Extended(y_train,preds,f,mid,slopeL,slopeR,scaleFDLoss,ax,dist=qdist)
    elif FDType =='T': # trapezoidal
        loss = FDLossNNFD_T(y_train,preds,f,mid,mid2,slopeL,slopeR,scaleFDLoss,dist=qdist)
    elif FDType =='G': # greenshield
        loss = FDLossNNFD_G(y_train,preds,f,mid,mid2,slopeL,slopeR,scaleFDLoss,dist=qdist)
    else:
        print("Please set the FD type to 'D'.")
    
    return loss.mean()

# NNFD loss - Triangle
def FDLossNNFD_D(y_train,preds,f,mid,slopeL,slopeR,scaleFDLoss,ax=0,dist=0):
    y_train_q = y_train.detach().numpy()[:,0] 
    y_train_k = y_train.detach().numpy()[:,1] 
    qq = preds.detach().numpy()[:,0]
    kk = preds.detach().numpy()[:,1]
    
    # 1. idcs - Actual/Pred Left/Right Idcs Inside (below) FD curve
    ALIdcsI = np.where((y_train_k<mid) & (dist + slopeL*y_train_k > y_train_q))[ax]
    PLIdcsI = np.where((kk       <mid) & (dist + slopeL*kk        > qq       ))[ax]
    ARIdcsI = np.where((y_train_k>=mid) & ((dist + slopeL*mid+(y_train_k-mid)*slopeR) > y_train_q))[ax]
    PRIdcsI = np.where((kk       >=mid) & ((dist + slopeL*mid+(kk       -mid)*slopeR) > qq       ))[ax]
    PLIdcs = np.where((kk<mid))[ax]
    PRIdcs = np.where((kk>=mid))[ax]
    
    # 2. filtering factors --> all but (PredictedOutside&ActualOutside) = 0; all other values become gamma
    leftfs = np.full((max(kk.shape),2), f)
    leftfs[ALIdcsI,:] = 0
    leftfs[PLIdcsI,:] = 0
    lfs = torch.Tensor(leftfs[PLIdcs])
    rightfs = np.full((max(kk.shape),2), f)
    rightfs[ARIdcsI,:] = 0
    rightfs[PRIdcsI,:] = 0
    rfs = torch.Tensor(rightfs[PRIdcs])
    
    # 3. calculate losses --> regular*(1-factor)+FD*factor
    # NOTE: WHEN CHANGING 'MAE' TO  'MSE', MAKE SURE TO CHANGE REG, lFD, AND rFD FUNCTIONs
    power = 2 # 1 if MAE, 2 if MSE
    Reg = abs(y_train-preds)**power
    (q_train, k_train) = y_train.split(split_size=1, dim=1)
    (q_pred, k_pred) = preds.split(split_size=1, dim=1)
    # 3.b. right
    rFDq   = abs(  (slopeL*mid+dist+(k_pred-mid)*slopeR)  -  q_pred  )**power 
    rFDk   = abs(  (mid+(q_pred-slopeL*mid-dist)/slopeR)  -  k_pred  )**power 
    rFD = torch.cat((rFDq,rFDk),1)
    rLoss = Reg[PRIdcs]*(1-rfs) + rfs*rFD[PRIdcs]*scaleFDLoss              # replace regLoss
    # 3.a. left
    lFDq  = abs(  (slopeL*k_pred+dist)  -  q_pred   )**power  
    lFDk  = abs(  (q_pred-dist)/slopeL  -  k_pred   )**power  
    lFD = torch.cat((lFDq,lFDk),1)
    lLoss = Reg[PLIdcs]*(1-lfs) + lfs*lFD[PLIdcs]*scaleFDLoss              # replace regLoss
    
    # 4. concatenate losses
    return torch.cat([lLoss,rLoss])

def FDLossNNFD_D_Extended(y_train,preds,f,mid,slopeL,slopeR,scaleFDLoss,ax=0,dist=0):
    # first column: flow, second column: density
    y_train_q = y_train.detach().numpy()[:,0] #x-axis
    y_train_k = y_train.detach().numpy()[:,1] #y-axis
    qq = preds.detach().numpy()[:,0]
    kk = preds.detach().numpy()[:,1]
    
    # 1. idcs - Actual/Pred Left/Right Idcs Inside (below) FD curve
    ALIdcsI = np.where((y_train_k<mid) & (dist + slopeL*y_train_k > y_train_q))[ax]
    PLIdcsI = np.where((kk       <mid) & (dist + slopeL*kk        > qq       ))[ax]
    ARIdcsI = np.where((y_train_k>=mid) & ((dist + slopeL*mid+(y_train_k-mid)*slopeR) > y_train_q))[ax]
    PRIdcsI = np.where((kk       >=mid) & ((dist + slopeL*mid+(kk       -mid)*slopeR) > qq       ))[ax]
    PLIdcs = np.where((kk<mid))[ax]
    PRIdcs = np.where((kk>=mid))[ax]
    
    # 2. filtering factors --> all but (PredictedOutside&ActualOutside) = 0; all other values become gamma
    leftfs = np.full((max(kk.shape),2), f)
    leftfs[ALIdcsI,:] = 0
    leftfs[PLIdcsI,:] = 0
    lfs = torch.Tensor(leftfs[PLIdcs])
    rightfs = np.full((max(kk.shape),2), f)
    rightfs[ARIdcsI,:] = 0
    rightfs[PRIdcsI,:] = 0
    rfs = torch.Tensor(rightfs[PRIdcs])
    
    # 3. calculate losses --> regular*(1-factor)+FD*factor
    # NOTE: WHEN CHANGING 'MAE' TO  'MSE', MAKE SURE TO CHANGE REG, lFD, AND rFD FUNCTIONs
    power = 2 # 1 if MAE, 2 if MSE
    Reg = abs(y_train[:,:2]-preds[:,:2])**power #for every value within the Tensor
    (q_train, k_train,stops_train, lanecs_train) = y_train.split(split_size=1, dim=1)
    (q_pred, k_pred, stops_pred, lanecs_pred) = preds.split(split_size=1, dim=1)
    # 3.b. right
    rFDq   = abs(  (slopeL*mid+dist+(k_pred-mid)*slopeR)  -  q_pred  )**power # for corresponding q
    rFDk   = abs(  (mid+(q_pred-slopeL*mid-dist)/slopeR)  -  k_pred  )**power #for corresponding k
    rFD = torch.cat((rFDq,rFDk),1)
    rLoss = Reg[PRIdcs]*(1-rfs) + rfs*rFD[PRIdcs]*scaleFDLoss              # replace regLoss
    # 3.a. left
    lFDq  = abs(  (slopeL*k_pred+dist)  -  q_pred   )**power  
    lFDk  = abs(  (q_pred-dist)/slopeL  -  k_pred   )**power  
    lFD = torch.cat((lFDq,lFDk),1)
    lLoss = Reg[PLIdcs]*(1-lfs) + lfs*lFD[PLIdcs]*scaleFDLoss              # replace regLoss
    
    # 4. concatenate losses
    qk_loss = torch.cat([lLoss,rLoss])
    Reg_sl = abs(y_train[:,-2:]-preds[:,-2:])**power
    return torch.cat((qk_loss,Reg_sl),1)




############################
# OTHER FUNCTIONS
############################    

# plot loss curves of networks for q and k
def plot_losses(losses,lr,target): 
    # plot loss curve
    losses = np.array(losses)
    try:
        plt.plot(losses[:,0],losses[:,3],label='q_mse_train_%s'%(lr))
        plt.plot(losses[:,0],losses[:,4],label='q_mse_val_%s'%(lr))
        plt.plot(losses[:,0],losses[:,1],label='k_mse_train_%s'%(lr))
        plt.plot(losses[:,0],losses[:,2],label='k_mse_val_%s'%(lr))
    except:
        plt.plot(losses[:,0],losses[:,1],label='%s_mse_train_%s'%(target[0],lr))
        plt.plot(losses[:,0],losses[:,2],label='%s_mse_val_%s'%(target[0],lr))
    plt.title(target)
    plt.legend()
    plt.show()
    
def accuracy(model, dataset):
    n = len(dataset)
    targets = dataset[0:n][1]
    with torch.no_grad():
        preds = model(dataset[0:n][0])
    loss = torch.nn.functional.mse_loss(preds, targets)
    acc = (1 - (loss / targets.mean()))*100
    return acc,loss

#Calculating the error values for all data with implemented functions 
def calc_error_values(p,t,verbose='off'):
    avg_rmse = np.sqrt(np.nanmean(((p-t)**2),axis=0))
    avg_mae = np.nanmean(abs(p-t),axis=0)
    avgtargets = np.nanmean(t,axis=0)
    avg_mape = [np.nanmean((abs((p[t[:,i]>1,i]-t[t[:,i]>1,i])/t[t[:,i]>1,i]))) for i in range(t.shape[1]) ]
    r2 = []
    # With R2 as correlation coefficient
    for i in range(t.shape[1]):
        mask = ~np.isnan(p[:,i]) & ~np.isnan(t[:,i])
        _,_,r_value,_,_ = linregress(p[mask,i],t[mask,i])
        r2.append(r_value*r_value)
    if verbose=='on':
        print('Loss MAE\n',avg_mae)#/avgtargets
        print('Loss MAPE\n',avg_mape)
        print('Loss RMSE\n',avg_rmse)#/avgtargets
        print('Loss R2\n',r2)
    return [avg_mae.tolist(),avg_mape,avg_rmse.tolist(),r2]

# Calculate error values using the default functions from sklearn.metrics
def calc_error_values_default(p,t,verbose='off'):
    avg_rmse = [0]*t.shape[1]
    avg_mae = [0]*t.shape[1]
    avg_mape = [0]*t.shape[1]
    r2 = [0]*t.shape[1]
    for i in range(t.shape[1]):
        mask = ~np.isnan(p[:,i]) & ~np.isnan(t[:,i])
        avg_rmse[i] = mean_squared_error(t[mask,i],p[mask,i])**0.5
        avg_mae[i] = mean_absolute_error(t[mask,i],p[mask,i])
        avg_mape[i] = mean_absolute_percentage_error(t[mask,i],p[mask,i])
        r2[i] = r2_score(t[mask,i],p[mask,i])
    if verbose=='on':
        print('Loss MAE\n',avg_mae)#/avgtargets
        print('Loss MAPE\n',avg_mape)
        print('Loss RMSE\n',avg_rmse)#/avgtargets
        print('Loss R2\n',r2)
    return [avg_mae,avg_mape,avg_rmse,r2]


# make qk plot of predictions and plot the FD curve
def plot_preds(alltrues,allpreds,mae,rmse,output_sensor_cols,FDparams,ff,scaleFDLoss,save='off'):
    for i in range(0,len(output_sensor_cols),2):
        trues_q = alltrues[i].detach().numpy()
        preds_q = allpreds[i]#.detach().nmupy()
        trues_k = alltrues[i+1].detach().numpy()
        preds_k = allpreds[i+1]#.detach().nmupy()
        plt.scatter(trues_k,trues_q,label='Observed',alpha=0.7,s=4,color='grey')
        plt.scatter(preds_k,preds_q,label='Pred',alpha=0.7,s=4,color='blue')
        plt.title(r'MLR: density vs flow at $\gamma$ = %s'%(ff))
        if ('all' in output_sensor_cols[i][2:]):
            #mid,slopeL,slopeR,end = 4,2.2,-0.4,25 #0.15,4,-0.5,1
            FDType,mid,mid2,slopeL,slopeR,qdist = FDparams
            end = max(trues_k)
            if FDType=='D':
                x1 = np.linspace(0,mid,20)
                y1 = slopeL*x1
                y1out = slopeL*x1+qdist
                x2 = np.linspace(mid,end,20)
                y2 = slopeL*mid+(x2-mid)*slopeR
                y2out = slopeL*mid+(x2-mid)*slopeR+qdist
                plt.plot(x1,y1,color='dimgray')
                plt.plot(x2,y2,color='dimgray')
                plt.plot(x1,y1out,color='black')
                plt.plot(x2,y2out,color='black')
            elif FDType=='T':
                x1 = np.linspace(0,mid,20)
                y1 = slopeL*x1
                x2 = np.linspace(mid2,end,20)
                y2 = slopeL*mid+(x2-mid2)*slopeR
                plt.plot(x1,y1)
                plt.plot(np.linspace(mid,mid2,4),np.linspace(slopeL*mid,slopeL*mid,4))
                plt.plot(x2,y2)
            elif FDType=='G':
                vfree = slopeL
                kjam = mid - slopeL*mid/slopeR # x intercept of slope R
                x1 = np.linspace(0,end,20)
                y1 = vfree*x1*(1-(x1/kjam))
                plt.plot(x1,y1)
            else:
                print("Please set the FD type to 'D', 'T', or 'G'.")
        plt.legend()
        plt.xlabel('Density (k) of %s'%output_sensor_cols[i][2:])
        plt.ylabel('Flow (q) of %s'%output_sensor_cols[i][2:])
        if save=='on':
            plt.savefig('NN_MLR_preds_%s.png'%(output_sensor_cols[0+i][2:]))
        plt.show()
        