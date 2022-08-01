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

# NNFD loss - Trapezoidal
def FDLossNNFD_T(y_train,preds,f,mid,mid2,slopeL,slopeR,scaleFDLoss,dist=0):
    y_train_q = y_train.detach().numpy()[:,0]
    y_train_k = y_train.detach().numpy()[:,1]
    qq = preds.detach().numpy()[:,0]
    kk = preds.detach().numpy()[:,1]
    
    # 1. idcs - Actual/Pred Left/Right Idcs Inside (below) FD curve
    ALIdcsI = np.where((y_train_k<mid)   & (dist + slopeL*y_train_k > y_train_q))[0]
    PLIdcsI = np.where((kk       <mid)   & (dist + slopeL*kk        > qq       ))[0]
    AMIdcsI = np.where((y_train_k>=mid)  & (y_train_k<mid2) & ((dist + slopeL*mid) > y_train_q))[0]
    PMIdcsI = np.where((kk       >=mid)  & (kk       <mid2) & ((dist + slopeL*mid) > qq       ))[0]
    ARIdcsI = np.where((y_train_k>=mid2) & ((dist + slopeL*mid+(y_train_k-mid2)*slopeR) > y_train_q))[0]
    PRIdcsI = np.where((kk       >=mid2) & ((dist + slopeL*mid+(kk       -mid2)*slopeR) > qq       ))[0]
    PLIdcs  = np.where((kk<mid))[0]
    PMIdcs  = np.where((kk>=mid) & (kk<mid2))[0]
    PRIdcs  = np.where((kk>=mid2))[0]

    # 2. filtering factors --> all but (PredictedOutside&ActualOutside) = 0
    leftfs = np.full((max(kk.shape),2), f)
    leftfs[ALIdcsI,:] = 0
    leftfs[PLIdcsI,:] = 0
    lfs = torch.Tensor(leftfs[PLIdcs])
    midfs = np.full((max(kk.shape),2), f)
    midfs[AMIdcsI,:] = 0
    midfs[PMIdcsI,:] = 0
    mfs = torch.Tensor(midfs[PMIdcs])
    rightfs = np.full((max(kk.shape),2), f)
    rightfs[ARIdcsI,:] = 0
    rightfs[PRIdcsI,:] = 0
    rfs = torch.Tensor(rightfs[PRIdcs])
    
    # 3. calculate losses --> regular*(1-factor)+FD*factor
    power = 2 # 1 if MAE, 2 if MSE
    Reg = abs(y_train-preds)**power
    (q_train, k_train) = y_train.split(split_size=1, dim=1)
    (q_pred, k_pred) = preds.split(split_size=1, dim=1)
    # 3.a. left
    lFDq  = abs(  (slopeL*k_pred+dist)   -  q_pred  )**power  
    lFDk  = abs(  (q_pred-dist)/slopeL   -  k_pred  )**power 
    lFD = torch.cat((lFDq,lFDk),1)
    lLoss = Reg[PLIdcs]*(1-lfs) + lfs*lFD[PLIdcs]*scaleFDLoss       # replace regLoss
    # 3.b. mid
    mFDq  = abs(  (slopeL*mid+dist)   -  q_pred  )**power  
    mFDk  = torch.zeros(mFDq.size())                                                      # no change needed for k, slope horizontal
    mFD = torch.cat((mFDq,mFDk),1)
    mLoss = Reg[PMIdcs]*(1-mfs) + mfs*mFD[PMIdcs]*scaleFDLoss       # replace regLoss
    # 3.c. right
    rFDq   = abs(  (slopeL*mid+dist+(k_pred-mid2)*slopeR)  -  q_pred   )**power 
    rFDk   = abs(  (mid2+(q_pred-slopeL*mid-dist)/slopeR)  -  k_pred   )**power
    rFD = torch.cat((rFDq,rFDk),1)
    rLoss = Reg[PRIdcs]*(1-rfs) + rfs*rFD[PRIdcs]*scaleFDLoss       # replace regLoss
    
    # 4. concatenate losses
    return torch.cat([lLoss,mLoss,rLoss])

# NNFD loss - Greenshield
def FDLossNNFD_G(y_train,preds,f,mid,mid2,slopeL,slopeR,scaleFDLoss,dist=0):
    y_train_q = y_train.detach().numpy()[:,0]
    y_train_k = y_train.detach().numpy()[:,1]
    qq = preds.detach().numpy()[:,0]
    kk = preds.detach().numpy()[:,1]
    vfree    = slopeL
    kjam     = mid - slopeL*mid/slopeR # x intercept of slope R
    qmax     = vfree*kjam*0.25
    
    # 1. idcs - Actual/Pred Left/Right Idcs Inside (below) FD curve
    # any q values > qmax
    ATIdcsO  = np.where(y_train_q-dist>=qmax)[0]
    PTIdcsO  = np.where(qq-dist>=qmax)[0]
    PTIdcs   = np.where(qq-dist>=qmax)[0]
    # all other values
    ABIdcsO  = np.where((dist + vfree*y_train_k*(1-(y_train_k/kjam)) < y_train_q) & (y_train_q-dist<qmax))[0]
    PBIdcsO  = np.where((dist + vfree*kk       *(1-(kk       /kjam)) < qq       ) & (qq-dist<qmax))[0]
    PBIdcs   = np.where(qq-dist<qmax)[0]
    
    # 2. filtering factors --> all but (PredictedOutside&ActualOutside) = 0
    Topfs    = np.full((max(kk.shape),2), float(0))
    Topfs[[i for i in ATIdcsO if i in PTIdcsO],:] = float(f)
    tfs      = torch.Tensor(Topfs[PTIdcs])
    Bottomfs = np.full((max(kk.shape),2), float(0))
    Bottomfs[[i for i in ABIdcsO if i in PBIdcsO],:] = float(f)
    bfs      = torch.Tensor(Bottomfs[PBIdcs])
    
    # 3. calculate losses --> regular*(1-factor)+FD*factor
    power = 2 # 1 if MAE, 2 if MSE
    Reg = abs(y_train-preds)**power
    (q_train, k_train) = y_train.split(split_size=1, dim=1)
    (q_pred, k_pred) = preds.split(split_size=1, dim=1)
    
    # 3.a. top - qq>qmax=vfree*kjam/4, so FDk=kmid=kjam/2
    tFDq  = abs(  (vfree*k_pred*(1-(k_pred/kjam)))   -  q_pred )**power  
    tFDk  = abs(  (kjam*0.5)                         -  k_pred )**power  
    tFD = torch.cat((tFDq,tFDk),1)
    tLoss = Reg[PTIdcs]*(1-tfs) + tfs*tFD[PTIdcs]*scaleFDLoss       # replace regLoss
    # 3.b. bottom - calc. both density values for that q values (bc of plusminus). Then use the smaller loss.
    with torch.no_grad(): # ok to use detached versions bc just want values
        discriminant = (vfree*vfree+4*qq*vfree/kjam)**(0.5) # b2-4ac
        lplus  = ((kk - (vfree + discriminant)/(2*vfree/kjam) )**2)
        lminus = ((kk - (vfree - discriminant)/(2*vfree/kjam) )**2)
        # use sign of closer k point on FD
        plusminus = torch.tensor([-1 if (p==p and m==m and p>m) else 1 for p,m in zip(lplus,lminus)]).reshape(-1,1)
        # so values above curve are not Nan (this step done for ALL points). Filtered later.
        removeaboves = torch.tensor([1 if (p==p and m==m) else 0 for p,m in zip(lplus,lminus)]).reshape(-1,1)
    bFDq   = abs(  (vfree*k_pred*(1-(k_pred/kjam)))                                          -  q_pred )**power 
    bFDk   = removeaboves*abs(  (vfree+plusminus*(vfree*vfree+removeaboves*4*q_pred*vfree/kjam)**(0.5))/(2*vfree/kjam) -  k_pred )**power
    bFD = torch.cat((bFDq,bFDk),1)
    bLoss = Reg[PBIdcs]*(1-bfs) + bfs*bFD[PBIdcs]*scaleFDLoss       # replace regLoss
    
    # 4. concatenate losses
    return torch.cat([bLoss,tLoss])

############################
# MLRFD LOSS FUNCTIONS
############################    

# MLRFD loss
def loss_function_MLRFD(y_train_k,y_train_q,kk,qq,fk,fq,FDparams,ax=0,scaleFDLoss=1,qdist=0):
    # values used for each FDType
    # D --> mid, -, slopeL, slopeR
    # T --> all
    # G --> mid, -, slopeL, slopeR --> indirectly for kjam
    FDType,mid,mid2,slopeL,slopeR,qdist = FDparams
    
    if FDType =='D':
        loss_k,loss_q = FDLossMLRFD_D(y_train_k,y_train_q,kk,qq,fk,mid,slopeL,slopeR,scaleFDLoss,qdist)
    elif FDType =='T':
        loss_k,loss_q = FDLossMLRFD_T(y_train_k,y_train_q,kk,qq,fk,mid,mid2,slopeL,slopeR,scaleFDLoss,qdist)
    elif FDType =='G':
        loss_k,loss_q = FDLossMLRFD_G(y_train_k,y_train_q,kk,qq,fk,mid,mid2,slopeL,slopeR,scaleFDLoss,qdist)
    else:
        print("Please set the FD type to 'D', 'T', or 'G'.")
    
    return loss_k.mean(),loss_q.mean()
   
# MLRFD loss - Triangle
def FDLossMLRFD_D(y_train_k,y_train_q,kk,qq,f,mid,slopeL,slopeR,scaleFDLoss,dist=0):
    # 1. idcs - Actual/Pred Left/Right Idcs Inside (below) FD curve
    # select [0] to get the rows, not the columns
    ALIdcsI = (np.where((y_train_k<mid)  & (dist + slopeL*y_train_k > y_train_q))[0])
    PLIdcsI = (np.where((kk       <mid)  & (dist + slopeL*kk        > qq       ))[0])
    ARIdcsI = (np.where((y_train_k>=mid) & ((dist + slopeL*mid+(y_train_k-mid)*slopeR) > y_train_q))[0])
    PRIdcsI = (np.where((kk       >=mid) & ((dist + slopeL*mid+(kk       -mid)*slopeR) > qq       ))[0])
    PLIdcs  = (np.where((kk<mid))[0])
    PRIdcs  = (np.where((kk>=mid))[0])
    
    # 2. filtering factors --> all but (PredictedOutside&ActualOutside) = 0
    leftfs = np.full((max(kk.shape),), f)
    leftfs[ALIdcsI] = 0
    leftfs[PLIdcsI] = 0
    leftfs = leftfs.reshape(-1,1)
    leftfs = torch.Tensor(leftfs[PLIdcs])
    rightfs = np.full((max(kk.shape),), f)
    rightfs[ARIdcsI] = 0
    rightfs[PRIdcsI] = 0
    rightfs = rightfs.reshape(-1,1)
    rightfs = torch.Tensor(rightfs[PRIdcs])
    
    # 3. calculate losses --> regular*(1-factor)+FD*factor
    # q
    leftFD  = ((qq - (slopeL*kk + dist))**2)
    rightFD = ((qq - (slopeL*mid+dist+(kk-mid)*slopeR))**2)
    Regq    = ((qq - y_train_q)**2)
    leftLossq  = Regq[PLIdcs]*(1-leftfs)  + leftfs*leftFD[PLIdcs]    *scaleFDLoss   # replace regLoss
    rightLossq = Regq[PRIdcs]*(1-rightfs) + rightfs*rightFD[PRIdcs]  *scaleFDLoss   # replace regLoss
    lossq      = torch.cat([leftLossq,rightLossq])
    # k
    leftFD  = ((kk - (qq-dist)/slopeL)**2)
    rightFD = ((kk - (mid+(qq-dist-slopeL*mid)/slopeR))**2) 
    Regk    = ((kk - y_train_k)**2) 
    leftLossk  = Regk[PLIdcs]*(1-leftfs)  + leftfs*leftFD[PLIdcs]    *scaleFDLoss   # replace regLoss
    rightLossk = Regk[PRIdcs]*(1-rightfs) + rightfs*rightFD[PRIdcs]  *scaleFDLoss   # replace regLoss
    lossk      = torch.cat([leftLossk,rightLossk])
    
    return lossk,lossq

# MLRFD loss - Triangle
def FDLossMLRFD_D(y_train_k,y_train_q,kk,qq,f,mid,slopeL,slopeR,scaleFDLoss,dist=0):
    # 1. idcs - Actual/Pred Left/Right Idcs Inside (below) FD curve
    ALIdcsI = list(np.where((y_train_k<mid)  & (dist + slopeL*y_train_k > y_train_q))[0])
    PLIdcsI = list(np.where((kk       <mid)  & (dist + slopeL*kk        > qq       ))[0])
    ARIdcsI = list(np.where((y_train_k>=mid) & ((dist + slopeL*mid+(y_train_k-mid)*slopeR) > y_train_q))[0])
    PRIdcsI = list(np.where((kk       >=mid) & ((dist + slopeL*mid+(kk       -mid)*slopeR) > qq       ))[0])
    PLIdcs  = list(np.where((kk<mid)))
    PRIdcs  = list(np.where((kk>=mid)))
    
    # 2. filtering factors --> all but (PredictedOutside&ActualOutside) = 0
    leftfs = np.full((max(kk.shape),), f)
    leftfs[ALIdcsI] = 0
    leftfs[PLIdcsI] = 0
    leftfs = leftfs.reshape(-1,1)
    leftfs = torch.Tensor(leftfs[PLIdcs[0]])
    rightfs = np.full((max(kk.shape),), f)
    rightfs[ARIdcsI] = 0
    rightfs[PRIdcsI] = 0
    rightfs = rightfs.reshape(-1,1)
    rightfs = torch.Tensor(rightfs[PRIdcs[0]])
    
    # 3. calculate losses --> regular*(1-factor)+FD*factor
    # q
    leftFD  = ((qq - (slopeL*kk + dist))**2)
    rightFD = ((qq - (slopeL*mid+dist+(kk-mid)*slopeR))**2)
    Regq    = ((qq - y_train_q)**2)
    leftLossq  = Regq[PLIdcs[0]]*(1-leftfs)  + leftfs*leftFD[PLIdcs[0]]    *scaleFDLoss   # replace regLoss
    rightLossq = Regq[PRIdcs[0]]*(1-rightfs) + rightfs*rightFD[PRIdcs[0]]  *scaleFDLoss   # replace regLoss
    lossq      = torch.cat([leftLossq,rightLossq])
    # k
    leftFD  = ((kk - (qq-dist)/slopeL)**2)
    rightFD = ((kk - (mid+(qq-dist-slopeL*mid)/slopeR))**2) 
    Regk    = ((kk - y_train_k)**2) 
    leftLossk  = Regk[PLIdcs[0]]*(1-leftfs)  + leftfs*leftFD[PLIdcs[0]]    *scaleFDLoss   # replace regLoss
    rightLossk = Regk[PRIdcs[0]]*(1-rightfs) + rightfs*rightFD[PRIdcs[0]]  *scaleFDLoss   # replace regLoss
    lossk      = torch.cat([leftLossk,rightLossk])
    
    return lossk,lossq

# MLRFD loss - Trapezoidal
def FDLossMLRFD_T(y_train_k,y_train_q,kk,qq,f,mid,mid2,slopeL,slopeR,scaleFDLoss,dist=0):
    # 1. idcs - Actual/Pred Left/Right Idcs Inside (below) FD curve
    ALIdcsI = (np.where((y_train_k<mid)   & (dist + slopeL*y_train_k > y_train_q))[0])
    PLIdcsI = (np.where((kk       <mid)   & (dist + slopeL*kk        > qq       ))[0])
    AMIdcsI = (np.where((y_train_k>=mid)  & (y_train_k<mid2) & ((dist + slopeL*mid) > y_train_q))[0])
    PMIdcsI = (np.where((kk       >=mid)  & (kk       <mid2) & ((dist + slopeL*mid) > qq       ))[0])
    ARIdcsI = (np.where((y_train_k>=mid2) & ((dist + slopeL*mid+(y_train_k-mid2)*slopeR) > y_train_q))[0])
    PRIdcsI = (np.where((kk       >=mid2) & ((dist + slopeL*mid+(kk       -mid2)*slopeR) > qq       ))[0])
    PLIdcs  = (np.where((kk<mid))[0])
    PMIdcs  = (np.where((kk>=mid) & (kk<mid2))[0])
    PRIdcs  = (np.where((kk>=mid2))[0])
    
    # 2. filtering factors --> all but (PredictedOutside&ActualOutside) = 0
    leftfs = np.full((max(kk.shape),), f)
    leftfs[ALIdcsI] = 0
    leftfs[PLIdcsI] = 0
    leftfs = leftfs.reshape(-1,1)
    leftfs = torch.Tensor(leftfs[PLIdcs])
    midfs = np.full((max(kk.shape),), f)
    midfs[AMIdcsI] = 0
    midfs[PMIdcsI] = 0
    midfs = midfs.reshape(-1,1)
    midfs = torch.Tensor(midfs[PMIdcs])
    rightfs = np.full((max(kk.shape),), f)
    rightfs[ARIdcsI] = 0
    rightfs[PRIdcsI] = 0
    rightfs = rightfs.reshape(-1,1)
    rightfs = torch.Tensor(rightfs[PRIdcs])
    
    # 3. calculate losses --> regular*(1-factor)+FD*factor
    # q
    Reg     = (qq - y_train_q)**2
    leftFD  = (qq - (slopeL*kk+dist))**2
    midFD   = (qq - (slopeL*mid+dist))**2
    rightFD = ((qq - (slopeL*mid+dist+(kk-mid2)*slopeR))**2)
    leftLoss  = Reg[PLIdcs]*(1-leftfs)  + leftfs*leftFD[PLIdcs]     *scaleFDLoss    # replace regLoss
    midLoss   = Reg[PMIdcs]*(1-midfs)   + midfs*midFD[PMIdcs]       *scaleFDLoss    # replace regLoss
    rightLoss = Reg[PRIdcs]*(1-rightfs) + rightfs*rightFD[PRIdcs]   *scaleFDLoss    # replace regLoss
    loss_q    = torch.cat([leftLoss,midLoss,rightLoss])
    #k
    Reg     = (kk - y_train_k)**2
    leftFD  = (kk - (qq-dist)/slopeL)**2
    midFD   = 0 # no change needed for k, slope horizontal
    rightFD = ((kk - (mid2+(qq-dist-slopeL*mid)/slopeR))**2)
    leftLoss  = Reg[PLIdcs]*(1-leftfs)  + leftfs*leftFD[PLIdcs]     *scaleFDLoss    # replace regLoss
    midLoss   = Reg[PMIdcs]             + 0
    rightLoss = Reg[PRIdcs]*(1-rightfs) + rightfs*rightFD[PRIdcs]   *scaleFDLoss    # replace regLoss
    loss_k    = torch.cat([leftLoss,midLoss,rightLoss])

    return loss_k,loss_q

# MLRFD loss - Greenshield
def FDLossMLRFD_G(y_train_k,y_train_q,kk,qq,f,mid,mid2,slopeL,slopeR,scaleFDLoss,dist=0):
    vfree    = slopeL
    kjam     = mid - slopeL*mid/slopeR # x intercept of slope R
    qmax     = vfree*kjam*0.25
    
    # 1. idcs - Actual/Pred Left/Right Idcs Inside (below) FD curve
    # any q values > qmax
    ATIdcsO  = (np.where(y_train_q-dist >=qmax)[0])
    PTIdcsO  = (np.where(qq-dist        >=qmax)[0])
    PTIdcs   = (np.where((qq-dist       >=qmax))[0]) 
    # all other values
    ABIdcsO  = (np.where((dist + vfree*y_train_k*(1-(y_train_k/kjam)) < y_train_q) & (y_train_q-dist<qmax))[0])
    PBIdcsO  = (np.where((dist + vfree*kk       *(1-(kk       /kjam)) < qq       ) & (qq-dist       <qmax))[0])
    PBIdcs   = (np.where((qq-dist       <qmax))[0]) 
    
    # 2. filtering factors --> all but (PredictedOutside&ActualOutside) = 0
    Topfs    = np.full((max(kk.shape),), float(0))
    Topfs[[i for i in ATIdcsO if i in PTIdcsO]] = float(f)
    Topfs    = Topfs.reshape(-1,1)
    Tfs      = torch.Tensor(Topfs[PTIdcs])
    Bottomfs = np.full((max(kk.shape),), float(0))
    Bottomfs[[i for i in ABIdcsO if i in PBIdcsO]] = float(f)
    Bottomfs = Bottomfs.reshape(-1,1)
    Bfs      = torch.Tensor(Bottomfs[PBIdcs])
    
    # 3. calculate losses --> regular*(1-factor)+FD*factor
    # 3.a. k - tricky
    Reg    = ((kk - y_train_k)**2)
    # top - qq>qmax=vfree*kjam/4, so FDk=kmid=kjam/2
    TFD    = ((kk - kjam*0.5)**2) # kk*0 + kjam*0.5 # k at max q
    TLoss  = Reg[PTIdcs]*(1-Tfs)          + Tfs*TFD[PTIdcs]     *scaleFDLoss    # replace regLoss
    # bottom - calc. both density values for that q values (bc of plusminus). Then use the smaller loss.
    with torch.no_grad():
        discriminant = vfree*vfree+4*qq*vfree/kjam # b2-4ac
        lplus  = ((kk - (vfree + (discriminant)**(0.5))/(2*vfree/kjam) )**2)
        lminus = ((kk - (vfree - (discriminant)**(0.5))/(2*vfree/kjam) )**2)
        # use sign of closer k point on FD
        plusminus = torch.tensor([-1 if (p==p and m==m and p>m) else 1 for p,m in zip(lplus,lminus)]).reshape(-1,1)
        # so values above curve are not Nan (this step done for ALL points). Filtered later.
        removeaboves = torch.tensor([1 if (p==p and m==m) else 0 for p,m in zip(lplus,lminus)]).reshape(-1,1)
    BFD    = removeaboves*((kk - (vfree+plusminus*(vfree*vfree+removeaboves*4*qq*vfree/kjam)**(0.5))/(2*vfree/kjam) )**2)
    BLoss  = Reg[PBIdcs]*(1-Bfs)         + Bfs*BFD[PBIdcs]     *scaleFDLoss    # replace regLoss
    # join
    loss_k = torch.cat([BLoss,TLoss])
    # 3.b. q - easy
    Reg    = ((qq - y_train_q)**2)
    FD     = ((qq - (vfree*kk*(1-(kk/kjam))))**2)
    TLoss  = Reg[PTIdcs]*(1-Tfs)          + Tfs*FD[PTIdcs]     *scaleFDLoss    # replace regLoss
    BLoss  = Reg[PBIdcs]*(1-Bfs)          + Bfs*FD[PBIdcs]     *scaleFDLoss    # replace regLoss
    loss_q = torch.cat([BLoss,TLoss])
    
    return loss_k,loss_q

############################
# OTHER FUNCTIONS
############################    

# plot loss curves of networks
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

def calc_error_values(p,t,verbose='on'):
    avg_rmse = np.sqrt(np.nanmean(((p-t)**2),axis=0))
    avg_mae = np.nanmean(abs(p-t),axis=0)
    avgtargets = np.nanmean(t,axis=0)
    avg_mape = [np.nanmean((abs((p[t[:,i]>1,i]-t[t[:,i]>1,i])/t[t[:,i]>1,i]))) for i in range(t.shape[1]) ]
    r2 = []
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

def calc_error_values_defaultr2(p,t,verbose='on'):
    avg_rmse = np.sqrt(np.nanmean(((p-t)**2),axis=0))
    avg_mae = np.nanmean(abs(p-t),axis=0)
    avgtargets = np.nanmean(t,axis=0)
    avg_mape = [np.nanmean((abs((p[t[:,i]>1,i]-t[t[:,i]>1,i])/t[t[:,i]>1,i]))) for i in range(t.shape[1]) ]
    r2 = [r2_score(t[:,i],p[:,i]) for i in range(0,t.shape[1])]
    if verbose=='on':
        print('Loss MAE\n',avg_mae)#/avgtargets
        print('Loss MAPE\n',avg_mape)
        print('Loss RMSE\n',avg_rmse)#/avgtargets
        print('Loss R2 (coefficient of determination)\n',r2)
    return [avg_mae.tolist(),avg_mape,avg_rmse.tolist(),r2]

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
        #plt.show()
        