#!/usr/bin/env python3
# author: jinhee

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import os
import sys
import utils
import models
import losses
from sklearn.metrics import f1_score

if __name__ == '__main__':

    # GPU allocation
    gid = 1
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f'cuda:{gid}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
    print ('cuda available :', use_cuda)

    # Set hyperparams
    dpath = './n1sec_ovlap_random_all_seg.npz' # data
    #dpath = '/mnt/sdd1/pjh/AD_EEG/data_resting/60sec_random.npz' # data for CCRNN
    
    epochs, batch = 3, 256
    lr = 5e-3
    nfd = 5 
    #segsize = 30 # for CCRNN

    spath = f'./weights/5cv/EEGNet_F16D4_b{batch}_ceflf1/' # savepath
    for cv in range(nfd): # CV
        # Load dataset
        #trs, tes, tensor_tr, tensor_te, num_class = utils.load_data_cross_cv(dpath, batch, cv=nfd, fd=cv) # cross
        trs, tes, tensor_tr, tensor_te, num_class = utils.load_data_within_cv(dpath, batch, cv=nfd, fd=cv) # within
        #trs, tes, tensor_tr, tensor_te, num_class = utils.load_mesh_data_cv(dpath, batch, segsize=segsize, cv=nfd, fd=cv) # for CCRNN
    
        xtr, ytr, tr_pidx = trs
        xte, yte, te_pidx = tes
        #print (xtr.shape, ytr.shape, tr_pidx.shape)
        #print (xte.shape, yte.shape, te_pidx.shape)
    
        # Create a folder to save
        sspath = f'{spath}fold{cv}/'
        if not os.path.exists(sspath):
            os.makedirs(sspath, exist_ok=True)
        
        # Network
        #cnn = models.EEGNet(bias=False, F1=8, D=2)
        cnn = models.sub_EEGNet(bias=False, F1=8, D=2)
        #cnn = models.CCRNN(nSeg=segsize)
        if use_cuda:
            cnn = cnn.cuda()
        
        # Loss
        focalloss, celoss, f1loss = losses.FocalLoss(), nn.CrossEntropyLoss(), losses.F1_Loss()
    
        # Optimizer
        optimizer = optim.Adam(cnn.parameters(), lr=lr, weight_decay=5e-4)
        
        # Train the network
        trloss, tracc = [], []
        teloss, teacc, tepred, tef1, tsacc = [], [], [], [], []
        for ep in range(epochs):
            sname = f'{sspath}{ep}.tar'
            cnn.train()
            
            trloss_, tracc_ = [], []
            for i, data in enumerate(tensor_tr): #training iteration
                x, y = data
                
                """ # for CCRNN
                x = x.reshape((np.prod(x.shape[:2]), x.shape[2], x.shape[3]))
                x = torch.unsqueeze(x, axis=1)
                """
                if use_cuda:
                    x, y = x.cuda(), y.cuda()
        
                optimizer.zero_grad()
                pred = cnn(x)            
                
                l1, l2, l3 = focalloss(pred, y), celoss(pred, y), f1loss(pred, y)
                loss = l1 + l2 + l3
                loss.backward() #back-prop
                optimizer.step() #update weight
                
                ltr = pred.argmax(dim=1)
                tra = y[y==ltr].size(0)
                
                trloss_.append(loss.item())
                tracc_.append(tra)
                del loss, pred
                
            trloss.append(np.average(trloss_))
            tracc.append(np.sum(tracc_)/len(xtr))

            # Inference per subject
            uteSub = np.unique(te_pidx)
            with torch.no_grad(): 
                cnn.eval()
                teinfo, sloss, sacc, sf1, spred = [], [], [], [], []
                for ts in uteSub: # per test subject
                    rtidx = np.where(te_pidx==ts)[0]
                    txte, tyte, tte_pidx = xte[rtidx], yte[rtidx], te_pidx[rtidx]
                    teinfo.append([ts, tyte[0]])
                    txte, tyte = torch.from_numpy(txte), torch.from_numpy(tyte)
                    tes = data_utils.TensorDataset(txte, tyte)
                    ttensor = data_utils.DataLoader(tes, batch_size=batch, shuffle=True)
        
                    teloss_, teacc_, tepred_ = [], [], []
                    for j, te in enumerate(ttensor):
                        x, y = te
                        """ for CCRNN
                        x = x.reshape((np.prod(x.shape[:2]), x.shape[2], x.shape[3]))
                        x = torch.unsqueeze(x, axis=1)
                        """
                        if use_cuda:
                            x, y = x.cuda(), y.cuda()
                        tpred = cnn(x)
                        lte = tpred.argmax(dim=1)
                        tea = y[y==lte].size(0)
    
                        tl1, tl2, tl3 = focalloss(tpred, y), celoss(tpred, y), f1loss(tpred, y)
                        tloss = tl1 + tl2 + tl3
                        teloss_.append(tloss.item())
                        teacc_.append(tea)
                        tepred_.append(lte.cpu().numpy())
                        del tpred, tloss
                        
                    sloss.append(np.average(teloss_))
                    sacc.append(np.sum(teacc_)/len(txte))
                    spred.append(np.concatenate((tepred_)))
                    
            teloss.append(np.average(sloss))
            teacc.append(np.average(sacc))
            tepred.append(np.array(spred))
            tsacc.append(np.array(sacc))
            
            #if (ep+1)%100==0:
            print (f'epoch : {ep}/{epochs} | trloss/acc : {trloss[-1]:.4f}/{tracc[-1]:.4f} | teloss/acc : {teloss[-1]:.4f}/{teacc[-1]:.4f}')
            
            # Save model
            torch.save({'model': cnn.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epochs, 'lr':lr, 'batch':batch}, sname)
    
        # Save results
        trloss, tracc = np.array(trloss), np.array(tracc)
        teloss, teacc = np.array(teloss), np.array(teacc)
        tepred = np.array(tepred)
        tsacc = np.array(tsacc)
        #print (trloss.shape, tracc.shape, teloss.shape, teacc.shape, tepred.shape)
        np.savez(sspath+'res.npz', trloss=trloss, tracc=tracc, teloss=teloss, teacc=teacc, tepred=tepred, tsacc=tsacc)
    
    
    # Concat cv results
    lists = np.sort(os.listdir(spath))
    assert len(lists)==5
    fres = []
    for ls in lists:
        dat = np.load(spath+ls+'/res.npz', allow_pickle=True)    
        tmp = []
        for var in np.sort(dat.files):
            tmp.append([var, dat[var]])
        fres.append(tmp)
    np.savez(spath+'fdres.npz', fd0=fres[0], fd1=fres[1], fd2=fres[2], fd3=fres[3], fd4=fres[4])
    
    
