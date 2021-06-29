#!/usr/bin/env python3
# author: jinhee

import torch
import torch.utils.data as data_utils
import numpy as np
import os
from sklearn.model_selection import KFold

def load_data_cross_cv(dpath, batch, cv=5, fd=0):
    #Load .npz
    datpath = dpath
    dat = np.load(datpath, allow_pickle=True)

    tr_x, tr_y, tr_pidx = dat['tr_x'], dat['tr_y'], dat['tr_pidx']
    te_x, te_y, te_pidx = dat['te_x'], dat['te_y'], dat['te_pidx']
    
    if tr_x.shape[1] != 32:
        tr_x, te_x = np.transpose(tr_x, [0,2,1]), np.transpose(te_x, [0,2,1])
    num_class = len(np.unique(tr_y))

    x, y, pidx = np.concatenate((tr_x, te_x)), np.concatenate((tr_y, te_y)), np.concatenate((tr_pidx, te_pidx))
    #print (x.shape, y.shape, pidx.shape)
    upidx = np.unique(pidx)
    
    # re-ordering
    seq = []
    for up in np.unique(pidx):
        tidx = np.where(pidx==up)[0]
        seq.append(tidx)
    seq = np.concatenate((seq))
    x, y, pidx = x[seq], y[seq], pidx[seq]

    # split data
    uy = np.unique(y)
    groups = [] #[0]:NC, [1]:aAD
    for uy_ in uy:
        tidx = np.where(y==uy_)[0]
        utpidx = np.unique(pidx[tidx])
        #print (utpidx.shape, np.unique(utpidx))
        #print (utpidx.shape, nsplit)
        kf = KFold(n_splits=cv)

        idxs = []
        for tridx, teidx in kf.split(utpidx):
            idxs.append(utpidx[teidx])
        groups.append(np.array(idxs))
    
    idxs = []
    for grp in range(len(groups[0])):
        teidx = np.concatenate((groups[0][grp], groups[1][grp]))
        tridx = np.setdiff1d(upidx, teidx)
        idxs.append([tridx, teidx])
    
    # find sample idx using pidx
    tridx = np.concatenate(([np.where(pidx==x)[0] for x in idxs[fd][0]]))
    teidx = np.concatenate(([np.where(pidx==x)[0] for x in idxs[fd][1]]))
    
    tr_x, tr_y, tr_pidx = x[tridx], y[tridx], pidx[tridx]
    te_x, te_y, te_pidx = x[teidx], y[teidx], pidx[teidx]
    tr_x = np.expand_dims(tr_x, axis=1)
    te_x = np.expand_dims(te_x, axis=1)

    ## array to tensor
    xtr, ytr = torch.from_numpy(tr_x), torch.from_numpy(tr_y)
    xte, yte = torch.from_numpy(te_x), torch.from_numpy(te_y)
    trn, te = data_utils.TensorDataset(xtr, ytr), data_utils.TensorDataset(xte, yte)
    trn_loader, te_loader = data_utils.DataLoader(trn, batch_size=batch, shuffle=True), data_utils.DataLoader(te, batch_size=batch, shuffle=True)
    iters = len(trn_loader)

    return (tr_x, tr_y, tr_pidx), (te_x, te_y, te_pidx), trn_loader, te_loader, num_class


def load_data_within_cv(dpath, batch, cv=5, fd=0):
    #Load .npz
    datpath = dpath
    dat = np.load(datpath, allow_pickle=True)

    tr_x, tr_y, tr_pidx = dat['tr_x'], dat['tr_y'], dat['tr_pidx']
    te_x, te_y, te_pidx = dat['te_x'], dat['te_y'], dat['te_pidx']

    if tr_x.shape[1] != 32:
        tr_x, te_x = np.transpose(tr_x, [0,2,1]), np.transpose(te_x, [0,2,1])
    num_class = len(np.unique(tr_y))

    x, y, pidx = np.concatenate((tr_x, te_x)), np.concatenate((tr_y, te_y)), np.concatenate((tr_pidx, te_pidx))
    #print (x.shape, y.shape, pidx.shape)
    upidx = np.unique(pidx)

    # re-ordering
    seq = []
    for up in np.unique(pidx):
        tidx = np.where(pidx==up)[0]
        seq.append(tidx)
    seq = np.concatenate((seq))
    x, y, pidx = x[seq], y[seq], pidx[seq]

    # split fold
    kf = KFold(n_splits=cv)
    idxs = []
    for tridx, teidx in kf.split(np.arange(len(pidx)/len(np.unique(pidx)))):
        idxs.append([tridx, teidx])
    idxs = np.array(idxs)

    fidx = []
    for ffd in range(len(idxs)):
        ttr, tte = [], []
        for sub in upidx:
            tidx = np.where(pidx==sub)[0]
            ttr.append(tidx[idxs[ffd][0]])
            tte.append(tidx[idxs[ffd][1]])
        ttr, tte = np.concatenate(ttr), np.concatenate(tte)
        fidx.append([ttr, tte])
    idxs = np.array(fidx)

    # find sample idx using pidx
    tr_x, tr_y, tr_pidx = x[idxs[fd][0]], y[idxs[fd][0]], pidx[idxs[fd][0]]
    te_x, te_y, te_pidx = x[idxs[fd][1]], y[idxs[fd][1]], pidx[idxs[fd][1]]

    tr_x = np.expand_dims(tr_x, axis=1)
    te_x = np.expand_dims(te_x, axis=1)

    ## array to tensor
    xtr, ytr = torch.from_numpy(tr_x), torch.from_numpy(tr_y)
    xte, yte = torch.from_numpy(te_x), torch.from_numpy(te_y)
    trn, te = data_utils.TensorDataset(xtr, ytr), data_utils.TensorDataset(xte, yte)
    trn_loader, te_loader = data_utils.DataLoader(trn, batch_size=batch, shuffle=True), data_utils.DataLoader(te, batch_size=batch, shuffle=True)
    iters = len(trn_loader)

    return (tr_x, tr_y, tr_pidx), (te_x, te_y, te_pidx), trn_loader, te_loader, num_class

def load_sub_data_within_cv(dpath, batch, cv=5, fd=0):
    dat = np.load(dpath, allow_pickle=True)

    tr_x, tr_y, tr_pidx = dat['tr_x'], dat['tr_y'], dat['tr_pidx']
    te_x, te_y, te_pidx = dat['te_x'], dat['te_y'], dat['te_pidx']

    if tr_x.shape[1] != 32:
        tr_x, te_x = np.transpose(tr_x, [0,2,1]), np.transpose(te_x, [0,2,1])

    x, y, pidx = np.concatenate((tr_x, te_x)), np.concatenate((tr_y, te_y)), np.concatenate((tr_pidx, te_pidx))
    upidx = np.unique(pidx)

    # re-ordering
    seq = []
    for up in np.unique(pidx):
        tidx = np.where(pidx==up)[0]
        seq.append(tidx)
    seq = np.concatenate((seq))
    x, y, pidx = x[seq], y[seq], pidx[seq]

    # re-labeling
    nlab = []
    for i in range(len(upidx)):
        nlab.append([upidx[i], i])
    nlab = np.array(nlab) # (exist_subidx, new_label)

    labs = [] # (exist_subidx, sevidx)
    new_y = y.copy()
    for i in upidx:
        tidx = np.where(pidx==i)[0]
        ny = np.where(nlab[:,0]==pidx[tidx[0]])[0][0]
        new_y[tidx] = ny
        labs.append([i, y[tidx[0]]])
    labs = np.array(labs)
    num_class = len(np.unique(new_y))

    labinfo = [] # (exist_subidx, new_label, sevidx)
    for i in range(len(labs)):
        tidx = np.where(nlab[i][0]==labs[:,0])[0][0]
        labinfo.append([nlab[i][0], nlab[i][1], labs[tidx,1]])
    labinfo = np.array(labinfo)

    # split fold
    kf = KFold(n_splits=cv)
    idxs = []
    for tridx, teidx in kf.split(np.arange(len(pidx)/len(np.unique(pidx)))):
        idxs.append([tridx, teidx])
    idxs = np.array(idxs)

    fidx = []
    for ffd in range(len(idxs)):
        ttr, tte = [], []
        for sub in upidx:
            tidx = np.where(pidx==sub)[0]
            ttr.append(tidx[idxs[ffd][0]])
            tte.append(tidx[idxs[ffd][1]])
        ttr, tte = np.concatenate(ttr), np.concatenate(tte)
        fidx.append([ttr, tte])
    idxs = np.array(fidx)
    
    # find sample idx using pidx
    tr_x, tr_sevy, tr_newy, tr_pidx = x[idxs[fd][0]], y[idxs[fd][0]], new_y[idxs[fd][0]], pidx[idxs[fd][0]]
    te_x, te_sevy, te_newy, te_pidx = x[idxs[fd][1]], y[idxs[fd][1]], new_y[idxs[fd][1]], pidx[idxs[fd][1]]

    tr_x = np.expand_dims(tr_x, axis=1)
    te_x = np.expand_dims(te_x, axis=1)

    ## array to tensor
    xtr, ytr = torch.from_numpy(tr_x), torch.from_numpy(tr_newy)
    xte, yte = torch.from_numpy(te_x), torch.from_numpy(te_newy)
    trn, te = data_utils.TensorDataset(xtr, ytr), data_utils.TensorDataset(xte, yte)
    trn_loader, te_loader = data_utils.DataLoader(trn, batch_size=batch, shuffle=True), data_utils.DataLoader(te, batch_size=batch, shuffle=True)
    iters = len(trn_loader)

    return (tr_x, tr_sevy, tr_newy, tr_pidx), (te_x, te_sevy, te_newy, te_pidx), trn_loader, te_loader, num_class, labinfo

def load_sub_data_cross_cv(dpath, batch, cv=5, fd=0):
    #Load .npz
    dat = np.load(dpath, allow_pickle=True)

    tr_x, tr_y, tr_pidx = dat['tr_x'], dat['tr_y'], dat['tr_pidx']
    te_x, te_y, te_pidx = dat['te_x'], dat['te_y'], dat['te_pidx']

    if tr_x.shape[1] != 32:
        tr_x, te_x = np.transpose(tr_x, [0,2,1]), np.transpose(te_x, [0,2,1])
    num_class = len(np.unique(tr_y))

    x, y, pidx = np.concatenate((tr_x, te_x)), np.concatenate((tr_y, te_y)), np.concatenate((tr_pidx, te_pidx))
    upidx = np.unique(pidx)

    # re-ordering
    seq = []
    for up in np.unique(pidx):
        tidx = np.where(pidx==up)[0]
        seq.append(tidx)
    seq = np.concatenate((seq))
    x, y, pidx = x[seq], y[seq], pidx[seq]

    # re-labeling
    nlab = []
    for i in range(len(upidx)):
        nlab.append([upidx[i], i])
    nlab = np.array(nlab) # (exist_subidx, new_label)

    labs = [] # (exist_subidx, sevidx)
    new_y = y.copy()
    for i in upidx:
        tidx = np.where(pidx==i)[0]
        ny = np.where(nlab[:,0]==pidx[tidx[0]])[0][0]
        new_y[tidx] = ny
        labs.append([i, y[tidx[0]]])
    labs = np.array(labs)
    num_class = len(np.unique(new_y))

    labinfo = [] # (exist_subidx, new_label, sevidx)
    for i in range(len(labs)):
        tidx = np.where(nlab[i][0]==labs[:,0])[0][0]
        labinfo.append([nlab[i][0], nlab[i][1], labs[tidx,1]])
    labinfo = np.array(labinfo)
    
    # split fold
    kf = KFold(n_splits=cv)
    idxs = []
    for tridx, teidx in kf.split(np.unique(new_y)):
        idxs.append([tridx, teidx])
    idxs = np.array(idxs)
    
    # find sample idx using pidx
    tridx = np.concatenate(([np.where(new_y==x)[0] for x in idxs[fd][0]]))
    teidx = np.concatenate(([np.where(new_y==x)[0] for x in idxs[fd][1]]))
    
    tr_x, tr_sevy, tr_newy, tr_pidx = x[tridx], y[tridx], new_y[tridx], pidx[tridx]
    te_x, te_sevy, te_newy, te_pidx = x[teidx], y[teidx], new_y[teidx], pidx[teidx]
    
    # re-labeling for training-set
    upidx = np.unique(tr_pidx)
    fdnlab = []
    for i in range(len(upidx)):
        fdnlab.append([upidx[i], i])
    fdnlab = np.array(fdnlab) # (exist_subidx, new_label)

    fdlabs = []
    fdnew_y = tr_sevy.copy()
    for i in upidx:
        tidx = np.where(tr_pidx==i)[0]
        ny = np.where(fdnlab[:,0]==tr_pidx[tidx[0]])[0][0]
        fdnew_y[tidx] = ny
        fdlabs.append([i, tr_sevy[tidx[0]]])
    fdlabs = np.array(fdlabs)
    num_class = len(np.unique(fdnew_y))

    fdlabinfo = [] # (exist_subidx, new_label, sevidx)
    for i in range(len(fdlabs)):
        tidx = np.where(fdnlab[i][0]==fdlabs[:,0])[0][0]
        fdlabinfo.append([fdnlab[i][0], fdnlab[i][1], fdlabs[tidx,1]])
    fdlabinfo = np.array(fdlabinfo)

    tr_x = np.expand_dims(tr_x, axis=1)
    te_x = np.expand_dims(te_x, axis=1)

    xtr, ytr = torch.from_numpy(tr_x), torch.from_numpy(fdnew_y)
    trn = data_utils.TensorDataset(xtr, ytr)
    trn_loader = data_utils.DataLoader(trn, batch_size=batch, shuffle=True)
    iters = len(trn_loader)
    return (te_x, te_sevy, te_newy, te_pidx), trn_loader, num_class, labinfo, fdlabinfo

def load_loocv_data(dpath, batch, idx):
    #Load .npz
    datpath = dpath
    dat = np.load(datpath, allow_pickle=True)

    tr_x, tr_y, tr_pidx = dat['tr_x'], dat['tr_y'], dat['tr_pidx']
    te_x, te_y, te_pidx = dat['te_x'], dat['te_y'], dat['te_pidx']

    if tr_x.shape[1] != 32:
        tr_x, te_x = np.transpose(tr_x, [0,2,1]), np.transpose(te_x, [0,2,1])
    num_class = len(np.unique(tr_y))

    x, y, pidx = np.concatenate((tr_x, te_x)), np.concatenate((tr_y, te_y)), np.concatenate((tr_pidx, te_pidx))
    #print (x.shape, y.shape, pidx.shape)
    upidx = np.unique(pidx)
    
    # re-ordering
    seq = []
    for up in np.unique(pidx):
        tidx = np.where(pidx==up)[0]
        seq.append(tidx)
    seq = np.concatenate((seq))
    x, y, pidx = x[seq], y[seq], pidx[seq]
    #print (x.shape, y.shape, pidx.shape)
    
    # split tr/te
    teidx = upidx[idx]
    tridx = np.setdiff1d(upidx, teidx)

    teidx = np.where(pidx==teidx)[0]
    tridx = np.concatenate([np.where(pidx==x)[0] for x in tridx])
    
    tr_x, tr_y, tr_pidx = x[tridx], y[tridx], pidx[tridx]
    te_x, te_y, te_pidx = x[teidx], y[teidx], pidx[teidx]
        
    tr_x = np.expand_dims(tr_x, axis=1)
    te_x = np.expand_dims(te_x, axis=1)

    ## array to tensor
    xtr, ytr = torch.from_numpy(tr_x), torch.from_numpy(tr_y)
    xte, yte = torch.from_numpy(te_x), torch.from_numpy(te_y)
    trn, te = data_utils.TensorDataset(xtr, ytr), data_utils.TensorDataset(xte, yte)
    trn_loader, te_loader = data_utils.DataLoader(trn, batch_size=batch, shuffle=True), data_utils.DataLoader(te, batch_size=batch, shuffle=True)
    iters = len(trn_loader)

    return (tr_x, tr_y, tr_pidx), (te_x, te_y, te_pidx), trn_loader, te_loader, num_class

def load_mesh_data_cv(dpath, batch, segsize=30, cv=5, fd=0):
    #Load .npz
    datpath = dpath
    dat = np.load(datpath, allow_pickle=True)

    tr_x, tr_y, tr_pidx = dat['tr_x'], dat['tr_y'], dat['tr_pidx']
    te_x, te_y, te_pidx = dat['te_x'], dat['te_y'], dat['te_pidx']
    ch_names = dat['ch_names']

    if tr_x.shape[1] != 32:
        tr_x, te_x = np.transpose(tr_x, [0,2,1]), np.transpose(te_x, [0,2,1])
    num_class = len(np.unique(tr_y))

    x, y, pidx = np.concatenate((tr_x, te_x)), np.concatenate((tr_y, te_y)), np.concatenate((tr_pidx, te_pidx))
    #print (x.shape, y.shape, pidx.shape)
    upidx = np.unique(pidx)

    # re-ordering
    seq = []
    for up in np.unique(pidx):
        tidx = np.where(pidx==up)[0]
        seq.append(tidx)
    seq = np.concatenate((seq))
    x, y, pidx = x[seq], y[seq], pidx[seq]

    # split data
    uy = np.unique(y)
    groups = [] #[0]:NC, [1]:aAD
    for uy_ in uy:
        tidx = np.where(y==uy_)[0]
        utpidx = np.unique(pidx[tidx])
        #print (utpidx.shape, np.unique(utpidx))
        #print (utpidx.shape, nsplit)
        kf = KFold(n_splits=cv)

        idxs = []
        for tridx, teidx in kf.split(utpidx):
            idxs.append(utpidx[teidx])
        groups.append(np.array(idxs))
    
    idxs = []
    for grp in range(len(groups[0])):
        teidx = np.concatenate((groups[0][grp], groups[1][grp]))
        tridx = np.setdiff1d(upidx, teidx)
        idxs.append([tridx, teidx])

    # find sample idx using pidx
    tridx = np.concatenate(([np.where(pidx==x)[0] for x in idxs[fd][0]]))
    teidx = np.concatenate(([np.where(pidx==x)[0] for x in idxs[fd][1]]))
    
    tr_x, tr_y, tr_pidx = x[tridx], y[tridx], pidx[tridx]
    te_x, te_y, te_pidx = x[teidx], y[teidx], pidx[teidx]

    #data normalization
    tr_x, te_x = scaling(tr_x), scaling(te_x)

    #convert mesh
    tr_x, mch = convert_mesh(tr_x, ch_names)
    te_x, _ = convert_mesh(te_x, ch_names)

    #Segmentation
    tr_x, tr_y, tr_pidx = mesh_segmentation(tr_x, tr_y, tr_pidx, segsize=segsize, ovlap=0.5)
    te_x, te_y, te_pidx = mesh_segmentation(te_x, te_y, te_pidx, segsize=segsize, ovlap=0.5)

    tr_x, te_x = np.transpose(tr_x, [0,3,1,2]), np.transpose(te_x, [0,3,1,2])

    #Array to tensor
    xtr, ytr = torch.from_numpy(tr_x), torch.from_numpy(tr_y)
    xte, yte = torch.from_numpy(te_x), torch.from_numpy(te_y)
    trn, te = data_utils.TensorDataset(xtr, ytr), data_utils.TensorDataset(xte, yte)
    trn_loader, te_loader = data_utils.DataLoader(trn, batch_size=batch, shuffle=True), data_utils.DataLoader(te, batch_size=batch, shuffle=True)
    iters = len(trn_loader)

    return (tr_x, tr_y, tr_pidx), (te_x, te_y, te_pidx), trn_loader, te_loader, num_class


def load_sub_mesh_data(dpath, segsize, batch):

    #Load .npz
    dat = np.load(dpath, allow_pickle=True)

    tr_x, tr_y, tr_pidx = dat['tr_x'], dat['tr_y'], dat['tr_pidx']
    te_x, te_y, te_pidx = dat['te_x'], dat['te_y'], dat['te_pidx']
    ch_names = dat['ch_names']
    num_class = len(np.unique(tr_y))

    #data normalization
    tr_x, te_x = scaling(tr_x), scaling(te_x)

    #convert mesh
    tr_x, mch = convert_mesh(tr_x, ch_names)
    te_x, _ = convert_mesh(te_x, ch_names)

    #Segmentation
    tr_x, tr_y, tr_pidx = mesh_segmentation(tr_x, tr_y, tr_pidx, segsize=segsize, ovlap=0.5)
    te_x, te_y, te_pidx = mesh_segmentation(te_x, te_y, te_pidx, segsize=segsize, ovlap=0.5)
    #print (tr_x.shape, tr_y.shape, tr_pidx.shape)
    #print (te_x.shape, te_y.shape, te_pidx.shape)

    tr_x, te_x = np.transpose(tr_x, [0,3,1,2]), np.transpose(te_x, [0,3,1,2])
    #tr_x, te_x = np.expand_dims(tr_x, axis=1), np.expand_dims(te_x, axis=1) # for 3d-cnn

    print (tr_x.shape, tr_y.shape, tr_pidx.shape)
    print (te_x.shape, te_y.shape, te_pidx.shape)
    
    utrpidx = np.unique(tr_pidx)
    nlab = []
    for i in range(len(utrpidx)):
        nlab.append([utrpidx[i], i])
    nlab = np.array(nlab)

    # re-labeling
    labs = []
    ntry = tr_y.copy()
    for i in utrpidx:
        tidx = np.where(tr_pidx==i)[0]
        ny = np.where(nlab[:,0]==tr_pidx[tidx[0]])[0][0]
        ntry[tidx] = ny
        labs.append([i, tr_y[tidx[0]]])
    labs = np.array(labs)
    num_class = len(np.unique(ntry))

    labinfo = []
    for i in range(len(labs)):
        tidx = np.where(nlab[i][0]==labs[:,0])[0][0]
        labinfo.append([nlab[i][0], nlab[i][1], labs[tidx,1]])
    labinfo = np.array(labinfo)

    ## array to tensor
    xtr, ytr = torch.from_numpy(tr_x), torch.from_numpy(ntry)
    xte, yte = torch.from_numpy(te_x), torch.from_numpy(te_y)
    trn, te = data_utils.TensorDataset(xtr, ytr), data_utils.TensorDataset(xte, yte)
    trn_loader, te_loader = data_utils.DataLoader(trn, batch_size=batch, shuffle=True), data_utils.DataLoader(te, batch_size=batch, shuffle=True)
    iters = len(trn_loader)

    return (tr_x, ntry, tr_y), (te_x, te_y, te_pidx), trn_loader, te_loader, num_class, labinfo


def scaling(datx):
    ndatx = []
    for dx in datx:
        mean, std = np.average(dx, axis=0), np.std(dx, axis=0)
        ndatx.append((dx-mean)/std)
    return np.array(ndatx)

def split_val(datx, daty, datidx, batch): # Separate tr to tr/val
    upidx = np.unique(datidx)
    ttridx, tvalidx = [], []
    for p in range(len(upidx)):
        tmpidx = np.where(datidx==upidx[p])[0]
        ntr = int(0.7*len(tmpidx))
        if ntr != 0:
            ttridx.append(tmpidx[:ntr])
            tvalidx.append(tmpidx[ntr:])
        else:
            #print ('# is zero! in split_val', upidx[p])
            ttridx.append(tmpidx)

    ttridx, tvalidx = np.concatenate(ttridx), np.concatenate(tvalidx)
    xttr, xtval = datx[ttridx], datx[tvalidx]
    yttr, ytval = daty[ttridx], daty[tvalidx]
    trpidx, valpidx = datidx[ttridx], datidx[tvalidx]
    num_class = len(np.unique(yttr))
    #print (xttr.shape, yttr.shape, xtval.shape, ytval.shape)

    ## array to tensor
    xtr, ytr = torch.from_numpy(xttr), torch.from_numpy(yttr)
    xval, yval = torch.from_numpy(xtval), torch.from_numpy(ytval)
    trn, val = data_utils.TensorDataset(xtr, ytr), data_utils.TensorDataset(xval, yval)
    trn_loader, val_loader = data_utils.DataLoader(trn, batch_size=batch, shuffle=True), data_utils.DataLoader(val, batch_size=batch, shuffle=True)
    iters = len(trn_loader)

    return (xttr, yttr, trpidx), (xtval, ytval, valpidx), trn_loader, val_loader, num_class

# Separate tr to tr/val
def split_val_sub(datx, daty, datidx, batch):
    upidx = np.unique(datidx)
    
    #continual labeling
    ninfo = []
    ndatidx = datidx.copy()
    for sp in range(len(upidx)):
        tidx = np.where(datidx==upidx[sp])[0]
        ninfo.append([sp, upidx[sp]])
        ndatidx[tidx] = sp
    
    #split
    upidx = np.unique(ndatidx)

    ttridx, tvalidx = [], []
    for p in range(len(upidx)):
        tmpidx = np.where(ndatidx==upidx[p])[0]
        ntr = int(0.8*len(tmpidx))
        ttridx.append(tmpidx[:ntr])
        tvalidx.append(tmpidx[ntr:])
    ttridx, tvalidx = np.concatenate(ttridx), np.concatenate(tvalidx)
    xttr, xtval = datx[ttridx], datx[tvalidx]
    yttr, ytval = daty[ttridx], daty[tvalidx]
    trpidx, valpidx = ndatidx[ttridx], ndatidx[tvalidx]
    num_class = len(np.unique(yttr))
    print (xttr.shape, yttr.shape, xtval.shape, ytval.shape)
    
    ## array to tensor
    xtr, ytr = torch.from_numpy(xttr), torch.from_numpy(trpidx)
    xval, yval = torch.from_numpy(xtval), torch.from_numpy(valpidx)
    trn, val = data_utils.TensorDataset(xtr, ytr), data_utils.TensorDataset(xval, yval)
    trn_loader, val_loader = data_utils.DataLoader(trn, batch_size=batch, shuffle=True), data_utils.DataLoader(val, batch_size=batch, shuffle=True)
    iters = len(trn_loader)
    num_class = len(np.unique(trpidx))
    
    return (xttr, yttr, trpidx), (xtval, ytval, valpidx), trn_loader, val_loader, np.array(ninfo)

def convert_mesh(x, ch_names):
    # for CCRNN, convert 1d-vector to 2d-mesh
    """
    Mdat, Mdat_ch = np.zeros((x.shape[0], 9, 5, x.shape[-1])), np.empty([9,5], dtype="<U3") 
    
    Mdat[:,0,0], Mdat_ch[0,0] = x[:,0], ch_names[0]
    Mdat[:,0,1], Mdat_ch[0,1] = x[:,1], ch_names[1]
    
    Mdat[:,1,1], Mdat_ch[1,1] = x[:,2], ch_names[2]
    Mdat[:,1,3], Mdat_ch[1,3] = x[:,3], ch_names[3]
    
    Mdat[:,2,0], Mdat_ch[2,0] = x[:,4], ch_names[4]
    Mdat[:,2,1], Mdat_ch[2,1] = x[:,5], ch_names[5]
    Mdat[:,2,2], Mdat_ch[2,2] = x[:,6], ch_names[6]
    Mdat[:,2,3], Mdat_ch[2,3] = x[:,7], ch_names[7]
    Mdat[:,2,4], Mdat_ch[2,4] = x[:,8], ch_names[8]
    
    Mdat[:,3,0], Mdat_ch[3,0] = x[:,9], ch_names[9]
    Mdat[:,3,1], Mdat_ch[3,1] = x[:,10], ch_names[10]
    Mdat[:,3,3], Mdat_ch[3,3] = x[:,11], ch_names[11]
    Mdat[:,3,4], Mdat_ch[3,4] = x[:,12], ch_names[12]
    
    Mdat[:,4,0], Mdat_ch[4,0] = x[:,13], ch_names[13]
    Mdat[:,4,1], Mdat_ch[4,1] = x[:,14], ch_names[14]
    Mdat[:,4,2], Mdat_ch[4,2] = x[:,15], ch_names[15]
    Mdat[:,4,3], Mdat_ch[4,3] = x[:,16], ch_names[16]
    Mdat[:,4,4], Mdat_ch[4,4] = x[:,17], ch_names[17]
    
    Mdat[:,5,0], Mdat_ch[5,0] = x[:,18], ch_names[18]
    Mdat[:,5,1], Mdat_ch[5,1] = x[:,19], ch_names[19]
    Mdat[:,5,3], Mdat_ch[5,3] = x[:,20], ch_names[20]
    Mdat[:,5,4], Mdat_ch[5,4] = x[:,21], ch_names[21]
    
    Mdat[:,6,0], Mdat_ch[6,0] = x[:,22], ch_names[22]
    Mdat[:,6,1], Mdat_ch[6,1] = x[:,23], ch_names[23]
    Mdat[:,6,2], Mdat_ch[6,2] = x[:,24], ch_names[24]
    Mdat[:,6,3], Mdat_ch[6,3] = x[:,25], ch_names[25]
    Mdat[:,6,4], Mdat_ch[6,4] = x[:,26], ch_names[26]
    
    Mdat[:,7,0], Mdat_ch[7,0] = x[:,27], ch_names[27]
    Mdat[:,7,1], Mdat_ch[7,1] = x[:,28], ch_names[28]
    Mdat[:,7,3], Mdat_ch[7,3] = x[:,29], ch_names[29]
    Mdat[:,7,4], Mdat_ch[7,4] = x[:,30], ch_names[30]
    
    Mdat[:,8,3], Mdat_ch[8,3] = x[:,31], ch_names[31]
    """
    Mdat, Mdat_ch = np.zeros((x.shape[0], 7, 5, x.shape[-1])), np.empty([7,5], dtype="<U3")
    
    Mdat[:,0,0], Mdat_ch[0,0] = x[:,0], ch_names[0]
    Mdat[:,0,3], Mdat_ch[0,3] = x[:,3], ch_names[3]
    Mdat[:,0,4], Mdat_ch[0,4] = x[:,1], ch_names[1]
    Mdat[:,0,1], Mdat_ch[0,1] = x[:,2], ch_names[2]
    
    Mdat[:,1,0], Mdat_ch[1,0] = x[:,4], ch_names[4]
    Mdat[:,1,1], Mdat_ch[1,1] = x[:,5], ch_names[5]
    Mdat[:,1,2], Mdat_ch[1,2] = x[:,6], ch_names[6]
    Mdat[:,1,3], Mdat_ch[1,3] = x[:,7], ch_names[7]
    Mdat[:,1,4], Mdat_ch[1,4] = x[:,8], ch_names[8]
    
    Mdat[:,2,0], Mdat_ch[2,0] = x[:,9], ch_names[9]
    Mdat[:,2,1], Mdat_ch[2,1] = x[:,10], ch_names[10]
    Mdat[:,2,3], Mdat_ch[2,3] = x[:,11], ch_names[11]
    Mdat[:,2,4], Mdat_ch[2,4] = x[:,12], ch_names[12]
    
    Mdat[:,3,0], Mdat_ch[3,0] = x[:,13], ch_names[13]
    Mdat[:,3,1], Mdat_ch[3,1] = x[:,14], ch_names[14]
    Mdat[:,3,2], Mdat_ch[3,2] = x[:,15], ch_names[15]
    Mdat[:,3,3], Mdat_ch[3,3] = x[:,16], ch_names[16]
    Mdat[:,3,4], Mdat_ch[3,4] = x[:,17], ch_names[17]
    
    Mdat[:,4,0], Mdat_ch[4,0] = x[:,18], ch_names[18]
    Mdat[:,4,1], Mdat_ch[4,1] = x[:,19], ch_names[19]
    Mdat[:,4,3], Mdat_ch[4,3] = x[:,20], ch_names[20]
    Mdat[:,4,4], Mdat_ch[4,4] = x[:,21], ch_names[21]
    
    Mdat[:,5,0], Mdat_ch[5,0] = x[:,22], ch_names[22]
    Mdat[:,5,1], Mdat_ch[5,1] = x[:,23], ch_names[23]
    Mdat[:,5,2], Mdat_ch[5,2] = x[:,24], ch_names[24]
    Mdat[:,5,3], Mdat_ch[5,3] = x[:,25], ch_names[25]
    Mdat[:,5,4], Mdat_ch[5,4] = x[:,26], ch_names[26]
    
    Mdat[:,6,0], Mdat_ch[6,0] = x[:,27], ch_names[27]
    Mdat[:,6,1], Mdat_ch[6,1] = x[:,28], ch_names[28]
    Mdat[:,6,2], Mdat_ch[6,2] = x[:,31], ch_names[31]
    Mdat[:,6,3], Mdat_ch[6,3] = x[:,29], ch_names[29]
    Mdat[:,6,4], Mdat_ch[6,4] = x[:,30], ch_names[30]

    return Mdat, Mdat_ch

def mesh_segmentation(datx, daty, pidx, segsize, ovlap):
    seg_datx, seg_daty, seg_pidx = [], [], []
    if ovlap == 0: #non-overlap
        numseg = int(datx.shape[1]/segsize)
    else: #overlap
        numseg = int(datx.shape[-1]/(segsize*ovlap)-1)
    
    segdatx = []
    for seg_ in range(numseg):
        if ovlap == 0:
            st = int(seg_*segsize)
        else:
            st = int(seg_*(segsize*ovlap))
        ed = st + segsize
        #print (st, ed)        
        segdatx.append(datx[:,:,:,st:ed])
    
    return np.concatenate(segdatx), np.tile(daty, numseg), np.tile(pidx, numseg)

