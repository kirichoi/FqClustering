# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:14:44 2021

@author: user
"""

import os
import numpy as np
import scipy.cluster
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import sklearn.metrics
import similaritymeasures as sm
from dynamicTreeCut import cutreeHybrid
from scipy.signal import argrelextrema, savgol_filter
from scipy.spatial.transform import Rotation
import copy
from collections import Counter

os.chdir(os.path.dirname(__file__))

PATH = r'./ALLEN_fq' # Form factor curve location

fp = [f for f in os.listdir(PATH) if os.path.exists(os.path.join(PATH, f))]
neuron_id = copy.deepcopy(fp)
neuron_id = [e[:-4] for e in neuron_id]
neuron_id = np.array(neuron_id)
fp = [os.path.join(PATH, f) for f in fp]

q = np.logspace(-3, 3, 601) # wave number q range

fq = np.empty((len(fp), len(q)))

for i,j in enumerate(fp):
    fq[i] = np.load(j)

dendrite_type = np.load('./dendrite_type.npy') # dendrite type info from Allen database

gouwens = pd.read_excel('./41593_2019_417_MOESM5_ESM.xlsx') # dendrite type info by Gouwens et al.

gouwens_df = gouwens[gouwens['m-type'].notna()]

inidx_1 = np.nonzero(np.in1d(neuron_id, np.array(gouwens_df['specimen_id']).astype(str)))[0]

inidx_2 = np.nonzero(np.in1d(np.array(gouwens_df['specimen_id']).astype(str), neuron_id))[0]

def radiusOfGyration(morph_coor, morph_dia):
    cML = np.empty((len(morph_coor), 3))
    rGy = np.empty(len(morph_coor))
    for i in range(len(morph_coor)):
        cML[i] = np.average(np.array(morph_coor[i]), axis=0, weights=morph_dia[i])
        rList = scipy.spatial.distance.cdist(np.array(morph_coor[i]), 
                                              np.array([cML[i]])).flatten()
        rGy[i] = np.sqrt(np.average(np.square(rList)))
    
    return (rGy, cML)

#%% Read reconstructions

PATH = r'./cell_types'

fp = [f for f in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, f))]
fp = [s + '/reconstruction.swc' for s in fp]
fp = [os.path.join(PATH, f) for f in fp]

morph_coor = []
morph_len = []
morph_dia = []
morph_parent = []
morph_id = []
somaP = []
branchNum = np.empty(len(fp))
branch_coor = []
branchTrk = []
length_branch = []
d2d = []

for f in range(len(fp)):
    print(f, fp[f])
    morph_neu_id = []
    morph_neu_parent = []
    morph_neu_prox = []
    morph_neu_dist = []
    
    gen = np.genfromtxt(fp[f])
    
    df = pd.DataFrame(gen)
    df.iloc[:,0] = df.iloc[:, 0].astype(int)
    df.iloc[:,1] = df.iloc[:, 1].astype(int)
    df.iloc[:,6] = df.iloc[:, 6].astype(int)
    
    scall = int(df.iloc[np.where(df[1] == 1)[0]].values[0][0])
    falseends = df.iloc[np.where(df[6] == -1)[0]][[0]].values.T
    falseends = falseends[falseends != scall]
    somaP.append(scall)
    ctr = Counter(df[6].tolist())
    ctrVal = list(ctr.values())
    ctrKey = list(ctr.keys())
    branchNum[f] = sum(i > 1 for i in ctrVal)
    branchInd = np.array(ctrKey)[np.where(np.array(ctrVal) > 1)[0]]
    branchInd = branchInd[branchInd != -1]
    
    morph_id.append(df[0].tolist())
    morph_parent.append(df[6].tolist())
    morph_coor.append(np.array(df[[2,3,4]]).tolist())
    morph_len.append(len(morph_coor[-1]))
    morph_dia.append(np.array(df[5]).tolist())
    
    list_end = np.setdiff1d(morph_id[f], morph_parent[f])
    list_end = np.unique(np.append(list_end, falseends))
    
    bPoint = np.append(branchInd, list_end)
    bPoint = np.unique(bPoint)
    
    neu_branchTrk = []
    length_branch_temp = []
    branch_dist_temp1 = []
    d2d_temp = []
    
    for bp in range(len(bPoint)):
        if bPoint[bp] != scall:
            neu_branchTrk_temp = []
            branch_dist_temp2 = []
            dist = 0
            
            neu_branchTrk_temp.append(bPoint[bp])
            branch_dist_temp2.append(morph_coor[f][morph_id[f].index(bPoint[bp])])
            parentTrck = bPoint[bp]
            parentTrck = morph_parent[f][morph_id[f].index(parentTrck)]
            if parentTrck != -1:
                neu_branchTrk_temp.append(parentTrck)
                rhs = branch_dist_temp2[-1]
                lhs = morph_coor[f][morph_id[f].index(parentTrck)]
                branch_dist_temp2.append(lhs)
                dist += np.linalg.norm(np.subtract(rhs, lhs))
                d2d_temp.append(np.linalg.norm(np.subtract(rhs, lhs)))
                
            while (parentTrck not in branchInd) and (parentTrck != -1):
                parentTrck = morph_parent[f][morph_id[f].index(parentTrck)]
                if parentTrck != -1:
                    neu_branchTrk_temp.append(parentTrck)
                    rhs = branch_dist_temp2[-1]
                    lhs = morph_coor[f][morph_id[f].index(parentTrck)]
                    branch_dist_temp2.append(lhs)
                    dist += np.linalg.norm(np.subtract(rhs, lhs))
                    d2d_temp.append(np.linalg.norm(np.subtract(rhs, lhs)))
                    
            if len(neu_branchTrk_temp) > 1:
                neu_branchTrk.append(neu_branchTrk_temp)
                branch_dist_temp1.append(branch_dist_temp2)
                length_branch_temp.append(dist)
                
    branchTrk.append(neu_branchTrk)
    branch_coor.append(branch_dist_temp1)
    length_branch.append(length_branch_temp)
    d2d.append(d2d_temp)


length_branch_flat = [item for sublist in length_branch for item in sublist]

branch_length_average = np.mean(length_branch_flat)

d2df = [item for sublist in d2d for item in sublist]


#%% Distance calculation

branch_length_average = 40.99789638178139
rgymean = 295.33080920530085

i1 = np.argmin(np.abs(q - 2*np.pi/rgymean))
i2 = np.argmin(np.abs(q - 2*np.pi/branch_length_average))+1

fq_dist = np.zeros((len(fp), len(fp)))
N_spiny = Counter(dendrite_type)['spiny']
N_aspiny = len(fp) - N_spiny
fq_dist_spiny = np.zeros((N_spiny, N_spiny))
fq_dist_aspiny = np.zeros((N_aspiny, N_aspiny))

for i in range(len(fp)):
    for j in range(len(fp)):
        exp_data = np.zeros((i2-i1, 2))
        exp_data[:,0] = np.log10(q)[i1:i2]
        exp_data[:,1] = np.log10(fq[i,i1:i2])
        num_data = np.zeros((i2-i1, 2))
        num_data[:,0] = np.log10(q)[i1:i2]
        num_data[:,1] = np.log10(fq[j,i1:i2])
        # L2
        fq_dist[i][j] = np.linalg.norm(np.log10(fq[i,i1:i2])-np.log10(fq[j,i1:i2]))
    
aspiny_idx = np.where(dendrite_type != 'spiny')[0]

for i in range(N_aspiny):
    for j in range(N_aspiny):
        exp_data = np.zeros((i2-i1, 2))
        exp_data[:,0] = np.log10(q)[i1:i2]
        exp_data[:,1] = np.log10(fq[aspiny_idx[i],i1:i2])
        num_data = np.zeros((i2-i1, 2))
        num_data[:,0] = np.log10(q)[i1:i2]
        num_data[:,1] = np.log10(fq[aspiny_idx[j],i1:i2])
        # L2
        fq_dist_aspiny[i][j] = np.linalg.norm(np.log10(fq[aspiny_idx[i],i1:i2])-np.log10(fq[aspiny_idx[j],i1:i2]))

spiny_idx = np.where(dendrite_type == 'spiny')[0]

for i in range(N_spiny):
    for j in range(N_spiny):
        exp_data = np.zeros((i2-i1, 2))
        exp_data[:,0] = np.log10(q)[i1:i2]
        exp_data[:,1] = np.log10(fq[spiny_idx[i],i1:i2])
        num_data = np.zeros((i2-i1, 2))
        num_data[:,0] = np.log10(q)[i1:i2]
        num_data[:,1] = np.log10(fq[spiny_idx[j],i1:i2])
        # L2
        fq_dist_spiny[i][j] = np.linalg.norm(np.log10(fq[spiny_idx[i],i1:i2])-np.log10(fq[spiny_idx[j],i1:i2]))

link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_dist), 
                                       method='complete', optimal_ordering=True)

link_aspiny = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_dist_aspiny), 
                                       method='complete', optimal_ordering=True)

link_spiny = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_dist_spiny), 
                                       method='complete', optimal_ordering=True)

#%% Hybrid tree cutting

ind2_aspiny = cutreeHybrid(link_aspiny, scipy.spatial.distance.squareform(fq_dist_aspiny), minClusterSize=4)['labels']

ind2_spiny = cutreeHybrid(link_spiny, scipy.spatial.distance.squareform(fq_dist_spiny), minClusterSize=4)['labels']

ind2_aspiny_idx = []

for i in np.unique(ind2_aspiny):
    ind2_aspiny_idx.append(aspiny_idx[np.where(ind2_aspiny == i)[0]])

ind2_spiny_idx = []

for i in np.unique(ind2_spiny):
    ind2_spiny_idx.append(spiny_idx[np.where(ind2_spiny == i)[0]])

ind2_aspiny_rgy = []

for i,j in enumerate(ind2_aspiny_idx):
    rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]), 
                                 list(np.array(morph_dia, dtype=object)[j]))
    ind2_aspiny_rgy.append(np.mean(rgyb))

ind2_aspiny_idx_sort = list(np.array(ind2_aspiny_idx, dtype=object)[np.argsort(ind2_aspiny_rgy)[::-1]])

ind2_spiny_rgy = []

for i,j in enumerate(ind2_spiny_idx):
    rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]), 
                                 list(np.array(morph_dia, dtype=object)[j]))
    ind2_spiny_rgy.append(np.mean(rgyb))

ind2_spiny_idx_sort = list(np.array(ind2_spiny_idx, dtype=object)[np.argsort(ind2_spiny_rgy)[::-1]])

#%% Aspiny F(q) curves per cluster

cmap = cm.get_cmap('viridis', len(ind2_aspiny_idx_sort))

for i,j in enumerate(ind2_aspiny_idx_sort):
    bl = list(np.array(length_branch, dtype=object)[j])
    blf = [item for sublist in bl for item in sublist]
    
    rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]), 
                                 list(np.array(morph_dia, dtype=object)[j]))
    
    d2db = list(np.array(d2d, dtype=object)[j])
    d2db = [item for sublist in d2db for item in sublist]
    
    fig = plt.figure(figsize=(6,4))
    if i == len(ind2_aspiny_idx_sort)-2:
        plt.plot(q, fq[j].T, color=(0.798216, 0.280197, 0.469538, 1.0))
    elif i == len(ind2_aspiny_idx_sort)-1:
        plt.plot(q, fq[j].T, color=(1.0, 0.25, 0.0, 1.0))
    else:
        plt.plot(q, fq[j].T, color=cmap(i))
    plt.vlines(2*np.pi/np.mean(blf), 1e-5, 10, color='k', ls='dashed')
    plt.vlines(2*np.pi/(2*np.mean(rgyb)), 1e-5, 10, color='k', ls='dotted')
    
    line1 = 2e-4*np.power(q, -16/7)
    line2 = 7e-10*np.power(q, -4/1)
    line3 = 3e-4*np.power(q, -1/0.395)
    line4 = 1/100*np.power(q, -1)
    line5 = 1e-5*np.power(q, -2/1)
    
    if i == 0:
        plt.plot(q[130:157], line1[130:157], lw=1.5, color='tab:blue')
        plt.plot(q[110:147], line2[110:147], lw=1.5, color='tab:red')
        plt.plot(q[180:207], line3[180:207], lw=1.5, color='tab:purple')
        plt.plot(q[220:270], line4[220:270], lw=1.5, color='k')
        plt.plot(q[150:200], line5[200:250], lw=1.5, color='tab:green')
        
        plt.text(0.03, 8e-1, r'$\mathcal{D} = 2.286$', fontsize=13, color='tab:blue')
        plt.text(0.007, 3e-3, r'$\mathcal{D} = 4$', fontsize=13, color='tab:red')
        plt.text(0.1, 2e-1, r'$\mathcal{D} = 2.53$', fontsize=13, color='tab:purple')
        plt.text(0.4, 1e-1, r'$\mathcal{D} = 1$', fontsize=13, color='k')
        plt.text(0.03, 1e-4, r'$\mathcal{D} = 2$', fontsize=13, color='tab:green')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-3, 1)
    plt.ylim(4e-5, 3)
    plt.ylabel('$F(q)$', fontsize=18)
    plt.xlabel('$q$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("$C^{aspiny}_{" + str(i+1) + "}$", fontsize=15, pad=10)
    # plt.savefig('./Allenfigures/Fq_aspiny2_' + str(i+1) + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()

#%% Spiny F(q) curves per cluter

cmap = cm.get_cmap('viridis', len(ind2_spiny_idx_sort))

for i,j in enumerate(ind2_spiny_idx_sort):
    bl = list(np.array(length_branch, dtype=object)[j])
    blf = [item for sublist in bl for item in sublist]
    
    rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]), 
                                 list(np.array(morph_dia, dtype=object)[j]))
    
    d2db = list(np.array(d2d, dtype=object)[j])
    d2db = [item for sublist in d2db for item in sublist]
    
    fig = plt.figure(figsize=(6,4))
    if i == len(ind2_spiny_idx_sort)-2:
        plt.plot(q, fq[j].T, color=(0.798216, 0.280197, 0.469538, 1.0))
    elif i == len(ind2_spiny_idx_sort)-1:
        plt.plot(q, fq[j].T, color=(1.0, 0.25, 0.0, 1.0))
    else:
        plt.plot(q, fq[j].T, color=cmap(i))
    plt.vlines(2*np.pi/np.mean(blf), 1e-5, 10, color='k', ls='dashed')
    plt.vlines(2*np.pi/(2*np.mean(rgyb)), 1e-5, 10, color='k', ls='dotted')
    
    line1 = 2e-4*np.power(q, -16/7)
    line2 = 7e-10*np.power(q, -4/1)
    line3 = 1e-4*np.power(q, -1/0.395)
    line4 = 1/1000*np.power(q, -1)
    line5 = 3e-5*np.power(q, -2/1)
    
    if i == 0:
        plt.plot(q[130:157], line1[130:157], lw=1.5, color='tab:blue')
        plt.plot(q[110:147], line2[110:147], lw=1.5, color='tab:red')
        plt.plot(q[170:197], line3[170:197], lw=1.5, color='tab:purple')
        plt.plot(q[200:250], line4[200:250], lw=1.5, color='k')
        plt.plot(q[160:200], line5[200:240], lw=1.5, color='tab:green')
        
        plt.text(0.03, 8e-1, r'$\mathcal{D} = 2.286$', fontsize=13, color='tab:blue')
        plt.text(0.007, 3e-3, r'$\mathcal{D} = 4$', fontsize=13, color='tab:red')
        plt.text(0.1, 1e-1, r'$\mathcal{D} = 2.53$', fontsize=13, color='tab:purple')
        plt.text(0.13, 1e-2, r'$\mathcal{D} = 1$', fontsize=13)
        plt.text(0.1, 1e-3, r'$\mathcal{D} = 2$', fontsize=13, color='tab:green')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-3, 1)
    plt.ylim(1e-4, 3)
    plt.ylabel('$F(q)$', fontsize=18)
    plt.xlabel('$q$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("$C^{spiny}_{" + str(i+1) + "}$", fontsize=15, pad=10)
    # plt.savefig('./Allenfigures/Fq_spiny2_' + str(i+1) + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()


#%% Spiny and aspiny neurons branch length histogram

sc5 = []

for i in ind2_spiny_idx_sort[4]:
    sc5.append(length_branch[i])
    
sc5f = [item for sublist in sc5 for item in sublist]

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
plt.hist(sc5f, bins=67, density=True, color='tab:red')
plt.xlabel('Branch Length $l$ ($\mu m$)', fontsize=15)
plt.ylabel('$P(l)$', fontsize=15)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
# plt.savefig('./Allenfigures/l_dist_sc5_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%% Box plot of branch length statistics for all neurons

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16,22))
medianprops = dict(linestyle='-', linewidth=1.5, color='k')
bp = ax[0].boxplot(length_branch[:125], 
                    notch=False, 
                    vert=False, 
                    patch_artist=True, 
                    labels=neuron_id[:125],
                    medianprops=medianprops,
                    showfliers=False)
colors = np.repeat('tab:red', len(neuron_id))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
bp = ax[1].boxplot(length_branch[125:250], 
                    notch=False, 
                    vert=False, 
                    patch_artist=True, 
                    labels=neuron_id[125:250],
                    medianprops=medianprops,
                    showfliers=False)
colors = np.repeat('tab:red', len(neuron_id))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
bp = ax[2].boxplot(length_branch[250:375], 
                    notch=False, 
                    vert=False, 
                    patch_artist=True, 
                    labels=neuron_id[250:375],
                    medianprops=medianprops,
                    showfliers=False)
colors = np.repeat('tab:red', len(neuron_id))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
bp = ax[3].boxplot(length_branch[375:], 
                    notch=False, 
                    vert=False, 
                    patch_artist=True, 
                    labels=neuron_id[375:],
                    medianprops=medianprops,
                    showfliers=False)
colors = np.repeat('tab:red', len(neuron_id))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax[0].tick_params(axis = 'y', which = 'major', labelsize = 14)
ax[0].tick_params(axis = 'y', which = 'minor', labelsize = 14)
ax[0].tick_params(axis = 'x', which = 'major', labelsize = 14)
ax[0].set_xlabel(r'Length $(\mu m)$', fontsize=17)
ax[1].tick_params(axis = 'y', which = 'major', labelsize = 14)
ax[1].tick_params(axis = 'y', which = 'minor', labelsize = 14)
ax[1].tick_params(axis = 'x', which = 'major', labelsize = 14)
ax[1].set_xlabel(r'Length $(\mu m)$', fontsize=17)
ax[2].tick_params(axis = 'y', which = 'major', labelsize = 14)
ax[2].tick_params(axis = 'y', which = 'minor', labelsize = 14)
ax[2].tick_params(axis = 'x', which = 'major', labelsize = 14)
ax[2].set_xlabel(r'Length $(\mu m)$', fontsize=17)
ax[3].tick_params(axis = 'y', which = 'major', labelsize = 14)
ax[3].tick_params(axis = 'y', which = 'minor', labelsize = 14)
ax[3].tick_params(axis = 'x', which = 'major', labelsize = 14)
ax[3].set_xlabel(r'Length $(\mu m)$', fontsize=17)
plt.tight_layout()
# plt.savefig('./Allenfigures/length_all_box.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%% Rotate reconstructions for visualization

def get_rotation_matrix(vec1, vec2):
    
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = Rotation.align_vectors(vec2, vec1)
    
    return r[0].as_matrix()

morph_coor_rot = []
branch_coor_rot = []

for i in range(len(branch_coor)): 
    branch_coor_rot_t = []
    
    apical_idx = np.argmax(length_branch[i])
    
    somcoor = morph_coor[i][somaP[i]]
    
    if np.linalg.norm(np.subtract(somcoor, branch_coor[i][apical_idx][0])) > np.linalg.norm(np.subtract(somcoor, branch_coor[i][apical_idx][-1])):
        vec = np.subtract(somcoor, branch_coor[i][apical_idx][0])
    else:
        vec = np.subtract(somcoor, branch_coor[i][apical_idx][-1])
    
    vec = vec/np.linalg.norm(vec)

    rotmat = get_rotation_matrix(vec, [0,0,-1])
    rot = Rotation.from_matrix(rotmat)
    
    for j in range(len(branch_coor[i])):
        branch_coor_rot_t.append(rot.apply(branch_coor[i][j]))
        
    branch_coor_rot.append(branch_coor_rot_t)
    morph_coor_rot.append(rot.apply(morph_coor[i]))
    
#%% Spiny and aspiny neuron reconstruction diagram per cluster

max_range_list = []

for i in range(len(morph_coor_rot)):
    Y = np.array(morph_coor_rot[i])[:,1]
    Z = np.array(morph_coor_rot[i])[:,2]
    max_range = np.array([Y.max()-Y.min(), Z.max()-Z.min()]).max()
    max_range_list.append(max_range)

max_range = np.max(max_range_list)

N_cluster = len(np.unique(ind2_spiny))

cmap = cm.get_cmap('viridis', N_cluster)

for i,c in enumerate(np.unique(ind2_spiny)[np.argsort(ind2_spiny_rgy)[::-1]]):
    dirpath = './Allenfigures/spiny_c' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    for n,j in enumerate(spiny_idx[np.where(ind2_spiny == c)]):
        
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        tararr = np.array(morph_coor_rot[j])
        somaIdx = np.where(np.array(morph_parent[j]) < 0)[0]
        for p in range(len(morph_parent[j])):
            if morph_parent[j][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_coor_rot[j]
                                        [morph_id[j].index(morph_parent[j][p])], morph_coor_rot[j][p]))
                if i == len(ind2_spiny_idx_sort)-2:
                    plt.plot(morph_line[:,1], morph_line[:,2], color=(0.798216, 0.280197, 0.469538, 1.0), lw=1)
                elif i == len(ind2_spiny_idx_sort)-1:
                    plt.plot(morph_line[:,1], morph_line[:,2], color=(1.0, 0.25, 0.0, 1.0), lw=1)
                else:
                    plt.plot(morph_line[:,1], morph_line[:,2], color=cmap(i), lw=1)
        
        Y = np.array(morph_coor_rot[j])[:,1]
        Z = np.array(morph_coor_rot[j])[:,2]
        
        ax.set_title(neuron_id[j], fontsize=15)
        ax.set_xlim((0.5*(Y.max()+Y.min())-0.5*max_range, 0.5*(Y.max()+Y.min())+0.5*max_range))
        ax.set_ylim((0.5*(Z.max()+Z.min())-0.5*max_range, 0.5*(Z.max()+Z.min())+0.5*max_range))

        plt.tight_layout()
        # plt.savefig(os.path.join(dirpath, neuron_id[j]), dpi=300, bbox_inches='tight')
        plt.close()


N_cluster = len(np.unique(ind2_aspiny))

cmap = cm.get_cmap('viridis', N_cluster)

for i,c in enumerate(np.unique(ind2_aspiny)[np.argsort(ind2_aspiny_rgy)[::-1]]):
    dirpath = './Allenfigures/aspiny_c' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    for n,j in enumerate(aspiny_idx[np.where(ind2_aspiny == c)]):
        
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        tararr = np.array(morph_coor_rot[j])
        somaIdx = np.where(np.array(morph_parent[j]) < 0)[0]
        for p in range(len(morph_parent[j])):
            if morph_parent[j][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_coor_rot[j]
                                        [morph_id[j].index(morph_parent[j][p])], morph_coor_rot[j][p]))
                if i == len(ind2_aspiny_idx_sort)-2:
                    plt.plot(morph_line[:,1], morph_line[:,2], color=(0.798216, 0.280197, 0.469538, 1.0), lw=1)
                elif i == len(ind2_aspiny_idx_sort)-1:
                    plt.plot(morph_line[:,1], morph_line[:,2], color=(1.0, 0.25, 0.0, 1.0), lw=1)
                else:
                    plt.plot(morph_line[:,1], morph_line[:,2], color=cmap(i), lw=1)
        
        Y = np.array(morph_coor_rot[j])[:,1]
        Z = np.array(morph_coor_rot[j])[:,2]
        
        ax.set_title(neuron_id[j], fontsize=15)
        ax.set_xlim((0.5*(Y.max()+Y.min())-0.5*max_range, 0.5*(Y.max()+Y.min())+0.5*max_range))
        ax.set_ylim((0.5*(Z.max()+Z.min())-0.5*max_range, 0.5*(Z.max()+Z.min())+0.5*max_range))

        plt.tight_layout()
        # plt.savefig(os.path.join(dirpath, neuron_id[j]), dpi=300, bbox_inches='tight')
        plt.close()


#%% mtype Comparison with Gouwens et al.

m_type_aspiny = np.array(['Aspiny_1', 'Aspiny_2', 'Aspiny_3', 'Aspiny_4', 'Aspiny_5',
                          'Aspiny_6', 'Aspiny_7', 'Aspiny_8', 'Aspiny_9', 'Aspiny_10', 
                          'Aspiny_11', 'Aspiny_12', 'Aspiny_13', 'Aspiny_14', 
                          'Aspiny_15', 'Aspiny_16', 'Aspiny_17', 'Aspiny_18', 'Aspiny_19'])
       
m_type_spiny = np.array(['Spiny_1', 'Spiny_2', 'Spiny_3', 'Spiny_4', 'Spiny_5', 'Spiny_6', 'Spiny_7',
                         'Spiny_8', 'Spiny_9', 'Spiny_10', 'Spiny_11', 'Spiny_12', 'Spiny_13', 'Spiny_14',
                         'Spiny_15', 'Spiny_16', 'Spiny_17', 'Spiny_18', 'Spiny_19'])

m_nid_aspiny = []

for i in m_type_aspiny:
    idx = np.where(gouwens_df['m-type'] == i)[0]
    m_nid_aspiny.append(np.array(gouwens_df['specimen_id'].iloc[idx]))
    
m_nid_spiny = []

for i in m_type_spiny:
    idx = np.where(gouwens_df['m-type'] == i)[0]
    m_nid_spiny.append(np.array(gouwens_df['specimen_id'].iloc[idx]))

m_obs_aspiny = []
m_c_aspiny = []

for i in range(len(m_nid_aspiny)):
    m_obs_aspiny_temp = []
    for j in m_nid_aspiny[i]:
        if str(j) in neuron_id:
            m_obs_aspiny_temp.append(ind2_aspiny[np.argwhere(aspiny_idx == np.where(neuron_id == str(j))[0])[0]][0])
            m_c_aspiny.append(m_type_aspiny[i])
    m_obs_aspiny.append(m_obs_aspiny_temp)
    
        
m_obs_spiny = []
m_c_spiny = []

for i in range(len(m_nid_spiny)):
    m_obs_spiny_temp = []
    for j in m_nid_spiny[i]:
        if str(j) in neuron_id:
            m_obs_spiny_temp.append(ind2_spiny[np.argwhere(spiny_idx == np.where(neuron_id == str(j))[0])[0]][0])
            m_c_spiny.append(m_type_spiny[i])
    m_obs_spiny.append(m_obs_spiny_temp)

    
#%%

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi_val, p_val, dof, expected = scipy.stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi_val/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))), p_val
    
m_nid_aspiny_flat = [item for sublist in m_obs_aspiny for item in sublist]
m_nid_spiny_flat = [item for sublist in m_obs_spiny for item in sublist]

print(cramers_v(np.array(m_nid_spiny_flat), np.array(m_c_spiny)))
print(cramers_v(np.array(m_nid_aspiny_flat), np.array(m_c_aspiny)))


#%% Metric testing

branch_length_average = 40.99789638178139
rgymean = 295.33080920530085

i1 = np.argmin(np.abs(q - 2*np.pi/rgymean))
i2 = np.argmin(np.abs(q - 2*np.pi/branch_length_average))+1

aspiny_idx = np.where(dendrite_type != 'spiny')[0]
spiny_idx = np.where(dendrite_type == 'spiny')[0]

ind2_aspiny_idx_sort_all = []
ind2_spiny_idx_sort_all = []

for n in range(5):
    fq_dist = np.zeros((len(fp), len(fp)))
    N_spiny = Counter(dendrite_type)['spiny']
    N_aspiny = len(fp) - N_spiny
    fq_dist_spiny = np.zeros((N_spiny, N_spiny))
    fq_dist_aspiny = np.zeros((N_aspiny, N_aspiny))
    
    for i in range(len(fp)):
        for j in range(len(fp)):
            exp_data = np.zeros((i2-i1, 2))
            exp_data[:,0] = np.log10(q)[i1:i2]
            exp_data[:,1] = np.log10(fq[i,i1:i2])
            num_data = np.zeros((i2-i1, 2))
            num_data[:,0] = np.log10(q)[i1:i2]
            num_data[:,1] = np.log10(fq[j,i1:i2])
            if n == 0:
                # L2
                fq_dist[i][j] = np.linalg.norm(np.log10(fq[i,i1:i2])-np.log10(fq[j,i1:i2]))
            elif n == 1:
                # L1
                fq_dist[i][j] = np.linalg.norm((np.log10(fq[i,i1:i2])-np.log10(fq[j,i1:i2])), ord=1)
            elif n == 2:
                # cosine
                fq_dist[i][j] = scipy.spatial.distance.cosine(np.log10(fq[i,i1:i2]), np.log10(fq[j,i1:i2]))
            elif n == 3:
                # Frechet
                fq_dist[i][j] = sm.frechet_dist(exp_data[:,1], num_data[:,1])
            else:
                # PCM
                fq_dist[i][j] = sm.pcm(exp_data, num_data)
    
    for i in range(N_aspiny):
        for j in range(N_aspiny):
            exp_data = np.zeros((i2-i1, 2))
            exp_data[:,0] = np.log10(q)[i1:i2]
            exp_data[:,1] = np.log10(fq[aspiny_idx[i],i1:i2])
            num_data = np.zeros((i2-i1, 2))
            num_data[:,0] = np.log10(q)[i1:i2]
            num_data[:,1] = np.log10(fq[aspiny_idx[j],i1:i2])
            if n == 0:
                # L2
                fq_dist_aspiny[i][j] = np.linalg.norm(np.log10(fq[aspiny_idx[i],i1:i2])-np.log10(fq[aspiny_idx[j],i1:i2]))
            elif n == 1:
                # L1
                fq_dist_aspiny[i][j] = np.linalg.norm((np.log10(fq[aspiny_idx[i],i1:i2])-np.log10(fq[aspiny_idx[j],i1:i2])), ord=1)
            elif n == 2:
                # cosine
                fq_dist_aspiny[i][j] = scipy.spatial.distance.cosine(np.log10(fq[aspiny_idx[i],i1:i2]), np.log10(fq[aspiny_idx[j],i1:i2]))
            elif n == 3:
                # Frechet
                fq_dist_aspiny[i][j] = sm.frechet_dist(exp_data[:,1], num_data[:,1])
            else:
                # PCM
                fq_dist_aspiny[i][j] = sm.pcm(exp_data, num_data)
    
    for i in range(N_spiny):
        for j in range(N_spiny):
            exp_data = np.zeros((i2-i1, 2))
            exp_data[:,0] = np.log10(q)[i1:i2]
            exp_data[:,1] = np.log10(fq[spiny_idx[i],i1:i2])
            num_data = np.zeros((i2-i1, 2))
            num_data[:,0] = np.log10(q)[i1:i2]
            num_data[:,1] = np.log10(fq[spiny_idx[j],i1:i2])
            if n == 0:
                # L2
                fq_dist_spiny[i][j] = np.linalg.norm(np.log10(fq[spiny_idx[i],i1:i2])-np.log10(fq[spiny_idx[j],i1:i2]))
            elif n == 1:
                # L1
                fq_dist_spiny[i][j] = np.linalg.norm((np.log10(fq[spiny_idx[i],i1:i2])-np.log10(fq[spiny_idx[j],i1:i2])), ord=1)
            elif n == 2:
                # cosine
                fq_dist_spiny[i][j] = scipy.spatial.distance.cosine(np.log10(fq[spiny_idx[i],i1:i2]), np.log10(fq[spiny_idx[j],i1:i2]))
            elif n == 3:
                # Frechet
                fq_dist_spiny[i][j] = sm.frechet_dist(exp_data[:,1], num_data[:,1])
            else:
                # PCM
                fq_dist_spiny[i][j] = sm.pcm(exp_data, num_data)
    
    link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_dist), 
                                           method='complete', optimal_ordering=True)
    
    link_aspiny = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_dist_aspiny), 
                                           method='complete', optimal_ordering=True)
    
    link_spiny = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_dist_spiny), 
                                           method='complete', optimal_ordering=True)
    
    
    ind2 = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_dist), minClusterSize=4)['labels']
    
    ind2_aspiny = cutreeHybrid(link_aspiny, scipy.spatial.distance.squareform(fq_dist_aspiny), minClusterSize=4)['labels']
    
    ind2_spiny = cutreeHybrid(link_spiny, scipy.spatial.distance.squareform(fq_dist_spiny), minClusterSize=4)['labels']
    
    ind2_aspiny_idx = []
    
    for i in np.unique(ind2_aspiny):
        ind2_aspiny_idx.append(list(aspiny_idx[np.where(ind2_aspiny == i)[0]]))
    
    ind2_spiny_idx = []
    
    for i in np.unique(ind2_spiny):
        ind2_spiny_idx.append(list(spiny_idx[np.where(ind2_spiny == i)[0]]))
    
    ind2_aspiny_rgy = []
    
    for i,j in enumerate(ind2_aspiny_idx):
        rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]), 
                                     list(np.array(morph_dia, dtype=object)[j]))
        ind2_aspiny_rgy.append(np.mean(rgyb))
    
    ind2_aspiny_idx_sort = list(np.array(ind2_aspiny_idx, dtype=object)[np.argsort(ind2_aspiny_rgy)[::-1]])
    
    ind2_spiny_rgy = []
    
    for i,j in enumerate(ind2_spiny_idx):
        rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]), 
                                     list(np.array(morph_dia, dtype=object)[j]))
        ind2_spiny_rgy.append(np.mean(rgyb))
    
    ind2_spiny_idx_sort = list(np.array(ind2_spiny_idx, dtype=object)[np.argsort(ind2_spiny_rgy)[::-1]])
    
    ind2_aspiny_idx_sort_all.append(ind2_aspiny_idx_sort)
    ind2_spiny_idx_sort_all.append(ind2_spiny_idx_sort)

#%% Metric testing plotting

import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost

ind2_aspiny_idx_sort_all_flat = []

aspiny_ind_chg = np.empty((len(ind2_aspiny_idx_sort_all), N_aspiny))

for i,j in enumerate(ind2_aspiny_idx_sort_all):
    ind2_aspiny_idx_sort_all_flat.append([item for sublist in j for item in sublist])
    idx = 0
    c = 1
    for k in j:
        aspiny_ind_chg[i][idx:idx+len(k)] = c
        idx = idx+len(k)
        c += 1

for i in range(len(ind2_aspiny_idx_sort_all_flat)-1):
    temp = []
    for j in ind2_aspiny_idx_sort_all_flat[0]:
        temp.append(np.where(j == ind2_aspiny_idx_sort_all_flat[i+1])[0][0])
    aspiny_ind_chg[i+1] = aspiny_ind_chg[i+1][temp]
    
ind2_spiny_idx_sort_all_flat = []

spiny_ind_chg = np.empty((len(ind2_spiny_idx_sort_all), N_spiny))

for i,j in enumerate(ind2_spiny_idx_sort_all):
    ind2_spiny_idx_sort_all_flat.append([item for sublist in j for item in sublist])
    idx = 0
    c = 1
    for k in j:
        spiny_ind_chg[i][idx:idx+len(k)] = c
        idx = idx+len(k)
        c += 1

for i in range(len(ind2_spiny_idx_sort_all_flat)-1):
    temp = []
    for j in ind2_spiny_idx_sort_all_flat[0]:
        temp.append(np.where(j == ind2_spiny_idx_sort_all_flat[i+1])[0][0])
    spiny_ind_chg[i+1] = spiny_ind_chg[i+1][temp]    

aspiny_ind_chg[1][aspiny_ind_chg[1] == 7] = 8
aspiny_ind_chg[2][aspiny_ind_chg[2] == 3] = 5
aspiny_ind_chg[3][aspiny_ind_chg[3] == 5] = 12
aspiny_ind_chg[3][aspiny_ind_chg[3] == 8] = 5
aspiny_ind_chg[3][aspiny_ind_chg[3] == 11] = 8
aspiny_ind_chg[3][aspiny_ind_chg[3] == 12] = 11
aspiny_ind_chg[3][aspiny_ind_chg[3] == 4] = 13
aspiny_ind_chg[3][aspiny_ind_chg[3] == 7] = 4
aspiny_ind_chg[3][aspiny_ind_chg[3] == 13] = 7
aspiny_ind_chg[3][aspiny_ind_chg[3] == 6] = 14
aspiny_ind_chg[3][aspiny_ind_chg[3] == 9] = 6
aspiny_ind_chg[3][aspiny_ind_chg[3] == 14] = 9
aspiny_ind_chg[3][aspiny_ind_chg[3] == 2] = 15
aspiny_ind_chg[3][aspiny_ind_chg[3] == 3] = 2
aspiny_ind_chg[3][aspiny_ind_chg[3] == 7] = 3
aspiny_ind_chg[3][aspiny_ind_chg[3] == 10] = 7
aspiny_ind_chg[3][aspiny_ind_chg[3] == 15] = 10
    
fig = plt.figure(figsize=(10,1))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(aspiny_ind_chg, cmap='tab20', aspect='auto')
ax1.set_xticks([]) 
ax1.set_yticks([]) 
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax3.axis["right"].set_visible(False)
ax3.set_yticks(np.arange(5))
ax3.invert_yaxis()
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(5) + 0.5)))
# ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(['Euclidean', 'Manhattan', 'Cosine', 'Frechet']))
ax3.axis["left"].minor_ticklabels.set(fontsize=6, rotation_mode='default')
# plt.savefig('./Allenfigures/aspiny_metric_test_1.svg', dpi=300, bbox_inches='tight')
plt.show()

spiny_ind_chg[1][spiny_ind_chg[1] == 3] = 9
spiny_ind_chg[1][spiny_ind_chg[1] == 4] = 3
spiny_ind_chg[1][spiny_ind_chg[1] == 6] = 10
spiny_ind_chg[1][spiny_ind_chg[1] == 9] = 6
spiny_ind_chg[1][spiny_ind_chg[1] == 5] = 12
spiny_ind_chg[1][spiny_ind_chg[1] == 2] = 5
spiny_ind_chg[1][spiny_ind_chg[1] == 8] = 11
spiny_ind_chg[1][spiny_ind_chg[1] == 12] = 8
spiny_ind_chg[1][spiny_ind_chg[1] == 7] = 9
spiny_ind_chg[2][spiny_ind_chg[2] == 1] = 7
spiny_ind_chg[2][spiny_ind_chg[2] == 3] = 8
spiny_ind_chg[2][spiny_ind_chg[2] == 4] = 10

spiny_ind_chg[3][spiny_ind_chg[3] == 6] = 14
spiny_ind_chg[3][spiny_ind_chg[3] == 7] = 6
spiny_ind_chg[3][spiny_ind_chg[3] == 14] = 7
spiny_ind_chg[3][spiny_ind_chg[3] == 8] = 15
spiny_ind_chg[3][spiny_ind_chg[3] == 9] = 8
spiny_ind_chg[3][spiny_ind_chg[3] == 15] = 9
spiny_ind_chg[3][spiny_ind_chg[3] == 11] = 16
spiny_ind_chg[3][spiny_ind_chg[3] == 12] = 11
spiny_ind_chg[3][spiny_ind_chg[3] == 16] = 12


fig = plt.figure(figsize=(10,1))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(spiny_ind_chg, cmap='tab20', aspect='auto')
ax1.set_xticks([])
ax1.set_yticks([])
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax3.axis["right"].set_visible(False)
ax3.set_yticks(np.arange(5))
ax3.invert_yaxis()
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(5) + 0.5)))
# ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(['Euclidean', 'Manhattan', 'Cosine', 'Frechet']))
ax3.axis["left"].minor_ticklabels.set(fontsize=6, rotation_mode='default')
# plt.savefig('./Allenfigures/spiny_metric_test_1.svg', dpi=300, bbox_inches='tight')
plt.show()



