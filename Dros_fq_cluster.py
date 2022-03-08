# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 13:55:26 2021

@author: user
"""

import os
import numpy as np
import pandas as pd
import scipy.cluster
import sklearn.metrics
from scipy.signal import argrelextrema, savgol_filter
from scipy.spatial.transform import Rotation
from dynamicTreeCut import cutreeHybrid
import similaritymeasures as sm
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter
import copy

os.chdir(os.path.dirname(__file__))

PATH = r'./Dros_fq'

fp = [f for f in os.listdir(PATH) if os.path.exists(os.path.join(PATH, f))]
neuron_id = copy.deepcopy(fp)
neuron_id = [e[:-4] for e in neuron_id]
neuron_id = np.array(neuron_id)
fp = [os.path.join(PATH, f) for f in fp]

q_full = np.logspace(-3, 3, 601)

fq_full = np.empty((len(fp), len(q_full)))

for i,j in enumerate(fp):
    fq_full[i] = np.load(j)

#%%

PATH2 = r'../PolymerConnectome/TEMCA2/Skels connectome_mod'

fp2 = [w.replace(PATH, PATH2) for w in fp]
fp2 = [w.replace('.npy', '.swc') for w in fp2]

morph_coor = []
morph_len = []
morph_parent = []
morph_id = []
somaP = []
d2d = []
branchNum = np.empty(len(fp2))
branch_coor = []
branchTrk = []
length_branch = []
MBcoor = []
MBcoor_trk = []
MBcoor_per_n = []
LHcoor = []
LHcoor_trk = []
LHcoor_per_n = []
ALcoor = []
ALcoor_trk = []
ALcoor_per_n = []

length_MB = []
length_LH = []
length_AL = []

MB_branchTrk = []
MB_branchP = []
MB_endP = []
LH_branchTrk = []
LH_branchP = []
LH_endP = []
AL_branchTrk = []
AL_branchP = []
AL_endP = []

r_d_x = -10
r_rad_x = np.radians(r_d_x)
r_x = np.array([0, 1, 0])
r_vec_x = r_rad_x * r_x
rotx = Rotation.from_rotvec(r_vec_x)

r_d_y = -25
r_rad_y = np.radians(r_d_y)
r_y = np.array([0, 1, 0])
r_vec_y = r_rad_y * r_y
roty = Rotation.from_rotvec(r_vec_y)

r_d_z = -40
r_rad_z = np.radians(r_d_z)
r_z = np.array([0, 1, 0])
r_vec_z = r_rad_z * r_z
rotz = Rotation.from_rotvec(r_vec_z)

for f in range(len(fp2)):
    print(f, fp2[f])
    morph_neu_id = []
    morph_neu_parent = []
    morph_neu_prox = []
    morph_neu_dist = []
    
    gen = np.genfromtxt(fp2[f])
    
    df = pd.DataFrame(gen)
    df.iloc[:,0] = df.iloc[:, 0].astype(int)
    df.iloc[:,1] = df.iloc[:, 1].astype(int)
    df.iloc[:,6] = df.iloc[:, 6].astype(int)
    
    scall = int(df.iloc[np.where(df[6] == -1)[0]].values[0][0])
    somaP.append(scall)
    ctr = Counter(df[6].tolist())
    ctrVal = list(ctr.values())
    ctrKey = list(ctr.keys())
    branchNum[f] = sum(i > 1 for i in ctrVal)
    branchInd = np.array(ctrKey)[np.where(np.array(ctrVal) > 1)[0]]
    branchInd = branchInd[branchInd != -1]
    
    morph_id.append(df[0].tolist())
    morph_parent.append(df[6].tolist())
    morph_coor.append(np.divide(np.array(df[[2,3,4]]), 1000).tolist())
    morph_len.append(len(morph_coor[-1]))

    list_end = np.setdiff1d(morph_id[f], morph_parent[f])
    list_end = np.unique(list_end)
    
    bPoint = np.append(branchInd, list_end)
    bPoint = np.unique(bPoint)
    
    neu_branchTrk = []
    d2d_temp = []
    length_branch_temp = []
    branch_dist_temp1 = []
    MBcoor_per_n_temp = []
    LHcoor_per_n_temp = []
    ALcoor_per_n_temp = []
    length_MB_per_n = []
    length_LH_per_n = []
    length_AL_per_n = []
    MB_branchTrk_temp = []
    MB_branchP_temp = []
    LH_branchTrk_temp = []
    LH_branchP_temp = []
    AL_branchTrk_temp = []
    AL_branchP_temp = []
    
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
                
                # rotate -25 degrees on y-axis
                branch_dist_temp2_rot = roty.apply(branch_dist_temp2)
                
                # rotate -35 degrees on x-axis
                branch_dist_temp2_rot2 = rotx.apply(branch_dist_temp2)
                
                # rotate 50 degrees on z-axis
                branch_dist_temp2_rot3 = rotz.apply(branch_dist_temp2)
                
                if ((np.array(branch_dist_temp2_rot)[:,0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[:,0] < 426.14).all() and
                    (np.array(branch_dist_temp2_rot)[:,1] > 176.68).all() and (np.array(branch_dist_temp2_rot)[:,1] < 272.91).all() and
                    (np.array(branch_dist_temp2_rot3)[:,2] > 434.08).all() and (np.array(branch_dist_temp2_rot3)[:,2] < 496.22).all()):
                    MBcoor.append(branch_dist_temp2)
                    MBcoor_trk.append(f)
                    MBcoor_per_n_temp.append(branch_dist_temp2)
                    length_MB_per_n.append(dist)
                    MB_branchTrk_temp.append(neu_branchTrk_temp)
                    MB_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                elif ((np.array(branch_dist_temp2_rot)[:,0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[:,1] > 176.68).all() and
                      (np.array(branch_dist_temp2_rot)[:,1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[:,2] > 286.78).all() and
                      (np.array(branch_dist_temp2_rot)[:,2] < 343.93).all()):
                    LHcoor.append(branch_dist_temp2)
                    LHcoor_trk.append(f)
                    LHcoor_per_n_temp.append(branch_dist_temp2)
                    length_LH_per_n.append(dist)
                    LH_branchTrk_temp.append(neu_branchTrk_temp)
                    LH_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                elif ((np.array(branch_dist_temp2_rot)[:,0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[:,0] < 533.42).all() and 
                      (np.array(branch_dist_temp2_rot)[:,1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[:,1] < 363.12).all() and
                      (np.array(branch_dist_temp2_rot2)[:,2] < 180.77).all()):
                    ALcoor.append(branch_dist_temp2)
                    ALcoor_trk.append(f)
                    ALcoor_per_n_temp.append(branch_dist_temp2)
                    length_AL_per_n.append(dist)
                    AL_branchTrk_temp.append(neu_branchTrk_temp)
                    AL_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                
    branchTrk.append(neu_branchTrk)
    branch_coor.append(branch_dist_temp1)
    length_branch.append(length_branch_temp)
    d2d.append(d2d_temp)
    
    MBcoor_per_n.append(MBcoor_per_n_temp)
    LHcoor_per_n.append(LHcoor_per_n_temp)
    ALcoor_per_n.append(ALcoor_per_n_temp)
    length_MB.append(length_MB_per_n)
    length_LH.append(length_LH_per_n)
    length_AL.append(length_AL_per_n)
    MB_branchTrk.append(MB_branchTrk_temp)
    MB_branchP.append(np.unique([item for sublist in MB_branchP_temp for item in sublist]).tolist())
    LH_branchTrk.append(LH_branchTrk_temp)
    LH_branchP.append(np.unique([item for sublist in LH_branchP_temp for item in sublist]).tolist())
    AL_branchTrk.append(AL_branchTrk_temp)
    AL_branchP.append(np.unique([item for sublist in AL_branchP_temp for item in sublist]).tolist())
    

length_branch_flat_full = [item for sublist in length_branch for item in sublist]

branch_length_average_full = np.mean(length_branch_flat_full)

d2df = [item for sublist in d2d for item in sublist]

ALcoor_per_n[92] = []

def radiusOfGyration(morph_coor):
    cML = np.empty((len(morph_coor), 3))
    rGy = np.empty(len(morph_coor))
    for i in range(len(morph_coor)):
        cML[i] = np.average(np.array(morph_coor[i]), axis=0)
        rList = scipy.spatial.distance.cdist(np.array(morph_coor[i]), 
                                              np.array([cML[i]])).flatten()
        rGy[i] = np.sqrt(np.average(np.square(rList)))
    
    return (rGy, cML)


#%% Branch length histogram

fig = plt.figure(figsize=(6,4))
plt.hist(length_branch_flat_full, bins=46, density=True)
plt.xlabel('Branch Length ($l$)', fontsize=15)
plt.ylabel('Probability Density', fontsize=15)
plt.ylim(0, 0.005)
# plt.xlim(-10, 20)
# plt.savefig('./Drosfigures/l_dist_full_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%

branch_length_average_full = 2.789401932780958

rgymean_full = 2*72.07108928773508

i1 = np.argmin(np.abs(q_full - 2*np.pi/rgymean_full))
i2 = np.argmin(np.abs(q_full - 2*np.pi/branch_length_average_full))+1

fq_full_dist = np.zeros((len(fp), len(fp)))

for i in range(len(fp)):
    for j in range(len(fp)):
        exp_data = np.zeros((i2-i1, 2))
        exp_data[:,0] = np.log10(q_full)[i1:i2]
        exp_data[:,1] = np.log10(fq_full[i,i1:i2])
        num_data = np.zeros((i2-i1, 2))
        num_data[:,0] = np.log10(q_full)[i1:i2]
        num_data[:,1] = np.log10(fq_full[j,i1:i2])
        # L2
        # fq_full_dist[i][j] = np.linalg.norm((np.log10(fq_full[i,i1:i2])-np.log10(fq_full[j,i1:i2])))
        # L1
        # fq_full_dist[i][j] = np.linalg.norm((np.log10(fq_full[i,i1:i2])-np.log10(fq_full[j,i1:i2])), ord=1)
        # cosine
        # fq_full_dist[i][j] = scipy.spatial.distance.cosine(np.log10(fq_full[i,i1:i2]), np.log10(fq_full[j,i1:i2]))
        # Frechet
        fq_full_dist[i][j] = sm.frechet_dist(exp_data[:,1], num_data[:,1])
        # PCM
        fq_full_dist[i][j] = sm.pcm(exp_data, num_data)

link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_full_dist), 
                                       method='complete', optimal_ordering=True)

#%% Dynamic tree cut

ind_full = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_full_dist), minClusterSize=1)['labels']

#%% Maximum silhouette

silhoutte_full = []

for k in np.arange(2, 70):
    sil_full = scipy.cluster.hierarchy.fcluster(link, k, 'maxclust')
    silhoutte_full.append(sklearn.metrics.silhouette_score(fq_full_dist, sil_full, metric="precomputed"))

ind_full = scipy.cluster.hierarchy.fcluster(link, np.arange(2,70)[np.argmax(silhoutte_full)], 'maxclust')

#%%

ind_full_idx = []

for i in np.unique(ind_full):
    ind_full_idx.append(np.where(ind_full == i)[0])

ind_full_idx_rgy = []

for i,j in enumerate(ind_full_idx):
    rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]))
    ind_full_idx_rgy.append(np.mean(rgyb))

ind_full_idx_sort = list(np.array(ind_full_idx, dtype=object)[np.argsort(ind_full_idx_rgy)[::-1]])

#%%

cmap = cm.get_cmap('viridis', len(ind_full_idx_sort))

for i,j in enumerate(ind_full_idx_sort):
    bl = list(np.array(length_branch, dtype=object)[j])
    blf = [item for sublist in bl for item in sublist]
    
    rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]))
    
    d2db = list(np.array(d2d, dtype=object)[j])
    d2db = [item for sublist in d2db for item in sublist]
    
    fig = plt.figure(figsize=(6,4))
    if i == len(ind_full_idx_sort)-2:
        plt.plot(q_full, fq_full[j].T, color=(0.798216, 0.280197, 0.469538, 1.0))
    elif i == len(ind_full_idx_sort)-1:
        plt.plot(q_full, fq_full[j].T, color=(1.0, 0.25, 0.0, 1.0))
    else:
        plt.plot(q_full, fq_full[j].T, color=cmap(i))
    plt.vlines(2*np.pi/np.mean(blf), 1e-5, 10, color='k', ls='dashed')
    plt.vlines(2*np.pi/(2*np.mean(rgyb)), 1e-5, 10, color='k', ls='dotted')
    
    line1 = 2e-4*np.power(q_full, -16/7)
    line2 = 7e-10*np.power(q_full, -4/1)
    line3 = 3e-4*np.power(q_full, -1/0.395)
    line4 = 1/100*np.power(q_full, -1)
    line5 = 1e-3*np.power(q_full, -2/1)
    
    # if i == 0:
    #     plt.plot(q_full[130:157], line1[130:157], lw=1.5, color='tab:blue')
    #     plt.plot(q_full[110:147], line2[110:147], lw=1.5, color='tab:red')
    #     plt.plot(q_full[180:207], line3[180:207], lw=1.5, color='tab:purple')
    #     plt.plot(q_full[220:270], line4[220:270], lw=1.5, color='k')
    #     plt.plot(q_full[150:200], line5[200:250], lw=1.5, color='tab:green')
        
    #     plt.text(0.03, 8e-1, r'$\nu = \dfrac{7}{16}$', fontsize=13, color='tab:blue')
    #     plt.text(0.007, 1e-2, r'$\nu = \dfrac{1}{4}$', fontsize=13, color='tab:red')
    #     plt.text(0.1, 2e-1, r'$\nu = 0.395$', fontsize=13, color='tab:purple')
    #     plt.text(0.4, 1e-2, r'$\nu = 1$', fontsize=13, color='k')
    #     plt.text(0.03, 1e-2, r'$\nu = \dfrac{1}{2}$', fontsize=13, color='tab:green')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-3, 20)
    plt.ylim(1e-4, 3)
    plt.ylabel('$F(q)$', fontsize=18)
    plt.xlabel('$q$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title("$C^{Full}_{" + str(i+1) + "}$", fontsize=15, pad=10)
    # plt.savefig('./Drosfigures/Fq_Full_c' + str(i+1) + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()


#%% Morph 3D per cluster

cmap = cm.get_cmap('viridis', len(ind_full_idx_sort))

for i,j in enumerate(ind_full_idx_sort):
    fig = plt.figure(figsize=(24, 16))
    ax = plt.axes(projection='3d')
    ax.set_xlim(400, 600)
    ax.set_ylim(400, 150)
    ax.set_zlim(50, 200)
    ax.axis('off')
    
    for f in j:
        somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
        for p in range(len(morph_parent[f])):
            if morph_parent[f][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_coor[f][morph_id[f].index(morph_parent[f][p])], morph_coor[f][p]))
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)
    # plt.savefig('./Drosfigures/morph_Dros_full_3D_1_c' + str(i+1) + '.png', dpi=300, bbox_inches='tight')
    plt.close()


#%% Morph 3D per cluster full

cmap = cm.get_cmap('viridis', len(ind_full_idx_sort))

fig = plt.figure(figsize=(24, 16))
ax = plt.axes(projection='3d')
ax.set_xlim(400, 600)
ax.set_ylim(400, 150)
ax.set_zlim(50, 200)
ax.axis('off')

for i,j in enumerate(ind_full_idx_sort):
    for f in j:
        somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
        for p in range(len(morph_parent[f])):
            if morph_parent[f][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_coor[f][morph_id[f].index(morph_parent[f][p])], morph_coor[f][p]))
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)
                
# plt.savefig('./Drosfigures/morph_Dros_full_3D_full_1.png', dpi=300, bbox_inches='tight')
plt.close()

#%% Morph per neuron per cluster

cmap = cm.get_cmap('viridis', len(ind_full_idx_sort))

for i,j in enumerate(ind_full_idx_sort):
    dirpath = './Drosfigures/Full_C' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    for f in j:
        fig = plt.figure(figsize=(16, 16))
        # ax = plt.axes(projection='3d')
        ax = fig.add_subplot(111)
        ax.set_xlim(375, 625)
        # ax.set_ylim(400, 150)
        ax.set_ylim(0, 250)
        ax.axis('off')
        
        somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
        for p in range(len(morph_parent[f])):
            if morph_parent[f][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_coor[f][morph_id[f].index(morph_parent[f][p])], morph_coor[f][p]))
                plt.plot(morph_line[:,0], morph_line[:,2], color=cmap(i))
        plt.savefig(os.path.join(dirpath, neuron_id[f]), dpi=300, bbox_inches='tight')
        plt.close()


#%% AL Fq

PATH = r'./Dros_AL_fq'

fp_AL = [f for f in os.listdir(PATH) if os.path.exists(os.path.join(PATH, f))]
AL_id = copy.deepcopy(fp_AL)
AL_id = [e[:-4] for e in AL_id]
AL_id = np.array(AL_id)
fp_AL = [os.path.join(PATH, f) for f in fp_AL]

fp_AL.pop(73)

fq_AL = np.empty((len(fp_AL), len(q_full)))

for i,j in enumerate(fp_AL):
    fq_AL[i] = np.load(j)

    
#%%

length_branch_flat_AL = [item for sublist in length_AL for item in sublist]

branch_length_average_AL = np.mean(length_branch_flat_AL)

ALcoor_per_n_flat = []

for i in ALcoor_per_n:
    temp = [item for sublist in i for item in sublist]
    if len(temp) > 0:
        ALcoor_per_n_flat.append(temp)

rgy_AL, cML = radiusOfGyration(ALcoor_per_n_flat)

rgymean_AL = 2*np.mean(rgy_AL)

i1 = np.argmin(np.abs(q_full - 2*np.pi/rgymean_AL))
i2 = np.argmin(np.abs(q_full - 2*np.pi/branch_length_average_AL))+1

fq_AL_dist = np.zeros((len(fq_AL), len(fq_AL)))

for i in range(len(fq_AL)):
    for j in range(len(fq_AL)):
        fq_AL_dist[i][j] = np.linalg.norm(np.log10(fq_AL[i,i1:i2])-np.log10(fq_AL[j,i1:i2]))

link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_AL_dist), 
                                       method='complete', optimal_ordering=True)

#%% Dynamic tree cut

from dynamicTreeCut import cutreeHybrid

ind_AL = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_AL_dist), minClusterSize=1)['labels']

#%% Maximum silhouette

silhoutte_AL = []

for k in np.arange(2, 70):
    sil_AL = scipy.cluster.hierarchy.fcluster(link, k, 'maxclust')
    silhoutte_AL.append(sklearn.metrics.silhouette_score(fq_AL_dist, sil_AL, metric="precomputed"))

ind_AL = scipy.cluster.hierarchy.fcluster(link, np.arange(2,70)[np.argmax(silhoutte_AL)], 'maxclust')

#%%

ind_AL_idx = []

for i in np.unique(ind_AL):
    ind_AL_idx.append(np.where(ind_AL == i)[0])

ind_AL_idx_rgy = []

for i,j in enumerate(ind_AL_idx):
    rgyb, cmb = radiusOfGyration(list(np.array(ALcoor_per_n_flat, dtype=object)[j]))
    ind_AL_idx_rgy.append(np.mean(rgyb))

ind_AL_idx_sort = list(np.array(ind_AL_idx, dtype=object)[np.argsort(ind_AL_idx_rgy)[::-1]])

#%%

length_AL_ = []

for i in length_AL:
    if len(i) > 0:
        length_AL_.append(i)

cmap = cm.get_cmap('viridis', len(ind_AL_idx_sort))

for i,j in enumerate(ind_AL_idx_sort):
    bl = list(np.array(length_AL_, dtype=object)[j])
    blf = [item for sublist in bl for item in sublist]
    
    rgyb, cmb = radiusOfGyration(list(np.array(ALcoor_per_n_flat, dtype=object)[j]))
    
    fig = plt.figure(figsize=(6,4))
    if i == len(ind_AL_idx_sort)-2:
        plt.plot(q_full, fq_AL[j].T, color=(0.798216, 0.280197, 0.469538, 1.0))
    elif i == len(ind_AL_idx_sort)-1:
        plt.plot(q_full, fq_AL[j].T, color=(1.0, 0.25, 0.0, 1.0))
    else:
        plt.plot(q_full, fq_AL[j].T, color=cmap(i))
    plt.vlines(2*np.pi/np.mean(blf), 1e-5, 10, color='k', ls='dashed')
    plt.vlines(2*np.pi/(2*np.mean(rgyb)), 1e-5, 10, color='k', ls='dotted')
    
    line1 = 2e-4*np.power(q_full, -16/7)
    line2 = 7e-10*np.power(q_full, -4/1)
    line3 = 3e-4*np.power(q_full, -1/0.395)
    line4 = 1/100*np.power(q_full, -1)
    line5 = 1e-3*np.power(q_full, -2/1)
    
    # if i == 0:
    #     plt.plot(q_full[130:157], line1[130:157], lw=1.5, color='tab:blue')
    #     plt.plot(q_full[110:147], line2[110:147], lw=1.5, color='tab:red')
    #     plt.plot(q_full[180:207], line3[180:207], lw=1.5, color='tab:purple')
    #     plt.plot(q_full[220:270], line4[220:270], lw=1.5, color='k')
    #     plt.plot(q_full[150:200], line5[200:250], lw=1.5, color='tab:green')
        
    #     plt.text(0.03, 8e-1, r'$\nu = \dfrac{7}{16}$', fontsize=13, color='tab:blue')
    #     plt.text(0.007, 1e-2, r'$\nu = \dfrac{1}{4}$', fontsize=13, color='tab:red')
    #     plt.text(0.1, 2e-1, r'$\nu = 0.395$', fontsize=13, color='tab:purple')
    #     plt.text(0.4, 1e-2, r'$\nu = 1$', fontsize=13, color='k')
    #     plt.text(0.03, 1e-2, r'$\nu = \dfrac{1}{2}$', fontsize=13, color='tab:green')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-2, 10)
    plt.ylim(1e-4, 3)
    plt.ylabel('$F(q)$', fontsize=18)
    plt.xlabel('$q$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title("$C^{AL}_{" + str(i+1) + "}$", fontsize=15, pad=10)
    # plt.savefig('./Drosfigures/Fq_AL_c' + str(i+1) + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()

#%% AL plot cluster individual neurons

ALcoor_per_n_ = []

for i in ALcoor_per_n:
    if len(i) > 0:
        ALcoor_per_n_.append(i)

cmap = cm.get_cmap('viridis', len(ind_AL_idx_sort))

for i,j in enumerate(ind_AL_idx_sort):
    dirpath = './Drosfigures/AL_C' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
    for f in j:
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        ax.set_xlim(475, 625)
        ax.set_ylim(0, 150)
        ax.axis('off')
        
        for p in ALcoor_per_n_[f]:
            for b in range(len(p)-1):
                morph_line = np.vstack((p[b], p[b+1]))
                if i == len(ind_AL_idx_sort)-2:
                    plt.plot(morph_line[:,0], morph_line[:,2], color=(0.798216, 0.280197, 0.469538, 1.0))
                elif i == len(ind_AL_idx_sort)-1:
                    plt.plot(morph_line[:,0], morph_line[:,2], color=(1.0, 0.25, 0.0, 1.0))
                else:
                    plt.plot(morph_line[:,0], morph_line[:,2], color=cmap(i))
        # plt.savefig(os.path.join(dirpath, AL_id[f]), dpi=300, bbox_inches='tight')
        plt.close()

#%% AL plot cluster all neurons

ALcoor_per_n_ = []

for i in ALcoor_per_n:
    if len(i) > 0:
        ALcoor_per_n_.append(i)

cmap = cm.get_cmap('viridis', len(ind_AL_idx_sort))

for i,j in enumerate(ind_AL_idx_sort):
    dirpath = './Drosfigures/AL_C' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
    fig = plt.figure(figsize=(24, 16))
    ax = plt.axes(projection='3d')
    ax.set_xlim(475, 625)
    ax.set_ylim(450, 300)
    ax.set_zlim(0, 150)
    ax.axis('off')
    
    for f in j:
        for p in ALcoor_per_n_[f]:
            for b in range(len(p)-1):
                morph_line = np.vstack((p[b], p[b+1]))
                plt.plot(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)
    plt.savefig(os.path.join(dirpath, 'AL_C' + str(i+1)), dpi=300, bbox_inches='tight')
    plt.close()


#%% AL plot all clusters

ALcoor_per_n_ = []

for i in ALcoor_per_n:
    if len(i) > 0:
        ALcoor_per_n_.append(i)

cmap = cm.get_cmap('viridis', len(ind_AL_idx_sort))

fig = plt.figure(figsize=(24, 16))
ax = plt.axes(projection='3d')
ax.set_xlim(475, 625)
ax.set_ylim(450, 300)
ax.set_zlim(0, 150)
ax.axis('off')

for i,j in enumerate(ind_AL_idx_sort):
    for f in j:
        for p in ALcoor_per_n_[f]:
            for b in range(len(p)-1):
                morph_line = np.vstack((p[b], p[b+1]))
                plt.plot(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)
                
# plt.savefig('./Drosfigures/morph_Dros_AL_full_1.png', dpi=300, bbox_inches='tight')
plt.close()

#%% AL plot cluster all neurons in 2D + boundary

from scipy.spatial import ConvexHull

ALcoor_per_n_ = []

for i in ALcoor_per_n:
    if len(i) > 0:
        ALcoor_per_n_.append(i)

ALcoor_flat = [item for sublist in ALcoor for item in sublist]

# ALcoor_flat = np.array(ALcoor_flat)[np.where(np.array(ALcoor_flat)[:,1] < 250)]
# ALcoor_flat = np.array(ALcoor_flat)[np.where(np.array(ALcoor_flat)[:,1] > 190)]
# ALcoor_flat = np.array(ALcoor_flat)[np.where(np.array(ALcoor_flat)[:,2] > 125)]

hull_AL = ConvexHull(np.array(ALcoor_flat))

tri_AL = []
for i in range(len(hull_AL.simplices)):
    tt = []
    for j in range(len(hull_AL.simplices[i])):
        tt.append(np.where(hull_AL.vertices == hull_AL.simplices[i][j])[0][0])
    tri_AL.append(tuple(tt))

cmap = cm.get_cmap('viridis', len(ind_AL_idx_sort))

for i,j in enumerate(ind_AL_idx_sort):
    dirpath = './Drosfigures/AL_C' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    for z in range(2):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        ax.set_box_aspect((1,1,1))
        ax.set_xlim(475, 625)
        ax.set_ylim(400, 250)
        ax.set_zlim(0, 150)
        ax.dist = 7
        ax.axis('off')
        
        if z == 0:
            hull_AL_test = ConvexHull(np.array(ALcoor_flat)[:,:2])
            vert = np.append(hull_AL_test.vertices, hull_AL_test.vertices[0])
            ax.plot(np.array(ALcoor_flat)[vert][:,0], 
                    np.array(ALcoor_flat)[vert][:,1], 
                    75,
                    color='k',
                    lw=3)
        else:
            hull_AL_test = ConvexHull(np.array(ALcoor_flat)[:,[0,2]])
            vert = np.append(hull_AL_test.vertices, hull_AL_test.vertices[0])
            ax.plot(np.array(ALcoor_flat)[vert][:,0], 
                    np.repeat(325, len(vert)),
                    np.array(ALcoor_flat)[vert][:,2], 
                    color='k',
                    lw=3)
        
        for f in j:
            for p in ALcoor_per_n_[f]:
                for b in range(len(p)-1):
                    morph_line = np.vstack((p[b], p[b+1]))
                    plt.plot(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)

        if z == 0:
            ax.view_init(elev=90., azim=-90)
            plt.savefig(os.path.join(dirpath, 'AL_C' + str(i+1) + '_t_1'), dpi=300, bbox_inches='tight', transparent=True)
        else:
            ax.view_init(elev=0., azim=-90)
            plt.savefig(os.path.join(dirpath, 'AL_C' + str(i+1) + '_f_1'), dpi=300, bbox_inches='tight', transparent=True)
            
        plt.close()


#%% MB Fq

PATH = r'./Dros_MB_fq'

fp_MB = [f for f in os.listdir(PATH) if os.path.exists(os.path.join(PATH, f))]
MB_id = copy.deepcopy(fp_MB)
MB_id = [e[:-4] for e in MB_id]
MB_id = np.array(MB_id)
fp_MB = [os.path.join(PATH, f) for f in fp_MB]

MB_id = np.delete(MB_id, [40, 41])
fp_MB = np.delete(fp_MB, [40, 41])

fq_MB = np.empty((len(fp_MB), len(q_full)))

for i,j in enumerate(fp_MB):
    fq_MB[i] = np.load(j)


#%%

MBcoor_per_n_flat = []
length_MB_new = []

for i,j in enumerate(MBcoor_per_n):
    temp = [item for sublist in j for item in sublist]
    if len(temp) > 0:
        MBcoor_per_n_flat.append(temp)
        length_MB_new.append(length_MB[i])

MBcoor_per_n_flat = list(np.delete(np.array(MBcoor_per_n_flat, dtype=object), [40, 41]))
length_MB_new = list(np.delete(np.array(length_MB_new, dtype=object), [40, 41]))

length_branch_flat_MB = [item for sublist in length_MB_new for item in sublist]

branch_length_average_MB = np.mean(length_branch_flat_MB)

rgy_MB, cML = radiusOfGyration(MBcoor_per_n_flat)

rgymean_MB = 2*np.mean(rgy_MB)

i1 = np.argmin(np.abs(q_full - 2*np.pi/rgymean_MB))
i2 = np.argmin(np.abs(q_full - 2*np.pi/branch_length_average_MB))+1

fq_MB_dist = np.zeros((len(fq_MB), len(fq_MB)))

for i in range(len(fq_MB)):
    for j in range(len(fq_MB)):
        fq_MB_dist[i][j] = np.linalg.norm(np.log10(fq_MB[i,i1:i2])-np.log10(fq_MB[j,i1:i2]))

link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_MB_dist), 
                                       method='complete', optimal_ordering=True)

#%%

from dynamicTreeCut import cutreeHybrid

ind_MB = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_MB_dist), minClusterSize=1)['labels']


#%%

silhoutte_MB = []

for k in np.arange(2, 70):
    sil_MB = scipy.cluster.hierarchy.fcluster(link, k, 'maxclust')
    silhoutte_MB.append(sklearn.metrics.silhouette_score(fq_MB_dist, sil_MB, metric="precomputed"))

ind_MB = scipy.cluster.hierarchy.fcluster(link, np.arange(2,70)[np.argmax(silhoutte_MB)], 'maxclust')


#%%

ind_MB_idx = []

for i in np.unique(ind_MB):
    ind_MB_idx.append(np.where(ind_MB == i)[0])

ind_MB_idx_rgy = []

for i,j in enumerate(ind_MB_idx):
    rgyb, cmb = radiusOfGyration(list(np.array(MBcoor_per_n_flat, dtype=object)[j]))
    ind_MB_idx_rgy.append(np.mean(rgyb))

ind_MB_idx_sort = list(np.array(ind_MB_idx, dtype=object)[np.argsort(ind_MB_idx_rgy)[::-1]])

#%%

length_MB_ = []

for i in length_MB_new:
    if len(i) > 0:
        length_MB_.append(i)

cmap = cm.get_cmap('viridis', len(ind_MB_idx_sort))

for i,j in enumerate(ind_MB_idx_sort):
    bl = list(np.array(length_MB_, dtype=object)[j])
    blf = [item for sublist in bl for item in sublist]
    
    rgyb, cmb = radiusOfGyration(list(np.array(MBcoor_per_n_flat, dtype=object)[j]))
    
    fig = plt.figure(figsize=(6,4))
    if i == len(ind_MB_idx_sort)-2:
        plt.plot(q_full, fq_MB[j].T, color=(0.798216, 0.280197, 0.469538, 1.0))
    elif i == len(ind_MB_idx_sort)-1:
        plt.plot(q_full, fq_MB[j].T, color=(1.0, 0.25, 0.0, 1.0))
    else:
        plt.plot(q_full, fq_MB[j].T, color=cmap(i))
    plt.vlines(2*np.pi/np.mean(blf), 1e-5, 10, color='k', ls='dashed')
    plt.vlines(2*np.pi/(2*np.mean(rgyb)), 1e-5, 10, color='k', ls='dotted')
    
    line1 = 2e-4*np.power(q_full, -16/7)
    line2 = 7e-10*np.power(q_full, -4/1)
    line3 = 3e-4*np.power(q_full, -1/0.395)
    line4 = 1/100*np.power(q_full, -1)
    line5 = 1e-3*np.power(q_full, -2/1)
    
    # if i == 0:
    #     plt.plot(q_full[130:157], line1[130:157], lw=1.5, color='tab:blue')
    #     plt.plot(q_full[110:147], line2[110:147], lw=1.5, color='tab:red')
    #     plt.plot(q_full[180:207], line3[180:207], lw=1.5, color='tab:purple')
    #     plt.plot(q_full[220:270], line4[220:270], lw=1.5, color='k')
    #     plt.plot(q_full[150:200], line5[200:250], lw=1.5, color='tab:green')
        
    #     plt.text(0.03, 8e-1, r'$\nu = \dfrac{7}{16}$', fontsize=13, color='tab:blue')
    #     plt.text(0.007, 1e-2, r'$\nu = \dfrac{1}{4}$', fontsize=13, color='tab:red')
    #     plt.text(0.1, 2e-1, r'$\nu = 0.395$', fontsize=13, color='tab:purple')
    #     plt.text(0.4, 1e-2, r'$\nu = 1$', fontsize=13, color='k')
    #     plt.text(0.03, 1e-2, r'$\nu = \dfrac{1}{2}$', fontsize=13, color='tab:green')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-2, 10)
    plt.ylim(1e-4, 3)
    plt.ylabel('$F(q)$', fontsize=18)
    plt.xlabel('$q$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title("$C^{MB}_{" + str(i+1) + "}$", fontsize=15, pad=10)
    # plt.savefig('./Drosfigures/Fq_MB_c' + str(i+1) + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()

#%% MB plot cluster individual neurons

MBcoor_per_n_ = []

for i in MBcoor_per_n:
    if len(i) > 0:
        MBcoor_per_n_.append(i)

MBcoor_per_n_ = list(np.delete(np.array(MBcoor_per_n_, dtype=object), [40, 41]))

cmap = cm.get_cmap('viridis', len(ind_MB_idx_sort))

for i,j in enumerate(ind_MB_idx_sort):
    dirpath = './Drosfigures/MB_C' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
    for f in j:
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        ax.set_xlim(450, 600)
        ax.set_ylim(100, 250)
        ax.axis('off')
        
        for p in MBcoor_per_n_[f]:
            for b in range(len(p)-1):
                morph_line = np.vstack((p[b], p[b+1]))
                if i == len(ind_MB_idx_sort)-2:
                    plt.plot(morph_line[:,0], morph_line[:,2], color=(0.798216, 0.280197, 0.469538, 1.0))
                elif i == len(ind_MB_idx_sort)-1:
                    plt.plot(morph_line[:,0], morph_line[:,2], color=(1.0, 0.25, 0.0, 1.0))
                else:
                    plt.plot(morph_line[:,0], morph_line[:,2], color=cmap(i))
        # plt.savefig(os.path.join(dirpath, MB_id[f]), dpi=300, bbox_inches='tight')
        plt.close()

#%% MB plot cluster all neurons

MBcoor_per_n_ = []

for i in MBcoor_per_n:
    if len(i) > 0:
        MBcoor_per_n_.append(i)

cmap = cm.get_cmap('viridis', len(ind_MB_idx_sort))

for i,j in enumerate(ind_MB_idx_sort):
    dirpath = './Drosfigures/MB_C' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
    fig = plt.figure(figsize=(24, 16))
    ax = plt.axes(projection='3d')
    ax.set_xlim(500, 650)
    ax.set_ylim(400, 250)
    ax.set_zlim(125, 275)
    ax.axis('off')
    
    for f in j:
        for p in MBcoor_per_n_[f]:
            for b in range(len(p)-1):
                morph_line = np.vstack((p[b], p[b+1]))
                plt.plot(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)
    plt.savefig(os.path.join(dirpath, 'MB_C' + str(i+1)), dpi=300, bbox_inches='tight')
    plt.close()


#%% MB plot all clusters

MBcoor_per_n_ = []

for i in MBcoor_per_n:
    if len(i) > 0:
        MBcoor_per_n_.append(i)

cmap = cm.get_cmap('viridis', len(ind_MB_idx_sort))

fig = plt.figure(figsize=(24, 16))
ax = plt.axes(projection='3d')
ax.set_xlim(500, 650)
ax.set_ylim(400, 250)
ax.set_zlim(125, 275)
ax.axis('off')

for i,j in enumerate(ind_MB_idx_sort):
    for f in j:
        for p in MBcoor_per_n_[f]:
            for b in range(len(p)-1):
                morph_line = np.vstack((p[b], p[b+1]))
                plt.plot(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)
                
# plt.savefig('./Drosfigures/morph_Dros_MB_full_1.png', dpi=300, bbox_inches='tight')
plt.close()

#%% MB plot cluster all neurons in 2D + boundary

from scipy.spatial import ConvexHull

MBcoor_per_n_ = []

for i in MBcoor_per_n:
    if len(i) > 0:
        MBcoor_per_n_.append(i)

MBcoor_flat = [item for sublist in MBcoor for item in sublist]

MBcoor_flat = np.array(MBcoor_flat)[np.where(np.array(MBcoor_flat)[:,1] < 250)]
MBcoor_flat = np.array(MBcoor_flat)[np.where(np.array(MBcoor_flat)[:,1] > 190)]
# MBcoor_flat = np.array(MBcoor_flat)[np.where(np.array(MBcoor_flat)[:,2] > 125)]

hull_MB = ConvexHull(np.array(MBcoor_flat))

tri_MB = []
for i in range(len(hull_MB.simplices)):
    tt = []
    for j in range(len(hull_MB.simplices[i])):
        tt.append(np.where(hull_MB.vertices == hull_MB.simplices[i][j])[0][0])
    tri_MB.append(tuple(tt))

cmap = cm.get_cmap('viridis', len(ind_MB_idx_sort))

for i,j in enumerate(ind_MB_idx_sort):
    dirpath = './Drosfigures/MB_C' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    for z in range(2):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        ax.set_box_aspect((1,1,1))
        ax.set_xlim(450, 600)
        ax.set_ylim(300, 150)
        ax.set_zlim(75, 225)
        ax.dist = 7
        ax.axis('off')
        
        if z == 0:
            hull_MB_test = ConvexHull(np.array(MBcoor_flat)[:,:2])
            vert = np.append(hull_MB_test.vertices, hull_MB_test.vertices[0])
            ax.plot(np.array(MBcoor_flat)[vert][:,0], 
                    np.array(MBcoor_flat)[vert][:,1], 
                    150,
                    color='k',
                    lw=3)
        else:
            hull_MB_test = ConvexHull(np.array(MBcoor_flat)[:,[0,2]])
            vert = np.append(hull_MB_test.vertices, hull_MB_test.vertices[0])
            ax.plot(np.array(MBcoor_flat)[vert][:,0], 
                    np.repeat(225, len(vert)),
                    np.array(MBcoor_flat)[vert][:,2], 
                    color='k',
                    lw=3)
        
        for f in j:
            for p in MBcoor_per_n_[f]:
                for b in range(len(p)-1):
                    morph_line = np.vstack((p[b], p[b+1]))
                    plt.plot(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)

        if z == 0:
            ax.view_init(elev=90., azim=-90)
            plt.savefig(os.path.join(dirpath, 'MB_C' + str(i+1) + '_t_1'), dpi=300, bbox_inches='tight', transparent=True)
        else:
            ax.view_init(elev=0., azim=-90)
            plt.savefig(os.path.join(dirpath, 'MB_C' + str(i+1) + '_f_1'), dpi=300, bbox_inches='tight', transparent=True)
            
        plt.close()
    
    
#%% LH Fq

PATH = r'./Dros_LH_fq'

fp_LH = [f for f in os.listdir(PATH) if os.path.exists(os.path.join(PATH, f))]
LH_id = copy.deepcopy(fp_LH)
LH_id = [e[:-4] for e in LH_id]
LH_id = np.array(LH_id)
fp_LH = [os.path.join(PATH, f) for f in fp_LH]

fq_LH = np.empty((len(fp_LH), len(q_full)))

for i,j in enumerate(fp_LH):
    fq_LH[i] = np.load(j)

    
#%%

length_branch_flat_LH = [item for sublist in length_LH for item in sublist]

branch_length_average_LH = np.mean(length_branch_flat_LH)

LHcoor_per_n_flat = []

for i in LHcoor_per_n:
    temp = [item for sublist in i for item in sublist]
    if len(temp) > 0:
        LHcoor_per_n_flat.append(temp)

rgy_LH, cML = radiusOfGyration(LHcoor_per_n_flat)

rgymean_LH = np.mean(rgy_LH)

i1 = np.argmin(np.abs(q_full - 2*np.pi/rgymean_LH))
i2 = np.argmin(np.abs(q_full - 2*np.pi/branch_length_average_LH))+1

fq_LH_dist = np.zeros((len(fq_LH), len(fq_LH)))

for i in range(len(fq_LH)):
    for j in range(len(fq_LH)):
        fq_LH_dist[i][j] = np.linalg.norm(np.log10(fq_LH[i,i1:i2])-np.log10(fq_LH[j,i1:i2]))

link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_LH_dist), 
                                       method='centroid', optimal_ordering=True)

#%%

from dynamicTreeCut import cutreeHybrid

ind_LH = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_LH_dist), minClusterSize=1)['labels']

#%%

silhoutte_LH = []

for k in np.arange(2, 70):
    sil_LH = scipy.cluster.hierarchy.fcluster(link, k, 'maxclust')
    silhoutte_LH.append(sklearn.metrics.silhouette_score(fq_LH_dist, sil_LH, metric="precomputed"))

ind_LH = scipy.cluster.hierarchy.fcluster(link, np.arange(2, 70)[np.argmax(silhoutte_LH)], 'maxclust')

#%%

ind_LH_idx = []

for i in np.unique(ind_LH):
    ind_LH_idx.append(np.where(ind_LH == i)[0])

ind_LH_idx_rgy = []

for i,j in enumerate(ind_LH_idx):
    rgyb, cmb = radiusOfGyration(list(np.array(LHcoor_per_n_flat, dtype=object)[j]))
    ind_LH_idx_rgy.append(np.mean(rgyb))

ind_LH_idx_sort = list(np.array(ind_LH_idx, dtype=object)[np.argsort(ind_LH_idx_rgy)[::-1]])

#%%

length_LH_ = []

for i in length_LH:
    if len(i) > 0:
        length_LH_.append(i)

cmap = cm.get_cmap('viridis', len(ind_LH_idx_sort))

for i,j in enumerate(ind_LH_idx_sort):
    bl = list(np.array(length_LH_, dtype=object)[j])
    blf = [item for sublist in bl for item in sublist]
    
    rgyb, cmb = radiusOfGyration(list(np.array(LHcoor_per_n_flat, dtype=object)[j]))
    
    fig = plt.figure(figsize=(6,4))
    if i == len(ind_LH_idx_sort)-2:
        plt.plot(q_full, fq_LH[j].T, color=(0.798216, 0.280197, 0.469538, 1.0))
    elif i == len(ind_LH_idx_sort)-1:
        plt.plot(q_full, fq_LH[j].T, color=(1.0, 0.25, 0.0, 1.0))
    else:
        plt.plot(q_full, fq_LH[j].T, color=cmap(i))
    plt.vlines(2*np.pi/np.mean(blf), 1e-5, 10, color='k', ls='dashed')
    plt.vlines(2*np.pi/(2*np.mean(rgyb)), 1e-5, 10, color='k', ls='dotted')
    
    line1 = 2e-4*np.power(q_full, -16/7)
    line2 = 7e-10*np.power(q_full, -4/1)
    line3 = 3e-4*np.power(q_full, -1/0.395)
    line4 = 1/100*np.power(q_full, -1)
    line5 = 1e-3*np.power(q_full, -2/1)
    
    # if i == 0:
    #     plt.plot(q_full[130:157], line1[130:157], lw=1.5, color='tab:blue')
    #     plt.plot(q_full[110:147], line2[110:147], lw=1.5, color='tab:red')
    #     plt.plot(q_full[180:207], line3[180:207], lw=1.5, color='tab:purple')
    #     plt.plot(q_full[220:270], line4[220:270], lw=1.5, color='k')
    #     plt.plot(q_full[150:200], line5[200:250], lw=1.5, color='tab:green')
        
    #     plt.text(0.03, 8e-1, r'$\nu = \dfrac{7}{16}$', fontsize=13, color='tab:blue')
    #     plt.text(0.007, 1e-2, r'$\nu = \dfrac{1}{4}$', fontsize=13, color='tab:red')
    #     plt.text(0.1, 2e-1, r'$\nu = 0.395$', fontsize=13, color='tab:purple')
    #     plt.text(0.4, 1e-2, r'$\nu = 1$', fontsize=13, color='k')
    #     plt.text(0.03, 1e-2, r'$\nu = \dfrac{1}{2}$', fontsize=13, color='tab:green')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-2, 10)
    plt.ylim(1e-4, 3)
    plt.ylabel('$F(q)$', fontsize=18)
    plt.xlabel('$q$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title("$C^{LH}_{" + str(i+1) + "}$", fontsize=15, pad=10)
    # plt.savefig('./Drosfigures/Fq_LH_c' + str(i+1) + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()

#%% LH plot cluster individual neurons

LHcoor_per_n_ = []

for i in LHcoor_per_n:
    if len(i) > 0:
        LHcoor_per_n_.append(i)

cmap = cm.get_cmap('viridis', len(ind_LH_idx_sort))

for i,j in enumerate(ind_LH_idx_sort):
    dirpath = './Drosfigures/LH_C' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
    for f in j:
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        ax.set_xlim(375, 525)
        ax.set_ylim(100, 250)
        ax.axis('off')
        
        for p in LHcoor_per_n_[f]:
            for b in range(len(p)-1):
                morph_line = np.vstack((p[b], p[b+1]))
                if i == len(ind_LH_idx_sort)-2:
                    plt.plot(morph_line[:,0], morph_line[:,2], color=(0.798216, 0.280197, 0.469538, 1.0))
                elif i == len(ind_LH_idx_sort)-1:
                    plt.plot(morph_line[:,0], morph_line[:,2], color=(1.0, 0.25, 0.0, 1.0))
                else:
                    plt.plot(morph_line[:,0], morph_line[:,2], color=cmap(i))
        # plt.savefig(os.path.join(dirpath, LH_id[f]), dpi=300, bbox_inches='tight')
        plt.close()

#%% LH plot cluster all neurons

LHcoor_per_n_ = []

for i in LHcoor_per_n:
    if len(i) > 0:
        LHcoor_per_n_.append(i)

cmap = cm.get_cmap('viridis', len(ind_LH_idx_sort))

for i,j in enumerate(ind_LH_idx_sort):
    dirpath = './Drosfigures/LH_C' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
    fig = plt.figure(figsize=(24, 16))
    ax = plt.axes(projection='3d')
    ax.set_xlim(400, 550)
    ax.set_ylim(400, 250)
    ax.set_zlim(125, 275)
    ax.axis('off')
    
    for f in j:
        for p in LHcoor_per_n_[f]:
            for b in range(len(p)-1):
                morph_line = np.vstack((p[b], p[b+1]))
                plt.plot(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)
    plt.savefig(os.path.join(dirpath, 'LH_C' + str(i+1)), dpi=300, bbox_inches='tight')
    plt.close()


#%% LH plot all clusters

LHcoor_per_n_ = []

for i in LHcoor_per_n:
    if len(i) > 0:
        LHcoor_per_n_.append(i)

cmap = cm.get_cmap('viridis', len(ind_LH_idx_sort))

fig = plt.figure(figsize=(24, 16))
ax = plt.axes(projection='3d')
ax.set_xlim(400, 550)
ax.set_ylim(400, 250)
ax.set_zlim(125, 275)
ax.axis('off')

for i,j in enumerate(ind_LH_idx_sort):
    for f in j:
        for p in LHcoor_per_n_[f]:
            for b in range(len(p)-1):
                morph_line = np.vstack((p[b], p[b+1]))
                plt.plot(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)
                
plt.savefig('./Drosfigures/morph_Dros_LH_full_1.png', dpi=300, bbox_inches='tight')
plt.close()

#%% LH plot cluster all neurons in 2D + boundary

from scipy.spatial import ConvexHull

LHcoor_per_n_ = []

for i in LHcoor_per_n:
    if len(i) > 0:
        LHcoor_per_n_.append(i)

LHcoor_flat = [item for sublist in LHcoor for item in sublist]

LHcoor_flat = np.array(LHcoor_flat)[np.where(np.array(LHcoor_flat)[:,1] < 255)]
LHcoor_flat = np.array(LHcoor_flat)[np.where(np.array(LHcoor_flat)[:,1] > 185)]
LHcoor_flat = np.array(LHcoor_flat)[np.where(np.array(LHcoor_flat)[:,2] > 125)]

hull_LH = ConvexHull(np.array(LHcoor_flat))

tri_LH = []
for i in range(len(hull_LH.simplices)):
    tt = []
    for j in range(len(hull_LH.simplices[i])):
        tt.append(np.where(hull_LH.vertices == hull_LH.simplices[i][j])[0][0])
    tri_LH.append(tuple(tt))

cmap = cm.get_cmap('viridis', len(ind_LH_idx_sort))

for i,j in enumerate(ind_LH_idx_sort):
    dirpath = './Drosfigures/LH_C' + str(i+1)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    for z in range(2):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        ax.set_box_aspect((1,1,1))
        ax.set_xlim(350, 500)
        ax.set_ylim(300, 150)
        ax.set_zlim(75, 225)
        ax.dist = 7
        ax.axis('off')
        
        if z == 0:
            hull_LH_test = ConvexHull(np.array(LHcoor_flat)[:,:2])
            vert = np.append(hull_LH_test.vertices, hull_LH_test.vertices[0])
            ax.plot(np.array(LHcoor_flat)[vert][:,0], 
                    np.array(LHcoor_flat)[vert][:,1], 
                    150,
                    color='k',
                    lw=3)
        else:
            hull_LH_test = ConvexHull(np.array(LHcoor_flat)[:,[0,2]])
            vert = np.append(hull_LH_test.vertices, hull_LH_test.vertices[0])
            ax.plot(np.array(LHcoor_flat)[vert][:,0], 
                    np.repeat(225, len(vert)),
                    np.array(LHcoor_flat)[vert][:,2], 
                    color='k',
                    lw=3)
        
        for f in j:
            for p in LHcoor_per_n_[f]:
                for b in range(len(p)-1):
                    morph_line = np.vstack((p[b], p[b+1]))
                    plt.plot(morph_line[:,0], morph_line[:,1], morph_line[:,2], color=cmap(i), lw=0.5)

        if z == 0:
            ax.view_init(elev=90., azim=-90)
            plt.savefig(os.path.join(dirpath, 'LH_C' + str(i+1) + '_t_1'), dpi=300, bbox_inches='tight', transparent=True)
        else:
            ax.view_init(elev=0., azim=-90)
            plt.savefig(os.path.join(dirpath, 'LH_C' + str(i+1) + '_f_1'), dpi=300, bbox_inches='tight', transparent=True)
            
        plt.close()


#%% metric testing full

branch_length_average_full = 2.789401932780958

rgymean_full = 2*72.07108928773508

i1 = np.argmin(np.abs(q_full - 2*np.pi/rgymean_full))
i2 = np.argmin(np.abs(q_full - 2*np.pi/branch_length_average_full))+1

ind_full_idx_sort_all = []

for n in range(4):
    fq_full_dist = np.zeros((len(fp), len(fp)))
    
    for i in range(len(fp)):
        for j in range(len(fp)):
            exp_data = np.zeros((i2-i1, 2))
            exp_data[:,0] = np.log10(q_full)[i1:i2]
            exp_data[:,1] = np.log10(fq_full[i,i1:i2])
            num_data = np.zeros((i2-i1, 2))
            num_data[:,0] = np.log10(q_full)[i1:i2]
            num_data[:,1] = np.log10(fq_full[j,i1:i2])
            if n == 0:
                # L2
                fq_full_dist[i][j] = np.linalg.norm((np.log10(fq_full[i,i1:i2])-np.log10(fq_full[j,i1:i2])))
            elif n == 1:
                # L1
                fq_full_dist[i][j] = np.linalg.norm((np.log10(fq_full[i,i1:i2])-np.log10(fq_full[j,i1:i2])), ord=1)
            elif n == 2:
                # cosine
                fq_full_dist[i][j] = scipy.spatial.distance.cosine(np.log10(fq_full[i,i1:i2]), np.log10(fq_full[j,i1:i2]))
            elif n == 3:
                # Frechet
                fq_full_dist[i][j] = sm.frechet_dist(exp_data[:,1], num_data[:,1])
    
    link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_full_dist), 
                                           method='complete', optimal_ordering=True)

    ind_full = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_full_dist), minClusterSize=1)['labels']
    
    ind_full_idx = []
    
    for i in np.unique(ind_full):
        ind_full_idx.append(list(np.where(ind_full == i)[0]))
    
    ind_full_idx_rgy = []
    
    for i,j in enumerate(ind_full_idx):
        rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]))
        ind_full_idx_rgy.append(np.mean(rgyb))
    
    ind_full_idx_sort = list(np.array(ind_full_idx, dtype=object)[np.argsort(ind_full_idx_rgy)[::-1]])
    
    ind_full_idx_sort_all.append(ind_full_idx_sort)
    
#%%

import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost

ind_full_idx_sort_all_flat = []

full_ind_chg = np.empty((len(ind_full_idx_sort_all), len(fp)))

for i,j in enumerate(ind_full_idx_sort_all):
    ind_full_idx_sort_all_flat.append([item for sublist in j for item in sublist])
    idx = 0
    c = 1
    for k in j:
        full_ind_chg[i][idx:idx+len(k)] = c
        idx = idx+len(k)
        c += 1

for i in range(len(ind_full_idx_sort_all_flat)-1):
    temp = []
    for j in ind_full_idx_sort_all_flat[0]:
        temp.append(np.where(j == ind_full_idx_sort_all_flat[i+1])[0][0])
    full_ind_chg[i+1] = full_ind_chg[i+1][temp]
    
full_ind_chg[1][full_ind_chg[1] == 6] = 7
full_ind_chg[1][full_ind_chg[1] == 5] = 6
full_ind_chg[2][full_ind_chg[2] == 4] = 7
full_ind_chg[2][full_ind_chg[2] == 4] = 7
full_ind_chg[3][full_ind_chg[3] == 7] = 9
full_ind_chg[3][full_ind_chg[3] == 8] = 7
full_ind_chg[3][full_ind_chg[3] == 9] = 8
full_ind_chg[3][full_ind_chg[3] == 3] = 9
full_ind_chg[3][full_ind_chg[3] == 4] = 3
full_ind_chg[3][full_ind_chg[3] == 9] = 4

    
fig = plt.figure(figsize=(10,1))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(full_ind_chg, cmap='tab20', aspect='auto')
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
# plt.savefig('./Drosfigures/full_metric_test_1.svg', dpi=300, bbox_inches='tight')
plt.show()

#%% metric testing AL

i1 = np.argmin(np.abs(q_full - 2*np.pi/rgymean_AL))
i2 = np.argmin(np.abs(q_full - 2*np.pi/branch_length_average_AL))+1

ind_AL_idx_sort_all = []

for n in range(4):
    fq_AL_dist = np.zeros((len(fq_AL), len(fq_AL)))
    
    for i in range(len(fq_AL)):
        for j in range(len(fq_AL)):
            exp_data = np.zeros((i2-i1, 2))
            exp_data[:,0] = np.log10(q_full)[i1:i2]
            exp_data[:,1] = np.log10(fq_AL[i,i1:i2])
            num_data = np.zeros((i2-i1, 2))
            num_data[:,0] = np.log10(q_full)[i1:i2]
            num_data[:,1] = np.log10(fq_AL[j,i1:i2])
            if n == 0:
                # L2
                fq_AL_dist[i][j] = np.linalg.norm((np.log10(fq_AL[i,i1:i2])-np.log10(fq_AL[j,i1:i2])))
            elif n == 1:
                # L1
                fq_AL_dist[i][j] = np.linalg.norm((np.log10(fq_AL[i,i1:i2])-np.log10(fq_AL[j,i1:i2])), ord=1)
            elif n == 2:
                # cosine
                fq_AL_dist[i][j] = scipy.spatial.distance.cosine(np.log10(fq_AL[i,i1:i2]), np.log10(fq_AL[j,i1:i2]))
            elif n == 3:
                # Frechet
                fq_AL_dist[i][j] = sm.frechet_dist(exp_data[:,1], num_data[:,1])
    
    link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_AL_dist), 
                                           method='complete', optimal_ordering=True)

    ind_AL = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_AL_dist), minClusterSize=1)['labels']
    
    ind_AL_idx = []
    
    for i in np.unique(ind_AL):
        ind_AL_idx.append(list(np.where(ind_AL == i)[0]))
    
    ind_AL_idx_rgy = []
    
    for i,j in enumerate(ind_AL_idx):
        rgyb, cmb = radiusOfGyration(list(np.array(ALcoor_per_n_flat, dtype=object)[j]))
        ind_AL_idx_rgy.append(np.mean(rgyb))
    
    ind_AL_idx_sort = list(np.array(ind_AL_idx, dtype=object)[np.argsort(ind_AL_idx_rgy)[::-1]])
    
    ind_AL_idx_sort_all.append(ind_AL_idx_sort)
    
    
#%%

import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost

ind_AL_idx_sort_all_flat = []

AL_ind_chg = np.empty((len(ind_AL_idx_sort_all), len(fq_AL)))

for i,j in enumerate(ind_AL_idx_sort_all):
    ind_AL_idx_sort_all_flat.append([item for sublist in j for item in sublist])
    idx = 0
    c = 1
    for k in j:
        AL_ind_chg[i][idx:idx+len(k)] = c
        idx = idx+len(k)
        c += 1

for i in range(len(ind_AL_idx_sort_all_flat)-1):
    temp = []
    for j in ind_AL_idx_sort_all_flat[0]:
        temp.append(np.where(j == ind_AL_idx_sort_all_flat[i+1])[0][0])
    AL_ind_chg[i+1] = AL_ind_chg[i+1][temp]
    
AL_ind_chg[1][AL_ind_chg[1] == 4] = 7
AL_ind_chg[1][AL_ind_chg[1] == 3] = 4
AL_ind_chg[1][AL_ind_chg[1] == 7] = 3

AL_ind_chg[3][AL_ind_chg[3] == 7] = 12
AL_ind_chg[3][AL_ind_chg[3] == 11] = 7
AL_ind_chg[3][AL_ind_chg[3] == 2] = 13
AL_ind_chg[3][AL_ind_chg[3] == 12] = 2
AL_ind_chg[3][AL_ind_chg[3] == 6] = 14
AL_ind_chg[3][AL_ind_chg[3] == 10] = 6
AL_ind_chg[3][AL_ind_chg[3] == 14] = 10
AL_ind_chg[3][AL_ind_chg[3] == 13] = 11
AL_ind_chg[3][AL_ind_chg[3] == 5] = 15
AL_ind_chg[3][AL_ind_chg[3] == 8] = 5
AL_ind_chg[3][AL_ind_chg[3] == 15] = 8

fig = plt.figure(figsize=(10,1))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(AL_ind_chg, cmap='tab20', aspect='auto')
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
# plt.savefig('./Drosfigures/AL_metric_test_1.svg', dpi=300, bbox_inches='tight')
plt.show()

#%% metric testing MB

i1 = np.argmin(np.abs(q_full - 2*np.pi/rgymean_MB))
i2 = np.argmin(np.abs(q_full - 2*np.pi/branch_length_average_MB))+1

ind_MB_idx_sort_all = []

for n in range(4):
    fq_MB_dist = np.zeros((len(fq_MB), len(fq_MB)))
    
    for i in range(len(fq_MB)):
        for j in range(len(fq_MB)):
            exp_data = np.zeros((i2-i1, 2))
            exp_data[:,0] = np.log10(q_full)[i1:i2]
            exp_data[:,1] = np.log10(fq_MB[i,i1:i2])
            num_data = np.zeros((i2-i1, 2))
            num_data[:,0] = np.log10(q_full)[i1:i2]
            num_data[:,1] = np.log10(fq_MB[j,i1:i2])
            if n == 0:
                # L2
                fq_MB_dist[i][j] = np.linalg.norm((np.log10(fq_MB[i,i1:i2])-np.log10(fq_MB[j,i1:i2])))
            elif n == 1:
                # L1
                fq_MB_dist[i][j] = np.linalg.norm((np.log10(fq_MB[i,i1:i2])-np.log10(fq_MB[j,i1:i2])), ord=1)
            elif n == 2:
                # cosine
                fq_MB_dist[i][j] = scipy.spatial.distance.cosine(np.log10(fq_MB[i,i1:i2]), np.log10(fq_MB[j,i1:i2]))
            elif n == 3:
                # Frechet
                fq_MB_dist[i][j] = sm.frechet_dist(exp_data[:,1], num_data[:,1])
    
    link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_MB_dist), 
                                           method='complete', optimal_ordering=True)

    ind_MB = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_MB_dist), minClusterSize=1)['labels']
    
    ind_MB_idx = []
    
    for i in np.unique(ind_MB):
        ind_MB_idx.append(list(np.where(ind_MB == i)[0]))
    
    ind_MB_idx_rgy = []
    
    for i,j in enumerate(ind_MB_idx):
        rgyb, cmb = radiusOfGyration(list(np.array(MBcoor_per_n_flat, dtype=object)[j]))
        ind_MB_idx_rgy.append(np.mean(rgyb))
    
    ind_MB_idx_sort = list(np.array(ind_MB_idx, dtype=object)[np.argsort(ind_MB_idx_rgy)[::-1]])
    
    ind_MB_idx_sort_all.append(ind_MB_idx_sort)
    
    
#%%

import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost

ind_MB_idx_sort_all_flat = []

MB_ind_chg = np.empty((len(ind_MB_idx_sort_all), len(fq_MB)))

for i,j in enumerate(ind_MB_idx_sort_all):
    ind_MB_idx_sort_all_flat.append([item for sublist in j for item in sublist])
    idx = 0
    c = 1
    for k in j:
        MB_ind_chg[i][idx:idx+len(k)] = c
        idx = idx+len(k)
        c += 1

for i in range(len(ind_MB_idx_sort_all_flat)-1):
    temp = []
    for j in ind_MB_idx_sort_all_flat[0]:
        temp.append(np.where(j == ind_MB_idx_sort_all_flat[i+1])[0][0])
    MB_ind_chg[i+1] = MB_ind_chg[i+1][temp]


MB_ind_chg[1][MB_ind_chg[1] == 5] = 6
MB_ind_chg[1][MB_ind_chg[1] == 4] = 5
MB_ind_chg[2][MB_ind_chg[2] == 3] = 6
MB_ind_chg[2][MB_ind_chg[2] == 2] = 3

MB_ind_chg[3][MB_ind_chg[3] == 1] = 11
MB_ind_chg[3][MB_ind_chg[3] == 2] = 1
MB_ind_chg[3][MB_ind_chg[3] == 3] = 2
MB_ind_chg[3][MB_ind_chg[3] == 5] = 10
MB_ind_chg[3][MB_ind_chg[3] == 6] = 5
MB_ind_chg[3][MB_ind_chg[3] == 7] = 6

MB_ind_chg[3][MB_ind_chg[3] == 11] = 3
MB_ind_chg[3][MB_ind_chg[3] == 10] = 7

fig = plt.figure(figsize=(10,1))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(MB_ind_chg, cmap='tab20', aspect='auto')
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
# plt.savefig('./Drosfigures/MB_metric_test_1.svg', dpi=300, bbox_inches='tight')
plt.show()


#%% metric testing LH

i1 = np.argmin(np.abs(q_full - 2*np.pi/rgymean_LH))
i2 = np.argmin(np.abs(q_full - 2*np.pi/branch_length_average_LH))+1

ind_LH_idx_sort_all = []

for n in range(4):
    fq_LH_dist = np.zeros((len(fq_LH), len(fq_LH)))
    
    for i in range(len(fq_LH)):
        for j in range(len(fq_LH)):
            exp_data = np.zeros((i2-i1, 2))
            exp_data[:,0] = np.log10(q_full)[i1:i2]
            exp_data[:,1] = np.log10(fq_LH[i,i1:i2])
            num_data = np.zeros((i2-i1, 2))
            num_data[:,0] = np.log10(q_full)[i1:i2]
            num_data[:,1] = np.log10(fq_LH[j,i1:i2])
            if n == 0:
                # L2
                fq_LH_dist[i][j] = np.linalg.norm((np.log10(fq_LH[i,i1:i2])-np.log10(fq_LH[j,i1:i2])))
            elif n == 1:
                # L1
                fq_LH_dist[i][j] = np.linalg.norm((np.log10(fq_LH[i,i1:i2])-np.log10(fq_LH[j,i1:i2])), ord=1)
            elif n == 2:
                # cosine
                fq_LH_dist[i][j] = scipy.spatial.distance.cosine(np.log10(fq_LH[i,i1:i2]), np.log10(fq_LH[j,i1:i2]))
            elif n == 3:
                # Frechet
                fq_LH_dist[i][j] = sm.frechet_dist(exp_data[:,1], num_data[:,1])
    
    link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_LH_dist), 
                                           method='complete', optimal_ordering=True)

    ind_LH = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_LH_dist), minClusterSize=1)['labels']
    
    ind_LH_idx = []
    
    for i in np.unique(ind_LH):
        ind_LH_idx.append(list(np.where(ind_LH == i)[0]))
    
    ind_LH_idx_rgy = []
    
    for i,j in enumerate(ind_LH_idx):
        rgyb, cmb = radiusOfGyration(list(np.array(LHcoor_per_n_flat, dtype=object)[j]))
        ind_LH_idx_rgy.append(np.mean(rgyb))
    
    ind_LH_idx_sort = list(np.array(ind_LH_idx, dtype=object)[np.argsort(ind_LH_idx_rgy)[::-1]])
    
    ind_LH_idx_sort_all.append(ind_LH_idx_sort)
    
    
#%%

import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost

ind_LH_idx_sort_all_flat = []

LH_ind_chg = np.empty((len(ind_LH_idx_sort_all), len(fq_LH)))

for i,j in enumerate(ind_LH_idx_sort_all):
    ind_LH_idx_sort_all_flat.append([item for sublist in j for item in sublist])
    idx = 0
    c = 1
    for k in j:
        LH_ind_chg[i][idx:idx+len(k)] = c
        idx = idx+len(k)
        c += 1

for i in range(len(ind_LH_idx_sort_all_flat)-1):
    temp = []
    for j in ind_LH_idx_sort_all_flat[0]:
        temp.append(np.where(j == ind_LH_idx_sort_all_flat[i+1])[0][0])
    LH_ind_chg[i+1] = LH_ind_chg[i+1][temp]
    
LH_ind_chg[1][LH_ind_chg[1] == 5] = 6
LH_ind_chg[1][LH_ind_chg[1] == 4] = 5
LH_ind_chg[2][LH_ind_chg[2] == 2] = 4
LH_ind_chg[2][LH_ind_chg[2] == 3] = 6
LH_ind_chg[3][LH_ind_chg[3] == 1] = 7
LH_ind_chg[3][LH_ind_chg[3] == 2] = 1
LH_ind_chg[3][LH_ind_chg[3] == 7] = 2

fig = plt.figure(figsize=(10,1))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(LH_ind_chg, cmap='tab20', aspect='auto')
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
# plt.savefig('./Drosfigures/LH_metric_test_1.svg', dpi=300, bbox_inches='tight')
plt.show()

