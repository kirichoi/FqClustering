# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 13:55:26 2021

@author: user
"""

import os
import numpy as np
import scipy.cluster
from dynamicTreeCut import cutreeHybrid
import similaritymeasures as sm
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter
import copy

os.chdir(os.path.dirname(__file__))

fp = [f for f in os.listdir('./CE_fq_new') if os.path.isfile(os.path.join('./CE_fq_new', f))]
neuron_id = copy.deepcopy(fp)
neuron_id = [e[:-4] for e in neuron_id]
neuron_id = np.array(neuron_id)
fp = [os.path.join('./CE_fq_new', f) for f in fp]

fq = np.empty((len(fp), 601))

for i,j in enumerate(fp):
    fq[i] = np.load(j)

q = np.logspace(-3, 3, 601)

fq_dist = np.zeros((len(fp), len(fp)))

#%% Read reconstructions

import os
import neuroml.loaders as loaders
import numpy as np

PATH = r'./CElegansNeuroML-SNAPSHOT_030213/CElegans/generatedNeuroML2'

fp = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]
fp = [f for f in fp if "Acetylcholine" not in f]
fp = [f for f in fp if "CElegans" not in f]
fp = [f for f in fp if "Dopamine" not in f]
fp = [f for f in fp if "FMRFamide" not in f]
fp = [f for f in fp if "GABA" not in f]
fp = [f for f in fp if "Glutamate" not in f]
fp = [f for f in fp if "LeakConductance" not in f]
fp = [f for f in fp if "Octapamine" not in f]
fp = [f for f in fp if "Serotonin" not in f]
fp = [f for f in fp if "README" not in f]
fp = [os.path.join(PATH, f) for f in fp]

neuron_id = []
morph_coor = []
morph_dia = []
neuron_type = []
sensory = []
inter = []
motor = []
polymodal = []
other = []
morph_parent = []
morph_id = []
branchNum = np.empty(len(fp))
somaP = []
branchTrk = []
branch_coor = []
length_branch = []
d2d = []

for f in range(len(fp)):
    morph_neu_id = []
    morph_neu_parent = []
    morph_neu_prox = []
    morph_neu_dist = []
    morph_neu_dia = []
    
    doc = loaders.NeuroMLLoader.load(fp[f])
    neuron_id.append(doc.cells[0].id)
    neuron_type.append(doc.cells[0].notes.strip())
    if doc.cells[0].notes.strip() == "SensoryNeuron":
        sensory.append(f)
    elif doc.cells[0].notes.strip() == "Interneuron":
        inter.append(f)
    elif doc.cells[0].notes.strip() == "Motor Neuron":
        motor.append(f)
    elif doc.cells[0].notes.strip() == "PolymodalNeuron":
        polymodal.append(f)
    else:
        other.append(f)
    sgmts = doc.cells[0].morphology
    for s in range(sgmts.num_segments):
        sgmt = doc.cells[0].morphology.segments[s]
        morph_neu_id.append(sgmt.id+1)
        if sgmt.parent != None:
            morph_neu_parent.append(sgmt.parent.segments+1)
        else:
            morph_neu_parent.append(-1)
            somaP.append(s)
        if sgmt.proximal != None:
            morph_neu_prox.append([sgmt.proximal.x, 
                                   sgmt.proximal.y, 
                                   sgmt.proximal.z])
        else:
            morph_neu_prox.append([])
        if sgmt.distal != None:
            morph_neu_dist.append([sgmt.distal.x, 
                                   sgmt.distal.y, 
                                   sgmt.distal.z])
            morph_neu_dia.append(sgmt.distal.diameter)
        else:
            morph_neu_dist.append([])
    
    morph_id.append(morph_neu_id)
    morph_parent.append(morph_neu_parent)
    morph_coor.append(morph_neu_dist)
    morph_dia.append(morph_neu_dia)
    ctr = Counter(morph_neu_parent)
    ctrVal = list(ctr.values())
    ctrKey = list(ctr.keys())
    branchNum[f] = sum(i > 1 for i in ctrVal)
    branchInd = np.array(ctrKey)[np.where(np.array(ctrVal) > 1)[0]]
    
    neu_branchTrk = []
    branch_coor_temp1 = []
    length_branch_temp = []
    d2d_temp = []
    
    list_end = np.setdiff1d(morph_id[f], morph_parent[f])
    
    bPoint = np.append(branchInd, list_end)
    bPoint = np.unique(bPoint)
    
    for bp in range(len(bPoint)):
        if bPoint[bp] != somaP[f]:
            neu_branchTrk_temp = []
            branch_coor_temp2 = []
            dist = 0
            
            neu_branchTrk_temp.append(bPoint[bp])
            branch_coor_temp2.append(morph_coor[f][morph_id[f].index(bPoint[bp])][:3])
            parentTrck = bPoint[bp]
            parentTrck = morph_parent[f][morph_id[f].index(parentTrck)]
            if parentTrck != -1:
                neu_branchTrk_temp.append(parentTrck)
                rhs = branch_coor_temp2[-1][:3]
                lhs = morph_coor[f][morph_id[f].index(parentTrck)][:3]
                branch_coor_temp2.append(lhs)
                dist += np.linalg.norm(np.subtract(rhs, lhs))
                d2d_temp.append(np.linalg.norm(np.subtract(rhs, lhs)))
                
            while (parentTrck not in branchInd) and (parentTrck != -1):
                parentTrck = morph_parent[f][morph_id[f].index(parentTrck)]
                if parentTrck != -1:
                    neu_branchTrk_temp.append(parentTrck)
                    rhs = branch_coor_temp2[-1][:3]
                    lhs = morph_coor[f][morph_id[f].index(parentTrck)][:3]
                    branch_coor_temp2.append(lhs)
                    dist += np.linalg.norm(np.subtract(rhs, lhs))
                    d2d_temp.append(np.linalg.norm(np.subtract(rhs, lhs)))
                    
            if len(neu_branchTrk_temp) > 1:
                neu_branchTrk.append(neu_branchTrk_temp)
                branch_coor_temp1.append(branch_coor_temp2)
                length_branch_temp.append(dist)
    branchTrk.append(neu_branchTrk)
    branch_coor.append(branch_coor_temp1)
    length_branch.append(length_branch_temp)
    d2d.append(d2d_temp)

length_branch_flat = [item for sublist in length_branch for item in sublist]

branch_length_average = np.mean(length_branch_flat)

d2df = [item for sublist in d2d for item in sublist]

print('Sensory: ' + str(len(sensory))) 
print('Inter: ' + str(len(inter))) 
print('Motor: ' + str(len(motor))) 
print('Polymodal: ' + str(len(polymodal))) 
print('Other: ' + str(len(other))) 


def radiusOfGyration(morph_coor, morph_dia):
    cML = np.empty((len(morph_coor), 3))
    rGy = np.empty(len(morph_coor))
    for i in range(len(morph_coor)):
        cML[i] = np.average(np.array(morph_coor[i]), axis=0, weights=morph_dia[i])
        rList = scipy.spatial.distance.cdist(np.array(morph_coor[i]), 
                                              np.array([cML[i]])).flatten()
        rGy[i] = np.sqrt(np.average(np.square(rList)))
    
    return (rGy, cML)


#%% Distance calculation

branch_length_average = 6.919827651963916
rgymean = 149.63549418586476

i1 = np.argmin(np.abs(q - 2*np.pi/rgymean))
i2 = np.argmin(np.abs(q - 2*np.pi/branch_length_average))+1

fig = plt.figure(figsize=(6,4))
plt.plot(q, fq.T)
plt.vlines(2*np.pi/rgymean, 1e-4, 2, color='k', ls='dotted')
plt.vlines(2*np.pi/branch_length_average, 1e-4, 2, color='k', ls='dashed')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-3, 2)
plt.ylabel('$F(q)$', fontsize=13)
plt.xlabel('$q$', fontsize=13)
plt.show()

for i in range(len(fp)):
    for j in range(len(fp)):
        fq_dist[i][j] = np.linalg.norm(np.log10(fq[i,i1:i2])-np.log10(fq[j,i1:i2]))

link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_dist), 
                                       method='complete', optimal_ordering=True)

#%% Hybrid tree cutting

ind3 = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_dist), minClusterSize=1)['labels']

ind3_idx = []

for i in np.unique(ind3):
    ind3_idx.append(np.where(ind3 == i)[0])

ind3_idx_rgy = []

for i,j in enumerate(ind3_idx):
    rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]), 
                                 list(np.array(morph_dia, dtype=object)[j]))
    ind3_idx_rgy.append(np.mean(rgyb))

ind3_idx_sort = list(np.array(ind3_idx, dtype=object)[np.argsort(ind3_idx_rgy)[::-1]])

#%% F(q) curves per cluster

cmap = cm.get_cmap('viridis', len(ind3_idx_sort))

for i,j in enumerate(ind3_idx_sort):
    bl = list(np.array(length_branch, dtype=object)[j])
    blf = [item for sublist in bl for item in sublist]
    
    rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]), 
                                 list(np.array(morph_dia, dtype=object)[j]))
    
    d2db = list(np.array(d2d, dtype=object)[j])
    d2db = [item for sublist in d2db for item in sublist]
    
    fig = plt.figure(figsize=(6,4))
    if i == len(ind3_idx_sort)-1:
        plt.plot(q, fq[j].T, color=(0.798216, 0.280197, 0.469538, 1.0))
    else:
        plt.plot(q, fq[j].T, color=cmap(i))
    plt.vlines(2*np.pi/np.mean(blf), 1e-3, 10, color='k', ls='dashed')
    plt.vlines(2*np.pi/(48), 1e-3, 10, color='k', ls='dotted')
    
    line1 = 2e-4*np.power(q, -16/7)
    line2 = 7e-9*np.power(q, -4/1)
    line3 = 3e-4*np.power(q, -1/0.388)
    line4 = 1/80*np.power(q, -1)
    line5 = 1e-3*np.power(q, -2/1)
    
    if i == 0:
        plt.plot(q[130:157], line1[130:157], lw=1.5, color='tab:blue')
        plt.plot(q[110:147], line2[110:147], lw=1.5, color='tab:red')
        plt.plot(q[180:207], line3[180:207], lw=1.5, color='tab:purple')
        plt.plot(q[220:270], line4[220:270], lw=1.5, color='k')
        plt.plot(q[150:200], line5[200:250], lw=1.5, color='tab:green')
        
        plt.text(0.03, 8e-1, r'$\nu = \dfrac{7}{16}$', fontsize=13, color='tab:blue')
        plt.text(0.007, 5e-2, r'$\nu = \dfrac{1}{4}$', fontsize=13, color='tab:red')
        plt.text(0.1, 2e-1, r'$\nu = 0.388$', fontsize=13, color='tab:purple')
        plt.text(0.4, 5e-2, r'$\nu = 1$', fontsize=13, color='k')
        plt.text(0.03, 5e-2, r'$\nu = \dfrac{1}{2}$', fontsize=13, color='tab:green')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(2*np.pi/np.mean(25*rgymean), 1)
    if i != 1:
        plt.ylim(2e-2, 3)
    else:
        plt.ylim(2e-3, 3)
    plt.ylabel('$F(q)$', fontsize=18)
    plt.xlabel('$q$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("$C^{CE}_{" + str(i+1) + "}$", fontsize=15, pad=10)
    # plt.savefig('./CEfigures/Fq_C_' + str(i+1) + '_3.pdf', dpi=300, bbox_inches='tight')
    plt.show()


#%% Neuron reconstruction diagram per cluster

for i in range(len(ind3_idx_sort)):
    fig = plt.figure(figsize=(24, 1))
    ax = fig.add_subplot(111)
    ax.axis('off')
    for f in ind3_idx_sort[i]:
        tararr = np.array(morph_coor[f])
        somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
        for p in range(len(morph_parent[f])):
            if morph_parent[f][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_coor[f][morph_id[f].index(morph_parent[f][p])], morph_coor[f][p]))
                if i == 4:
                    plt.plot(morph_line[:,1], morph_line[:,0], color=(0.798216, 0.280197, 0.469538, 1.0))
                else:
                    plt.plot(morph_line[:,1], morph_line[:,0], color=cmap(i))
    ax.set_xlim(-360, 470)
    ax.set_ylim(-40, 35)
    # plt.savefig('./CEfigures/morph_CE_c' + str(i+1) + '_3.png', dpi=300, bbox_inches='tight')
    plt.show()

#%% Metric testing

branch_length_average = 6.919827651963916
rgymean = 149.63549418586476

i1 = np.argmin(np.abs(q - 2*np.pi/rgymean))
i2 = np.argmin(np.abs(q - 2*np.pi/branch_length_average))+1

ind_idx_sort_all = []

for n in range(4):
    fq_dist = np.zeros((len(fp), len(fp)))
    
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
                fq_dist[i][j] = np.linalg.norm((np.log10(fq[i,i1:i2])-np.log10(fq[j,i1:i2])))
            elif n == 1:
                # L1
                fq_dist[i][j] = np.linalg.norm((np.log10(fq[i,i1:i2])-np.log10(fq[j,i1:i2])), ord=1)
            elif n == 2:
                # cosine
                fq_dist[i][j] = scipy.spatial.distance.cosine(np.log10(fq[i,i1:i2]), np.log10(fq[j,i1:i2]))
            elif n == 3:
                # Frechet
                fq_dist[i][j] = sm.frechet_dist(exp_data[:,1], num_data[:,1])
    
    link = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_dist), 
                                           method='complete', optimal_ordering=True)

    ind = cutreeHybrid(link, scipy.spatial.distance.squareform(fq_dist), minClusterSize=1)['labels']
    
    ind_idx = []
    
    for i in np.unique(ind):
        ind_idx.append(list(np.where(ind == i)[0]))
    
    ind_idx_rgy = []
    
    for i,j in enumerate(ind_idx):
        rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[j]), 
                                     list(np.array(morph_dia, dtype=object)[j]))
        ind_idx_rgy.append(np.mean(rgyb))
    
    ind_idx_sort = list(np.array(ind_idx, dtype=object)[np.argsort(ind_idx_rgy)[::-1]])
    
    ind_idx_sort_all.append(ind_idx_sort)

#%% Metric testing diagram

import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost

ind_idx_sort_all_flat = []

ind_chg = np.empty((len(ind_idx_sort_all), len(fq)))

for i,j in enumerate(ind_idx_sort_all):
    ind_idx_sort_all_flat.append([item for sublist in j for item in sublist])
    idx = 0
    c = 1
    for k in j:
        ind_chg[i][idx:idx+len(k)] = c
        idx = idx+len(k)
        c += 1

for i in range(len(ind_idx_sort_all_flat)-1):
    temp = []
    for j in ind_idx_sort_all_flat[0]:
        temp.append(np.where(j == ind_idx_sort_all_flat[i+1])[0][0])
    ind_chg[i+1] = ind_chg[i+1][temp]

ind_chg[2][ind_chg[2] == 3] = 5
ind_chg[2][ind_chg[2] == 4] = 6
ind_chg[2][ind_chg[2] == 2] = 4

ind_chg[3][ind_chg[3] == 6] = 7
ind_chg[3][ind_chg[3] == 4] = 6
ind_chg[3][ind_chg[3] == 5] = 4
ind_chg[3][ind_chg[3] == 7] = 5
    
fig = plt.figure(figsize=(10,1))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
plt.imshow(ind_chg, cmap='tab20', aspect='auto')
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
# plt.savefig('./CEfigures/CE_metric_test_1.svg', dpi=300, bbox_inches='tight')
plt.show()


