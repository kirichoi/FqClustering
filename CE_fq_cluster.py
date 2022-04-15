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

neuron_type_fixed = []

for i in neuron_type:
    if i == 'SensoryNeuron':
        neuron_type_fixed.append('Sensory')
    elif i == 'Interneuron':
        neuron_type_fixed.append('Inter')
    elif i == 'Motor Neuron':
        neuron_type_fixed.append('Motor')
    elif i == 'PolymodalNeuron':
        neuron_type_fixed.append('Polymodal')
    elif i == 'NeurUnkFunc':
        neuron_type_fixed.append('Unknown')

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


#%% Total contour length per cluster

tl = []
for i in ind3_idx_sort:
    ttl = []
    for j in i:
        ttl.append(np.sum(length_branch[j]))
    tl.append(ttl)
    
tl_sum = []
for i in tl:
    tl_sum.append(np.sum(i))

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
        
        plt.text(0.03, 8e-1, r'$\mathcal{D} = 2.286$', fontsize=13, color='tab:blue')
        plt.text(0.007, 5e-2, r'$\mathcal{D} = 4$', fontsize=13, color='tab:red')
        plt.text(0.1, 2e-1, r'$\mathcal{D} = 2.53$', fontsize=13, color='tab:purple')
        plt.text(0.4, 5e-2, r'$\mathcal{D} = 1$', fontsize=13, color='k')
        plt.text(0.03, 5e-2, r'$\mathcal{D} = 2$', fontsize=13, color='tab:green')
    
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


#%% Functional pie chart with different sizes

cl = np.array(['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple'])

test_dist3 = []

for i in range(len(ind3_idx_sort)):
    test_dist3_temp = []
    c = Counter(np.array(neuron_type_fixed)[ind3_idx_sort[i]])
    test_dist3_temp.append(c['Sensory'])
    test_dist3_temp.append(c['Inter'])
    test_dist3_temp.append(c['Motor'])
    test_dist3_temp.append(c['Polymodal'])
    test_dist3_temp.append(c['Unknown'])
    test_dist3.append(test_dist3_temp)

category_names = ['$C_{1}^{CE}$', '$C_{2}^{CE}$', '$C_{3}^{CE}$', '$C_{4}^{CE}$', 
                  '$C_{5}^{CE}$']

fig, ax = plt.subplots(1,5,figsize=(14,3))
for j in range(5):
    tidx = np.nonzero(test_dist3[j])
    fracs = np.array(test_dist3[j])[tidx]
    total = sum(fracs)
    _, _, autotexts = ax[j].pie(fracs, radius=2*np.sqrt(tl_sum[j]/np.max(tl_sum)), autopct=lambda p: '{:.0f}'.format(p * total / 100), colors=cl[tidx])
    # ax[j].axis('equal')
    ax[j].set_title(category_names[j], fontsize=18)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_size(15)
# plt.savefig('./CEfigures/pie_CE_ds_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%% Subclusters in cluster 4 and 5

fq_dist1 = fq_dist[ind3_idx_sort[3]]
fq_dist1 = fq_dist1[:,ind3_idx_sort[3]]

fq_dist2 = fq_dist[ind3_idx_sort[4]]
fq_dist2 = fq_dist2[:,ind3_idx_sort[4]]

link1 = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_dist1), 
                                       method='complete', optimal_ordering=True)

link2 = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(fq_dist2), 
                                       method='complete', optimal_ordering=True)

ind1 = cutreeHybrid(link1, scipy.spatial.distance.squareform(fq_dist1), minClusterSize=1)['labels']

ind1_idx = []

for i in np.unique(ind1):
    ind1_idx.append(np.where(ind1 == i)[0])

ind1_idx_rgy = []

for i,j in enumerate(ind1_idx):
    rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[ind3_idx_sort[3][j]]), 
                                 list(np.array(morph_dia, dtype=object)[ind3_idx_sort[3][j]]))
    ind1_idx_rgy.append(np.mean(rgyb))

ind1_idx_sort = list(np.array(ind1_idx, dtype=object)[np.argsort(ind1_idx_rgy)[::-1]])

tl1 = []
for i in ind1_idx_sort:
    ttl = []
    for j in i:
        ttl.append(np.sum(length_branch[j]))
    tl1.append(ttl)
    
tl1_sum = []
for i in tl1:
    tl1_sum.append(np.sum(i))

ind2 = cutreeHybrid(link2, scipy.spatial.distance.squareform(fq_dist2), minClusterSize=1)['labels']

ind2_idx = []

for i in np.unique(ind2):
    ind2_idx.append(np.where(ind2 == i)[0])

ind2_idx_rgy = []

for i,j in enumerate(ind2_idx):
    rgyb, cmb = radiusOfGyration(list(np.array(morph_coor, dtype=object)[ind3_idx_sort[4][j]]), 
                                 list(np.array(morph_dia, dtype=object)[ind3_idx_sort[4][j]]))
    ind2_idx_rgy.append(np.mean(rgyb))

ind2_idx_sort = list(np.array(ind2_idx, dtype=object)[np.argsort(ind2_idx_rgy)[::-1]])

tl2 = []
for i in ind2_idx_sort:
    ttl = []
    for j in i:
        ttl.append(np.sum(length_branch[j]))
    tl2.append(ttl)
    
tl2_sum = []
for i in tl2:
    tl2_sum.append(np.sum(i))

#%% Functional pie chart with different sizes for cluster 4 and 5

test_dist1 = []

for i in range(len(ind1_idx_sort)):
    test_dist1_temp = []
    c = Counter(np.array(neuron_type_fixed)[ind3_idx_sort[3][ind1_idx_sort[i]]])
    test_dist1_temp.append(c['Sensory'])
    test_dist1_temp.append(c['Inter'])
    test_dist1_temp.append(c['Motor'])
    test_dist1_temp.append(c['Polymodal'])
    test_dist1_temp.append(c['Unknown'])
    test_dist1.append(test_dist1_temp)

category_names = ['$C_{4;1}^{CE}$', '$C_{4;2}^{CE}$', '$C_{4;3}^{CE}$', '$C_{4;4}^{CE}$', 
                  '$C_{4;5}^{CE}$', '$C_{4;6}^{CE}$', '$C_{4;7}^{CE}$', '$C_{4;8}^{CE}$', 
                  '$C_{4;9}^{CE}$', '$C_{4;10}^{CE}$', '$C_{4;11}^{CE}$', '$C_{4;12}^{CE}$', 
                  '$C_{4;13}^{CE}$', '$C_{4;14}^{CE}$', '$C_{4;15}^{CE}$']

fig, ax = plt.subplots(3,5,figsize=(14,9))
for i in range(3):
    for j in range(5):
        tidx = np.nonzero(test_dist1[i*5+j])
        fracs = np.array(test_dist1[i*5+j])[tidx]
        total = sum(fracs)
        _, _, autotexts = ax[i][j].pie(fracs, radius=1.5*np.sqrt(tl1_sum[j]/np.max(tl1_sum)), autopct=lambda p: '{:.0f}'.format(p * total / 100), colors=cl[tidx])
        # ax[i][j].axis('equal')
        ax[i][j].set_title(category_names[i*5+j], fontsize=18)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_size(15)
# plt.savefig('./CEfigures/pie_subcluster_CE4_ds_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

test_dist2 = []

for i in range(len(ind2_idx_sort)):
    test_dist2_temp = []
    c = Counter(np.array(neuron_type_fixed)[ind3_idx_sort[4][ind2_idx_sort[i]]])
    test_dist2_temp.append(c['Sensory'])
    test_dist2_temp.append(c['Inter'])
    test_dist2_temp.append(c['Motor'])
    test_dist2_temp.append(c['Polymodal'])
    test_dist2_temp.append(c['Unknown'])
    test_dist2.append(test_dist2_temp)

category_names = ['$C_{5;1}^{CE}$', '$C_{5;2}^{CE}$', '$C_{5;3}^{CE}$', '$C_{5;4}^{CE}$', 
                  '$C_{5;5}^{CE}$', '$C_{5;6}^{CE}$', '$C_{5;7}^{CE}$', '$C_{5;8}^{CE}$', 
                  '$C_{5;9}^{CE}$', '$C_{5;10}^{CE}$']

fig, ax = plt.subplots(2,5,figsize=(14,6))
for i in range(2):
    for j in range(5):
        tidx = np.nonzero(test_dist2[i*5+j])
        fracs = np.array(test_dist2[i*5+j])[tidx]
        total = sum(fracs)
        _, _, autotexts = ax[i][j].pie(fracs, radius=1.5*np.sqrt(tl2_sum[j]/np.max(tl2_sum)), autopct=lambda p: '{:.0f}'.format(p * total / 100), colors=cl[tidx])
        # ax[i][j].axis('equal')
        ax[i][j].set_title(category_names[i*5+j], fontsize=18)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_size(15)
# plt.savefig('./CEfigures/pie_subcluster_CE5_ds_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%% Neuron reconstruction diagram per cluster with function

fig = plt.figure(figsize=(24, 1))
ax = fig.add_subplot(111)
ax.axis('off')
for f in ind3_idx_sort[3]:
    if neuron_type_fixed[f] == 'Sensory':
        co = 'tab:red'
    elif neuron_type_fixed[f] == 'Inter':
        co = 'tab:green'
    elif neuron_type_fixed[f] == 'Motor':
        co = 'tab:blue'
    elif neuron_type_fixed[f] == 'Polymodal':
        co = 'tab:orange'
    elif neuron_type_fixed[f] == 'Unknown':
        co = 'tab:purple'
    tararr = np.array(morph_coor[f])
    somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
    for p in range(len(morph_parent[f])):
        if morph_parent[f][p] < 0:
            pass
        else:
            morph_line = np.vstack((morph_coor[f][morph_id[f].index(morph_parent[f][p])], morph_coor[f][p]))
            plt.plot(morph_line[:,1], morph_line[:,0], color=co)
# ax.set_xlim(-360, -200)
# ax.set_ylim(-30, 25)
ax.set_xlim(-360, 470)
ax.set_ylim(-40, 35)
# ax.set_zlim(-300, 300)
# plt.savefig('./CEfigures/morph_CE_c4_f_t_1.png', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(24, 1))
ax = fig.add_subplot(111)
ax.axis('off')
for f in ind3_idx_sort[4]:
    if neuron_type_fixed[f] == 'Sensory':
        co = 'tab:red'
    elif neuron_type_fixed[f] == 'Inter':
        co = 'tab:green'
    elif neuron_type_fixed[f] == 'Motor':
        co = 'tab:blue'
    elif neuron_type_fixed[f] == 'Polymodal':
        co = 'tab:orange'
    elif neuron_type_fixed[f] == 'Unknown':
        co = 'tab:purple'
    tararr = np.array(morph_coor[f])
    somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
    for p in range(len(morph_parent[f])):
        if morph_parent[f][p] < 0:
            pass
        else:
            morph_line = np.vstack((morph_coor[f][morph_id[f].index(morph_parent[f][p])], morph_coor[f][p]))
            plt.plot(morph_line[:,1], morph_line[:,0], color=co)
# ax.set_xlim(-360, -200)
# ax.set_ylim(-30, 25)
ax.set_xlim(-360, 470)
ax.set_ylim(-40, 35)
# ax.set_zlim(-300, 300)
# plt.savefig('./CEfigures/morph_CE_c5_f_1.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Neuron reconstruction diagram per function

for i in range(len(np.unique(neuron_type_fixed))):
    idx = np.where(np.array(neuron_type_fixed) == np.unique(neuron_type_fixed)[i])[0]
    
    if np.unique(neuron_type_fixed)[i] == 'Sensory':
        co = 'tab:red'
    elif np.unique(neuron_type_fixed)[i] == 'Inter':
        co = 'tab:green'
    elif np.unique(neuron_type_fixed)[i] == 'Motor':
        co = 'tab:blue'
    elif np.unique(neuron_type_fixed)[i] == 'Polymodal':
        co = 'tab:orange'
    elif np.unique(neuron_type_fixed)[i] == 'Unknown':
        co = 'tab:purple'
    
    fig = plt.figure(figsize=(24, 1))
    ax = fig.add_subplot(111)
    ax.axis('off')
    for f in idx:
        tararr = np.array(morph_coor[f])
        somaIdx = np.where(np.array(morph_parent[f]) < 0)[0]
        for p in range(len(morph_parent[f])):
            if morph_parent[f][p] < 0:
                pass
            else:
                morph_line = np.vstack((morph_coor[f][morph_id[f].index(morph_parent[f][p])], morph_coor[f][p]))
                plt.plot(morph_line[:,1], morph_line[:,0], color=co)
    ax.set_xlim(-360, 470)
    ax.set_ylim(-40, 35)
    # ax.set_zlim(-300, 300)
    
    # plt.savefig('./CEfigures/morph_CE_f_' + str(np.unique(neuron_type_fixed)[i]) + '_1.png', dpi=300, bbox_inches='tight')
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


