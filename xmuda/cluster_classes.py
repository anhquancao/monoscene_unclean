import pickle
import os
import numpy as np
import glob
from collections import defaultdict
from sklearn.cluster import KMeans
from xmuda.data.NYU.params import NYU_class_names
from tqdm import tqdm
from xmuda.data.semantic_kitti.params import kitti_class_names

dataset = "NYU"
# dataset = "kitti"
dir = "/gpfsscratch/rech/kvd/uyl37fq/temp/features/{}/*.pkl".format(dataset)
class_data = defaultdict(list)
class_labels = defaultdict(list)
# all_data = []
# all_labels = []
for file_path in tqdm(glob.glob(dir)):    
    out = pickle.load( open( file_path, "rb" ) )
    features = out['features']
    labels = out['labels']
    # all_data.append(features)
    # all_labels.append(labels)
    classes, cnts = np.unique(labels, return_counts=True)
    for c in classes:
        mask = (labels == c)
        class_data[c].append(features[:, mask])
        class_labels[c].append(labels[mask])
if dataset == "NYU":
    class_names = NYU_class_names
    n_vox_per_class = 5000
if dataset == "kitti":
    class_names = kitti_class_names
    n_vox_per_class = 5000
all_data = []
all_labels = []
for k in class_data.keys():
    class_data[k] = np.concatenate(class_data[k], axis=1)
    class_labels[k] = np.concatenate(class_labels[k], axis=0)
    print(class_names[k], len(class_labels[k]))
    if n_vox_per_class < len(class_labels[k]):
        idx = np.random.choice(len(class_labels[k]), n_vox_per_class, replace=False)
        all_data.append(class_data[k][:, idx])
        all_labels.append(class_labels[k][idx])
    else:
        all_data.append(class_data[k])
        all_labels.append(class_labels[k])
    
all_data = np.concatenate(all_data, axis=1).T
all_labels = np.concatenate(all_labels, axis=0)

print(all_data)
# print(all_data.shape, all_labels.shape)

kmeans = KMeans(n_clusters=3,  verbose=False).fit(all_data)
kmeans_labels = kmeans.labels_

pairs = np.concatenate([all_labels[:, None], kmeans_labels[:, None]], axis=1)
print(pairs.shape)
rows, cnts = np.unique(pairs, axis=0, return_counts=True)
assigned_cluster = defaultdict(int)
assigned_cluster_cnt = defaultdict(int)
for i in range(len(rows)):    
    if assigned_cluster_cnt[rows[i][0]] < cnts[i]:
        assigned_cluster_cnt[rows[i][0]] = cnts[i]
        assigned_cluster[rows[i][0]] = rows[i][1]

clusters = defaultdict(list)
for i in assigned_cluster.keys():
    print(class_names[i], assigned_cluster[i], assigned_cluster_cnt[i])
    clusters[assigned_cluster[i]].append(class_names[i])

print(clusters)