
"""
Script to check if my implementation of dbscan produces same reulsts as
the scikit-learn implementation
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dbscan import dbscan

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)


print('Running my implementation...')
my_labels = dbscan(eps=0.3, min_points=10)
my_labels.predict(X.T)

print('Running scikit-learn implementation...')
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
skl_labels = db.labels_

# Scikit learn starts cluster labeling at 0. My implementation starsts 
# numbering at 1, so increment the sklearn cluster numbers by 1.

for i in range(0, len(skl_labels)):
    if not skl_labels[i] == -1:
        skl_labels[i] += 1

num_disagree = 0

# Go through each label and make sure they match 
for i in range(0, len(skl_labels)):
    if not skl_labels[i] == my_labels.predict(X.T)[i]:
        print('Scikit learn:', skl_labels[i], 'mine:', my_labels.predict(X.T)[i])
        num_disagree += 1

if num_disagree == 0:
    print('PASS!')
else:
    print('FAIL -', num_disagree, 'labels don\'t match.')
