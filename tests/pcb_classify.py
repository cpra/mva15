
'''
Classify PCB images as mainboard or non-mainboard based on the geometrical features
size and aspect ratio using a random forest.
~ Christopher Pramerdorfer, 2015
'''

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn.cross_validation

import sys
import os.path
import argparse


def savefig(fig, filename, legend=None):
    l = matplotlib.pyplot.gca().get_legend()
    if l:
        l.get_frame().set_alpha(0.5)

    fig.savefig(filename.replace(' ', '_'), format='pdf', bbox_inches='tight')


# parse and check args

parser = argparse.ArgumentParser(description='Compute dataset statistics.')
parser.add_argument('--api', type=str, required=True, help='Directory that contains the DSLR dataset python API file.')
parser.add_argument('--db', type=str, required=True, help='Root directory of the DSLR dataset.')
parser.add_argument('--mbids', type=str, required=True, help='Path to a file that contains PCB IDs of the mainboard class.')
args = parser.parse_args()

if not os.path.isdir(args.api):
    sys.exit('"{}" is not a directory'.format(args.api))

if not os.path.isdir(args.db):
    sys.exit('"{}" is not a directory'.format(args.db))

if not os.path.isfile(args.mbids):
    sys.exit('"{}" is not a file'.format(args.mbids))

try:
    sys.path.insert(0, args.api)
    from pcb_dataset import PCBDataset
except:
    sys.exit('Failed to import DSLR dataset API .. wrong directory?')

# setup LaTeX style figure export

s = 10
matplotlib.rc('font', family='serif', serif=['Times New Roman'], size=s)
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(2.8, 2))
matplotlib.rc('legend', fontsize=s, frameon=True, fancybox=True, loc='best')
matplotlib.rc('xtick', labelsize=s)
matplotlib.rc('ytick', labelsize=s)

# load IDs of mainboards

mbids = np.loadtxt(args.mbids, dtype=np.int32)
print('Loaded IDs of {} mainboards'.format(mbids.size))

# obtain data

print('Loading data ...')

db = PCBDataset(args.db)
data = []

for pcb in db.pcbs():
    print(' {}'.format(pcb))

    id = pcb.id()

    for r in pcb.recordings():  # use size of smallest to minimize errors due to perspective
        sz = np.sum(pcb.mask(rec=r) > 0) / (87.4**2)

        cnt, _ = cv2.findContours(pcb.mask(rec=r), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(cnt) != 1:
            print(' Warning: Mask of recording {} contains {} components!'.format(r, len(cnt)))

        rect = cv2.minAreaRect(cnt[0])
        asp = max(rect[1][0], rect[1][1]) / min(rect[1][0], rect[1][1])

        data.append((id, r, sz, asp))

# create feature vectors

features_mb = np.array([(d[2], d[3]) for d in data if d[0] in mbids], dtype=np.float32)
features_others = np.array([(d[2], d[3]) for d in data if d[0] not in mbids], dtype=np.float32)

# visualize

print('Computing joint distribution visualization')

fig = plt.figure()
plt.scatter(features_mb[:, 0], features_mb[:, 1], s=12, alpha=0.2, linewidths=0, marker='^', c='r')
plt.scatter(features_others[:, 0], features_others[:, 1], s=8, alpha=0.2, linewidths=0, marker='o', c='b')
plt.ylim((0.9, np.max(np.hstack((features_mb[:, 1], features_others[:, 1])))*1.02))
plt.xlabel('Size in cm$^2$')
plt.ylabel('Aspect Ratio')
savefig(fig, 'scatter-pcb-size-shape.pdf')

# test classification performance using 10-fold cross validation

print('Classifying ...')

X = np.vstack((features_mb, features_others))
y = np.hstack((np.zeros((features_mb.shape[0]), dtype=np.int16), np.ones((features_others.shape[0]), dtype=np.int16)))

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, max_depth=3)
cv = sklearn.cross_validation.StratifiedKFold(y, n_folds=10, random_state=0)
result = sklearn.cross_validation.cross_val_score(rf, X, y, cv=cv)  # 10-fold stratified

print('Done, crossval scores: {}'.format(result))
print(' avg: {}, sd: {}'.format(np.mean(result), np.std(result)))

plt.show()
