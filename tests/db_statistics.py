
'''
Compute, print, and visualize different statistics of the DSLR dataset.
~ Christopher Pramerdorfer, 2015
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import os.path
import argparse
import sys


def histc(lst):
    lst = np.array(lst)
    hist = np.vstack([np.unique(lst), np.unique(lst)])
    for i in range(hist.shape[1]):
        hist[1, i] = np.sum(lst == hist[0, i])

    return hist


def savefig(fig, filename, legend=None):
    l = matplotlib.pyplot.gca().get_legend()
    if l:
        l.get_frame().set_alpha(0.5)

    fig.savefig(filename.replace(' ', '_'), format='pdf', bbox_inches='tight')


# parse and check args

parser = argparse.ArgumentParser(description='Compute dataset statistics.')
parser.add_argument('--api', type=str, required=True, help='Directory that contains the DSLR dataset python API file.')
parser.add_argument('--db', type=str, required=True, help='Root directory of the DSLR dataset.')
args = parser.parse_args()

if not os.path.isdir(args.api):
    sys.exit('"{}" is not a directory'.format(args.api))

if not os.path.isdir(args.db):
    sys.exit('"{}" is not a directory'.format(args.db))

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

# gather information

num_pcbs = 0  # number of different PCBs
num_images = {}  # total number of images for each PCB ID
pcb_sizes = {}  # PCB size in cm^2 for each PCB ID
num_ics = {}  # number of ICs for each PCB ID
num_labeled_ics = {}  # number of ICs with label information for each PCB ID
ic_size = []  # size of each IC in cm^2
ic_aspect = []  # aspect ratio of each IC

db = PCBDataset(args.db)

for pcb in db.pcbs():
    print(pcb)

    id = pcb.id()
    num_pcbs += 1
    num_images[id] = len(pcb.recordings())

    ics = pcb.ics(rec=1)
    num_ics[id] = len(ics)
    num_labeled_ics[id] = len([ic for ic in ics if ic.text])

    for ic in ics:
        ic_size.append(ic.size_cm2())
        ic_aspect.append(ic.aspect())

    pcbsz = 0
    for r in pcb.recordings():  # use size of smallest to minimize errors due to perspective
        sz = np.sum(pcb.mask(rec=r) > 0) / (87.4**2)
        if pcbsz == 0 or sz < pcbsz:
            pcbsz = sz
    pcb_sizes[id] = pcbsz

# count statistics

print('')
print('{} unique PCBs'.format(num_pcbs))
print('{} PCB images'.format(sum([num_images[k] for k in num_images])))
print('{} unique ICs'.format(sum([num_ics[k] for k in num_ics])))
print('{} unique ICs with labels'.format(sum([num_labeled_ics[k] for k in num_labeled_ics])))
print('{} ICs including multiple views'.format(sum([num_ics[k] * num_images[k] for k in num_ics])))
print('{} ICs with labels including multiple views'.format(sum([num_labeled_ics[k] * num_images[k] for k in num_labeled_ics])))

# pcb size statistics

pcbsz = np.array([pcb_sizes[k] for k in pcb_sizes])
print('')
print('PCB sizes in cm^2: min: {:.1f}, median: {:.1f}, max: {:.1f}'.format(pcbsz.min(), np.median(pcbsz), pcbsz.max()))

# ic count statistics

hist = histc([num_ics[k] for k in num_ics])
print('')
for i in np.arange(hist.shape[1]):
    print('{} PCBs ({}%) have at most {} ICs)'.format(np.sum(hist[1, :i+1]), 100.0*np.sum(hist[1, :i+1])/np.sum(hist[1, :]), hist[0][i]))

# ic size statistics

print('')
print('IC size percentiles in cm^2:')
for perc in (0, 25, 50, 75, 95, 100):
    print(' p{} : {:.3f}'.format(perc, np.percentile(np.array(ic_size), perc)))

# histogram of IC counts per PCB

fig = plt.figure()
plt.bar(hist[0, :], hist[1, :], linewidth=0)
plt.xlabel('Number of ICs')
plt.ylabel('Number of PCBs')
savefig(fig, 'hist-num-ics.pdf')

# size/aspect scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(ic_size, ic_aspect, s=3, alpha=0.2, linewidths=0)
plt.xscale('log')
plt.xlim((0, max(ic_size)*1.25))
plt.ylim((0.9, max(ic_aspect)*1.02))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel('Size in cm$^2$ (log Scale)')
plt.ylabel('Aspect Ratio')
savefig(fig, 'scatter-ic-size-shape.pdf')

plt.show()
