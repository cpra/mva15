
'''
Transfer IC labels between different images of the same PCB via keypoint-based homography estimation.
~ Christopher Pramerdorfer, Computer Vision Lab, Vienna University of Technology
'''

import cv2
import numpy as np

import sys
import os.path
import argparse
import copy


def filter_matches(kp1, kp2, matches, ratio=0.8):
    '''
    Filter descriptor matches using descriptor ratio.
    kp1: list of 1st set of keypoints.
    kp2: list of 2nd set of keypoints.
    matches: list of DMatch objects .
    ratio: maximum ratio of retained matches.
    '''

    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])

    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])

    return p1, p2, zip(mkp1, mkp2)


def visualize_matches(img1, img2, kp_pairs, status=None, H=None):
    '''
    Visualize features matches.
    img1: first image.
    img2: second image.
    kp_pairs: corresponding keypoint pairs in both images.
    status: list of bool specifying what keypoints to draw (None = all)
    H: homography to use for visualizing transformed image boundaries (optional)
    '''

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    vis[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)

    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    blue = (255, 0, 0)
    red = (0, 0, 255)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if not inlier:
            r = 2
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), red, 3)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), red, 3)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), red, 3)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), red, 3)

    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), blue)

    return vis


# parse and check args

parser = argparse.ArgumentParser(description='Transfer IC labels between images of the same PCB.')
parser.add_argument('--api', type=str, required=True, help='Directory that contains the DSLR dataset python API file.')
parser.add_argument('--db', type=str, required=True, help='Root directory of the DSLR dataset.')
parser.add_argument('--pcb', type=int, required=True, help='ID of the PCB to process.')
parser.add_argument('--from', dest='from_', type=int, required=True, help='Source recording.')
parser.add_argument('--to', type=int, required=True, help='Destination recording.')
parser.add_argument('--write', action='store_true', help='Write transfered IC data to disk.')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
args = parser.parse_args()

if not os.path.isdir(args.api):
    sys.exit('"{}" is not a directory'.format(args.api))

if not os.path.isdir(args.db):
    sys.exit('"{}" is not a directory'.format(args.db))

if args.from_ == args.to:
    sys.exit('--from and --to must be different')

try:
    sys.path.insert(0, args.api)
    from pcb_dataset import PCBDataset
except:
    sys.exit('Failed to import DSLR dataset API .. wrong directory?')

# load data

db = PCBDataset(args.db)
pcb = db.pcb(args.pcb)

from_im = pcb.image(args.from_)
from_mask = pcb.mask(args.from_)
from_im[from_mask == 0] = 0

to_im = pcb.image(args.to)
to_mask = pcb.mask(args.to)
to_im[to_mask == 0] = 0

from_ics = pcb.ics(args.from_)

print('PCB {}: {} => {}'.format(args.pcb, args.from_, args.to))

# detect keypoints

detector = cv2.SURF(1000)

from_kp, from_desc = detector.detectAndCompute(from_im, None)
to_kp, to_desc = detector.detectAndCompute(to_im, None)

if len(from_kp) > 5000:  # sorted desc by confidence
    from_kp = from_kp[:5000]
    from_desc = from_desc[:5000]

if len(to_kp) > 5000:  # sorted desc by confidence
    to_kp = to_kp[:5000]
    to_desc = to_desc[:5000]

print('Detected {} / {} keypoints'.format(len(from_kp), len(to_kp)))

# match them, filter, visualize

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(from_desc, to_desc, k=2)

pt_from, pt_to, matches = filter_matches(from_kp, to_kp, matches)

if len(matches) >= 4:
    H, status = cv2.findHomography(pt_from, pt_to, cv2.RANSAC, 5.0)
    print('{} good matches, {} total'.format(np.sum(status), len(status)))

    vis = visualize_matches(from_im, to_im, matches, status, H)
else:
    sys.exit('Could not find at least 4 good matches')

# transfer IC bounding boxes

to_ics = copy.deepcopy(from_ics)
for ic in to_ics:
    bp = cv2.cv.BoxPoints(ic.rect)
    bpn = np.asarray([bp])
    bpt = cv2.perspectiveTransform(bpn, H)
    bpt = np.int0(bpt)
    ic.rect = cv2.minAreaRect(bpt)

# show results

for ic in from_ics:
    bp = cv2.cv.BoxPoints(ic.rect)
    bp = np.int0(bp)
    cv2.drawContours(vis, [bp], 0, (0, 255, 0), 3)

dx = from_im.shape[1]

for ic in to_ics:
    bp = cv2.cv.BoxPoints(ic.rect)
    bp = np.int0(bp)

    bp[:, 0] += dx

    cv2.drawContours(vis, [bp], 0, (0, 255, 0), 3)

vis = cv2.resize(vis, (0, 0), fx=0.3, fy=0.3)

cv2.imshow('Matches', vis)
cv2.waitKey(0)

if args.write:
    spath = os.path.join(pcb._root, 'rec{}-annot.txt'.format(args.to))
    if os.path.exists(spath) and not args.overwrite:
        sys.exit('File "{}" already exists (use --overwrite)'.format(spath))

    with open(spath, 'w') as f:
        for ic in to_ics:
            f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.3f} {}\n'.format(ic.rect[0][0], ic.rect[0][1], ic.rect[1][0], ic.rect[1][1], ic.rect[2], ic.text))

    print('Transfered labels written to "{}"'.format(os.path.basename(spath)))
