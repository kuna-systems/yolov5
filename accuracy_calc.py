import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (LOGGER, check_requirements, colorstr, increment_path, non_max_suppression,
                           print_args, scale_coords, xywh2xyxy)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


@torch.no_grad()
def run(
        imgsz=640,  # inference size (pixels)
        single_cls=False,  # treat as single-class dataset
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        plots=True,
        names={0: 'person', 1: 'bicycle', 2: 'car', 3: 'animal'}, # names for classes
        path_test = ROOT / 'test_yolo_dataset',
        path_test_pred = ROOT / 'test_yolo_dataset_predictions'
):
    # Directories
    save_dir = increment_path(Path(project) / name)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Configure
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    seen = 0
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []
    nc = len(names)
    confusion_matrix = ConfusionMatrix(nc=nc)

    for i in tqdm(os.listdir(path_test_pred)):
        if '.txt' not in i:
            continue
        targets = []
        with open(os.path.join(path_test, i)) as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip().split()
                targets.append([0, int(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4])])

        shapes = [imgsz, imgsz]
        labels = []
        with open(os.path.join(path_test_pred, i)) as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip().split()
                xywh = [float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5]), float(tmp[1]),int(tmp[0])]
                xyxy = [xywh[0]-xywh[2]/2, xywh[1]-xywh[3]/2, xywh[0]+xywh[2]/2, xywh[1]+xywh[3]/2]
                labels.append([xyxy[0]*shapes[0], xyxy[1]*shapes[0], xyxy[2]*shapes[1], xyxy[3]*shapes[1], xywh[-2], xywh[-1]])

        out = [torch.from_numpy(np.array(labels))]
        targets = torch.from_numpy(np.array(targets))
        height, width = shapes
        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height))  # to pixels

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            correct = torch.zeros(npr, niou, dtype=torch.bool)  # init
            seen += 1
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((3, 0))))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(s)
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (nc < 50) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Return results
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map), maps


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--names', default={0: 'person', 1: 'bicycle', 2: 'car', 3: 'animal'}, type=dict,
                        help='names of classes')
    parser.add_argument('--path_test', default=ROOT / 'test_yolo_dataset', help='path to the test dataset')
    parser.add_argument('--path_test_pred', default=ROOT / 'test_yolo_dataset_predictions', help='path to predictions')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
