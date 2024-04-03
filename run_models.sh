#!/bin/sh

# This script runs vectorization pipeline on imagefile provided (.png or .jpg)

image=${1:-"images/Cylindrical_Parts_011_1.png"}
filename=$(basename ${image})
logfolder=${2:-"images/"}

backdepthfile=npz/${filename//.png/_backdepth.npz}
depthfile=npz/${filename//.png/_depth.npz}
segmfile=npz/${filename//.png/_segm.npz}
normfile=npz/${filename//.png/_norms.npz}

echo $image - $depthfile
python evaluate_model.py --image $image --depth --ckpt models/depth_03_12_top_epoch_35-valid_loss=0.0000836.ckpt --out ${logfolder}/${depthfile} --plots --saveto ${logfolder}/reports/

echo $image - $segmfile
python evaluate_model.py --image $image --segm --ckpt models/segm_02_14_top_epoch_35-valid_dataset_iou=0.9972253.ckpt --out ${logfolder}/${segmfile} --plots --saveto ${logfolder}/reports/

echo $image - $normfile
python evaluate_model.py --image $image --normals --ckpt models/normals_11_01_top_epoch_34-valid1_loss=0.0380605.ckpt --out ${logfolder}/${normfile} --plots --saveto ${logfolder}/reports/
