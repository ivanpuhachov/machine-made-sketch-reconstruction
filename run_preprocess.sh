#!/bin/sh
set -e
set -o pipefail
# Any subsequent commands which fail will cause the shell script to exit immediately

image=${1:-"examples/Cylindrical_Parts_011_1.png"}

filename=$(basename ${image})
name=${filename//.png/}
workfolder=$(realpath results/${name})
echo $workfolder
# create directory
mkdir -p $workfolder
mkdir -p $workfolder/reports
mkdir -p $workfolder/npz

# copy original image
cp $image ${workfolder}/${filename}
python autocontrast.py --input ${workfolder}/${filename} --output ${workfolder}/${filename}
# bash balance.sh "-30,-25,0" ${workfolder}/${filename} ${workfolder}/${filename}
#convert $workfolder/${filename} -trim $workfolder/${filename}
#convert -bordercolor white -border 30 $workfolder/${filename} $workfolder/${filename}

# run model evaluation from conda environment
echo "run predictions"
bash run_models.sh ${workfolder}/${filename} ${workfolder}
conda run -n mmsketch --no-capture-output bash run_models.sh ${workfolder}/${filename} ${workfolder}

# clean segmentation with trapped ball
conda run -n mmsketch --no-capture-output python refine_segmentation.py --input ${workfolder}/npz/${name}_segm.npz

# # vectorize resized image

# convert "$(pwd;)"/$image -resize 512x512 ${workfolder}/${name}_512.png
bash morphology.sh -m grayscale -t erode ${workfolder}/${name}_resized512.png $workfolder/${name}_512_erode.png
# bash balance.sh "-30,-25,0" ${workfolder}/${name}_512_erode.png $workfolder/${name}_512_balanced.png

inputimage=${workfolder}/${name}_512_erode.png
# inputimage=${workfolder}/${name}_resized512.png
echo $inputimage

### Run vectorization
ptsfile=$workfolder/keypoints.pts
svgfile=$workfolder/${name}.svg
cleansvgfile=$workfolder/${name}_clean.svg
txtfile=$workfolder/vectorization_logs.txt

#echo $ptsfile
conda run -n mmsketch --no-capture-output python vectorize/prediction/usemodel.py --model vectorize/prediction/best_model_checkpoint.pth --input $inputimage --output $ptsfile
# this line is required for vectorization via ssh
export QT_QPA_PLATFORM=offscreen
vectorize/build/vectorize $inputimage $ptsfile $svgfile

# clean svg file
echo "Cleaning"
conda run -n mmsketch --no-capture-output python svg_cleaner.py --input $svgfile --output $cleansvgfile
