#!/bin/bash
for f in examples/*.png; do
    filename=$(basename ${f})
    name=${filename//.png/}
    bash run_preprocess.sh $f
    echo $name
    python main.py --pngname ${name} 2>&1 | tee output/${name}.txt
done
