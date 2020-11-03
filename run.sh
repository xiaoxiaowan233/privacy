#!/usr/bin/env bash

NET="Vgg"
# BIN=True
while getopts 'n:f' OPT; do
    case ${OPT} in
      n)
        NET=${OPTARG:0}
        ;;
      *)
        echo "无效的参数"
        ;;
    esac
done

if [ ! -d ${NET} ]; then 
　　echo "没有该目录: "${NET}
else
    cd ${NET}
    if [ ${NET} = "Vgg" ]; then
      CUDA_VISIBLE_DEVICES=1 python3 bin-net-train.py -dataset=cifar-100-python
   

    else
      CUDA_VISIBLE_DEVICES=1 python3 bin-net-train.py -dataset=cifar-100-python -resize_image=True

    fi
fi 

# export $HE_TRANSFORMER=$(pwd)