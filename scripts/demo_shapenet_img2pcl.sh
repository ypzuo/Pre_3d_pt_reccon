#!/bin/bash

gpu=0
exp=/media/z/local/3d-lmnet/expts/img2pcl_all_cat
dataset=shapenet
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
eval_set=valid
cat=car

python metrics_img2pcl.py \
	--gpu $gpu \
	--dataset $dataset \
	--data_dir_imgs ${data_dir_imgs} \
	--data_dir_pcl ${data_dir_pcl} \
	--exp $exp \
	--mode lm \
	--category $cat \
	--load_best \
	--bottleneck 512 \
	--bn_decoder \
	--eval_set ${eval_set} \
	--batch_size 24 \
	--visualize
