sh distributed_train.sh	10 /path_to/ImageNet/2012/ 	--train-split /path_to/ImageNet/2012/ILSVRC2012_img_train 	--val-split /path_to/ImageNet/2012/ILSVRC2012_img_val 	--model convmixer_1536_20_dcls 	-b 64     -j 10     --opt adamw     --epochs 150     --sched onecycle     --amp     --input-size 3 224 224    --lr 0.01     --aa rand-m9-mstd0.5-inc1     --cutmix 0.5     --mixup 0.5     --reprob 0.25     --remode pixel     --num-classes 1000     --warmup-epochs 0     --opt-eps=1e-3     --clip-grad 1.0

