   CUDA_VISIBLE_DEVICES=0 python -u train_VR_all.py \
 --database VR_all \
 --model_name PCONV_RN18 \
 --conv_base_lr 0.00005 \
 --epochs 30 \
 --train_batch_size 8 \
 --print_samples 1000 \
 --num_workers 1 \
 --resize 512 \
 --crop_size 512 \
 --decay_ratio 0.9 \
 --decay_interval 2 \
 --loss_type plcc \
 >> logs/PCONV_RN18_epochs30_bs8_lr000005_plcc.log

#  CUDA_VISIBLE_DEVICES=0 python -u train_Scanpath_VR_all.py \
# --database VR_all \
# --model_name RN18 \
# --conv_base_lr 0.00003 \
# --epochs 30 \
# --train_batch_size 160 \
# --print_samples 1000 \
# --num_workers 1 \
# --resize 224 \
# --crop_size 224 \
# --decay_ratio 0.9 \
# --decay_interval 2 \
# --loss_type plcc \
# >> logs/Scanpath_RN18_epochs30_bs160_lr000005_plcc.log
