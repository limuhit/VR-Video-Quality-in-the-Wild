"VR Video Quality in the Wild" 

*[1. Preparation]

 python setup.py install

 Remember to modify the configurations according to your machine in ncvv_args.

*[2. Database]

 Link: https://pan.baidu.com/s/18XEUNjMrOlIaqA2kQXEZfw?pwd=Fang 
 Password: Fang 

*[3. Run the preprocessor first]
 CUDA_VISIBLE_DEVICES=0 python erp_rotate/preprocessor.py

 or extract scanpaths for Scanpath-VQA 

 CUDA_VISIBLE_DEVICES=0 python erp_rotate/extract_scanpath.py

*[4. Train and test the proposed model]

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
 &gt;&gt; logs/PCONV_RN18_epochs30_bs8_lr000005_plcc.log


*[5. Train and test ERP-VQA]

 CUDA_VISIBLE_DEVICES=0 python -u train_VR_all.py \
  --database VR_all \
  --model_name RN18 \
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
 &gt;&gt; logs/PCONV_RN18_epochs30_bs8_lr000005_plcc.log


*[6. Train and test the Scanpath-VQA]

CUDA_VISIBLE_DEVICES=0 python -u train_Scanpath_VR_all.py \
 --database VR_all \
 --model_name RN18 \
 --conv_base_lr 0.00005 \
 --epochs 30 \
 --train_batch_size 160 \
 --print_samples 1000 \
 --num_workers 1 \
 --resize 224 \
 --crop_size 224 \
 --decay_ratio 0.9 \
 --decay_interval 2 \
 --loss_type plcc \
 &gt;&gt;  logs/Scanpath_RN18_epochs30_bs160_lr000005_plcc.log

*[7. Acknowledgement]
 https://github.com/sunwei925/MC360IQA
 https://github.com/limuhit/pseudocylindrical_convolution
