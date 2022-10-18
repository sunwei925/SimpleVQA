# SimpleVQA
A Deep Learning based No-reference Quality Assessment Model for UGC Videos
## Description
This is a repository for the model proposed in the paper "A Deep Learning based No-reference Quality Assessment Model for UGC Videos". [Arxiv Version](https://arxiv.org/abs/2204.14047) [ACM MM 2022 Version](https://dl.acm.org/doi/10.1145/3503161.3548329)

## Usage

### Install Requirements
```
pytorch
opencv
scipy
pandas
torchvision
torchvideo
```

### Download databases
[LSVQ](https://github.com/baidut/PatchVQ)
[KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html)
[Youtube-UGC](https://media.withyoutube.com/)

### Train models
1. Extract video frames
```shell
python -u extract_frame_LSVQ.py >> logs/extract_frame_LSVQ.log
```
2. Extract motion features
```shell
 CUDA_VISIBLE_DEVICES=0 python -u extract_SlowFast_features_LSVQ.py \
 --database LSVQ \
 --model_name SlowFast \
 --resize 224 \
 --feature_save_folder LSVQ_SlowFast_feature/ \
 >> logs/extracted_LSVQ_SlowFast_features.log
```
3. Train the model
```shell
 CUDA_VISIBLE_DEVICES=0 python -u train_baseline.py \
 --database LSVQ \
 --model_name UGC_BVQA_model \
 --conv_base_lr 0.00001 \
 --epochs 10 \
 --train_batch_size 8 \
 --print_samples 1000 \
 --num_workers 6 \
 --ckpt_path ckpts \
 --decay_ratio 0.9 \
 --decay_interval 2 \
 --exp_version 0 \
 --loss_type L1RankLoss \
 --resize 520 \
 --crop_size 448 \
 >> logs/train_UGC_BVQA_model_L1RankLoss_resize_520_crop_size_448_exp_version_0.log
```
### Test the model
You can download the trained model via [Google Drive](https://drive.google.com/file/d/137XJdq3reNMJ9tkBNqKUYTY_dTlcwXc3/view?usp=sharing).

Test on the public VQA database
```shell
CUDA_VISIBLE_DEVICES=0 python -u test_on_pretrained_model.py \
--database KoNViD-1k \
--train_database LSVQ \
--model_name UGC_BVQA_model \
--feature_type SlowFast \
--trained_model ckpts/UGC_BVQA_model.pth \
--num_workers 6 \
>> logs/test_on_KoNViD-1k_train_on_LSVQ.log
```

Test on a single video
```shell
CUDA_VISIBLE_DEVICES=0 python -u test_demo.py \
--method_name single-scale \
--dist videos/2999049224_original_centercrop_960x540_8s.mp4 \
--output result.txt \
--is_gpu \
>> logs/test_demo.log
```

### Citation

If you find this code is useful for your research, please cite:
```
@inproceedings{10.1145/3503161.3548329,
author = {Sun, Wei and Min, Xiongkuo and Lu, Wei and Zhai, Guangtao},
title = {A Deep Learning Based No-Reference Quality Assessment Model for UGC Videos},
year = {2022},
isbn = {9781450392037},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3503161.3548329},
doi = {10.1145/3503161.3548329},
pages = {856–865},
numpages = {10},
series = {MM '22}
}
```