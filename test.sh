CUDA_VISIBLE_DEVICES=0 python -u test_demo.py \
--method_name single-scale \
--dist videos/2999049224_original_centercrop_960x540_8s.mp4 \
--output result.txt \
--is_gpu \
>> logs/test_demo.log