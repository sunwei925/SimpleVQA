 CUDA_VISIBLE_DEVICES=0 python -u extract_SlowFast_features_LSVQ.py \
 --database LSVQ \
 --model_name SlowFast \
 --resize 224 \
 --feature_save_folder LSVQ_SlowFast_feature/ \
 >> logs/extracted_LSVQ_SlowFast_features.log

 CUDA_VISIBLE_DEVICES=0 python -u extracted_SlowFast_features_VQA.py \
 --database KoNViD-1k \
 --model_name SlowFast \
 --resize 224 \
 --feature_save_folder konvid1k_SlowFast_feature/ \
 >> logs/extracted_konvid1k_SlowFast_features.log

 CUDA_VISIBLE_DEVICES=1 python -u extracted_SlowFast_features_VQA.py \
 --database youtube_ugc \
 --model_name SlowFast \
 --resize 224 \
 --feature_save_folder youtube_ugc/youtube_ugc_SlowFast_feature/ \
 >> logs/extracted_youtube_ugc_SlowFast_features.log