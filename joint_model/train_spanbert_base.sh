python train.py \
 --trn_data ./data/trn_SpanBERTspanbert-large-cased_batches.json \
 --dev_data ./data/dev_SpanBERTspanbert-large-cased_batches.json \
 --tst_data ./data/tst_SpanBERTspanbert-large-cased_batches.json \
 --model SpanBERT/spanbert-base-cased \
 --dev_keys ../data/data_set_keys_scene_dev.json  \
 --trn_keys ../data/data_set_keys_scene_trn.json \
 --tst_keys ../data/data_set_keys_scene_tst.json \
 --num_epochs 101 \
 --fp16 