export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash tools/dist_test.sh configs/segnext/segnext_instance20_nusc.py \
 work_dirs/0604smallerclassweightdilate5/iter_40000.pth 8 \
    --cfg-options test_dataloader.batch_size=4 val_dataloader.batch_size=4 default_hooks.logger.interval=1 \
    --out work_dirs/nusc1024 
