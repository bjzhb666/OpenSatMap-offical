export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# bash tools/dist_train.sh \
#      configs/HLT/HLT_mscan-t_1xb8-adamw-160k_satellite-2048x2048-weight-1-20-seg.py \
#      8  --work-dir work_dirs/cut_pic1dilate_HLT_mscan-t_1xb8-adamw-160k_satellite-2048x2048-weight-1-20-seg 
WORK_DIR=work_dirs/0813data-zoom20trainnew-linewidth6@zoom20+zverskyloss
CONFIG=configs/segnext/segnext_instance20_small.py

PORT=8888 bash tools/dist_train.sh  \
     $CONFIG \
     8 --work-dir $WORK_DIR \
     --cfg-options default_hooks.checkpoint.interval=10000 # \
     # train_cfg.val_interval=50

bash tools/dist_test.sh $CONFIG \
 $WORK_DIR/iter_40000.pth 8 \
 --show-dir $WORK_DIR  \
 --cfg-options test_dataloader.batch_size=4 val_dataloader.batch_size=4 \
  test_evaluator.instance_dir=$WORK_DIR/instance_dir |
  tee $WORK_DIR/$(date +"%Y%m%d_%H%M%S").log