export CUDA_VISIBLE_DEVICES=7
# python tools/test.py configs/segnext/segnext_instance.py \
#  work_dirs/onlyadditionalhead-dilate3-0528/iter_80000.pth \
#     --show-dir work_dirs/debug  \
#     --cfg-options test_dataloader.batch_size=1 val_dataloader.batch_size=4 default_hooks.logger.interval=1
WORK_DIR=work_dirs/0813data-zoom19small
CONFIG=configs/segnext/segnext_instance19_small.py

python tools/test.py $CONFIG \
 $WORK_DIR/iter_40000.pth \
    --cfg-options test_dataloader.batch_size=4 val_dataloader.batch_size=4  \
     default_hooks.logger.interval=1 test_evaluator.instance_dir=$WORK_DIR/instance_dirdebug  \
     | tee $WORK_DIR/$(date +"%Y%m%d_%H%M%S").log
    #  default_hooks.visualization.draw=True --out $WORK_DIR/output
   #  --show-dir $WORK_DIR