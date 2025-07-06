export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
WORK_DIR=work_dirs/0813data-zoom20trainnew-linewidth6@zoom20
CONFIG=configs/segnext/segnext_instance20_small.py

bash tools/dist_test.sh $CONFIG \
 $WORK_DIR/iter_40000.pth 8 \
    --cfg-options test_dataloader.batch_size=4 val_dataloader.batch_size=4  \
     default_hooks.logger.interval=1 test_evaluator.instance_dir=$WORK_DIR/instance_dirtrain   \
     | tee $WORK_DIR/$(date +"%Y%m%d_%H%M%S").log
    #  default_hooks.visualization.draw=True --out $WORK_DIR/output
   #  --show-dir $WORK_DIR
