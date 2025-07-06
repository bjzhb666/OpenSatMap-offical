export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1
CONFIG=configs/segnext/segnext_instance19_small.py
export PYTHONPATH=$(pwd):$PYTHONPATH
python tools/train.py \
    $CONFIG \
    --work-dir work_dirs/debug \
    --cfg-option train_dataloader.batch_size=1 train_dataloader.num_workers=0 \
        val_dataloader.num_workers=0 \
        train_dataloader.persistent_workers=False \
        val_dataloader.persistent_workers=False \
        test_dataloader.persistent_workers=False  \
        default_hooks.logger.interval=1 # \
        # train_cfg.val_interval=50
