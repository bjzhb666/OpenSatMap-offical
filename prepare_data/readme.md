# OpenSatMap Data Preparation 
This folder is used to preprocess the data of the project. The data is stored in the folder `data/` and the preprocessed data is stored in the folder `processed_data/`. The data is preprocessed in the following way:
1. Put the data in picuse20/use or 19picuse/use
2. Prepare the json data
3. run the bash script
```bash
bash tools-release/run_all.sh
```
Note: You should comment/uncomment some code in the script to run the desired data.

Overall file structure:
```
├── 19picsuse
│   └── use/*.png
├── picuse20
│   └── use/*.png
├── 8.13
│   └── all/*.json
├── tools
│   ├── calculate_pixel_level_nums.py
│   ├── change_pixel.py
│   ├── check_npy.sh
│   ├── check_transfer.py
│   ├── cut_debug.sh
│   ├── cut_satellite.py
│   ├── draw_gt_tx.py
│   ├── find_seg_image_nusc.py
│   ├── generate_GT_direction_tag-multi.py
│   ├── generate_GT_direction_tag.py
│   ├── generate_GT_multiclass_only-multi.py
│   ├── generate_GT_multiclass_only.py
│   ├── get_black_images.py
│   ├── merge_4096_images.py
│   ├── merge_small_images.py
│   ├── readme.md
│   ├── remove_black_image.sh
│   ├── remove_black.py
│   ├── rename_19_pics.py
│   ├── run_all.sh
│   ├── run_black_images.sh
│   ├── run_cut_satellite.sh
│   ├── train_val_test_gt.py
│   ├── train_val_test_pic.py
│   ├── train_val_test_tag.py
│   └── visualize_gray_image.py

```
4. Move the processed data to `data/` folder