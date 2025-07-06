import os
import shutil

data_root = "./data/satellite19"
data_type = os.listdir(data_root)

selected = os.listdir("./data/satellite19/img_dir/test")[-50:]
# selected = ['Cities1to30_30cm_BBA_BGRN_22Q1_1303033001320_3_4_1024_1536_1536_2048.png']
for x in data_type:
    save_path = f"{data_root}/{x}/small_test"
    os.makedirs(save_path, exist_ok=True)

    data_path = f"{data_root}/{x}/test"
    data = os.listdir(data_path)

    for name in selected:
        if os.path.exists(os.path.join(data_path, name)):
            shutil.copy(os.path.join(data_path, name), os.path.join(save_path, name))
        else:
            shutil.copy(os.path.join(data_path, name.replace(".png", "-GT.png")), os.path.join(save_path, name.replace(".png", "-GT.png")))