# 识别/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/all/swe/目录下全部的文件
# [3]识别是否为malignant_, split分割[3]换成malignant

from natsort import natsorted
import os

name = "BreastCA-img"
imagepath = f"/root/autodl-tmp/ControlNet-main/data/{name}/train/all/swe/"
imagelist = natsorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')]) 

for old_name in imagelist:
    parts = old_name.split("_")
    # print(parts)
    print(parts[2])
    if parts[2] == 'malignant ':
        parts[2] = "malignant"
        new_name = "_".join(parts)
        print(new_name)
        # print(new_name)
        os.rename(os.path.join(imagepath, old_name), os.path.join(imagepath, new_name))

# 删除._*.png文件