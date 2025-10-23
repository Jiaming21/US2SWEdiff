import os
from natsort import natsorted

name = "BreastCA-img" # file name of stored data 

imagepath = f"./{name}/train/all/swe/" # real image path 
imagelist = natsorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')]) # os.listdir生成列表，natsorted按照文件名从大到小排序

cannypath = f"./{name}/train/all/us/"
cannylist = natsorted([f for f in os.listdir(cannypath) if f.lower().endswith('.png')])

captions = []
for index in range(len(imagelist)):
    oriDict = {}
    root = f"/root/autodl-tmp/ControlNet-main/data/{name}/train/all/"
    if imagelist[index].split("_")[2] == "benign": # 得到列表
        oriDict["source"] = root + "us/" + cannylist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = f"a photo of a benign breast cancer."
        captions.append(oriDict)
    else:
        oriDict["source"] = root + "us/" + cannylist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = f"a photo of a malignant breast cancer."
        captions.append(oriDict)

import json
# path to the folder containing the images
root = f"./{name}/train/all/"

# add metadata.json file to this folder
with open(root + "metadata_us.json", 'w') as f:
    for item in captions:
        f.write(json.dumps(item) + "\n")