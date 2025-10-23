import os
from natsort import natsorted
import json

imagepath = "../infer/BUSBRA/laplacian/" ###### 其实没有用到，修改地方1: real image path: "../train/swe/"
imagelist = natsorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')]) # os.listdir生成列表，natsorted按照文件名从大到小排序

condpath = "../infer/BUSBRA/laplacian/" ###### 修改地方2: condition image path: "../train/us/"; "../train/canny/"; "../train/laplacian/"
condlist = natsorted([f for f in os.listdir(condpath) if f.lower().endswith('.png')])

captions = []
for index in range(len(imagelist)):
    oriDict = {}
    root = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/infer/BUSBRA/"
    if imagelist[index].split("_")[0] == "benign": # 得到列表
        oriDict["source"] = root + "laplacian/" + condlist[index] # source是condition
        oriDict["target"] = root + "laplacian/" + imagelist[index] # 这里不改也没事，反正用不到
        oriDict["prompt_target"] = "a photo of a benign breast cancer."
        captions.append(oriDict)
    else:
        oriDict["source"] = root + "laplacian/" + condlist[index] # source是condition
        oriDict["target"] = root + "laplacian/" + imagelist[index] # 这里不改也没事，反正用不到
        oriDict["prompt_target"] = "a photo of a malignant breast cancer."
        captions.append(oriDict)


with open("../infer/metadata_laplacian.json", 'w') as f: ###### 修改地方3: metadata_us.json; metadata_canny.json; metadata_laplacian.json
    for item in captions:
        f.write(json.dumps(item) + "\n")