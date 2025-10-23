import os
from natsort import natsorted
import json

imagepath = "../train/swe/" ###### 修改地方1: real image path: "../train/swe/"
imagelist = natsorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')]) # os.listdir生成列表，natsorted按照文件名从大到小排序

condpath = "../train/us/" ###### 修改地方2: condition image path: "../train/us/"; "../train/canny/"; "../train/laplacian/"
condlist = natsorted([f for f in os.listdir(condpath) if f.lower().endswith('.png')])

captions = []
for index in range(len(imagelist)):
    oriDict = {}
    root = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/"
    if imagelist[index].split("_")[2] == "benign": # 得到列表
        oriDict["source"] = root + "us/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a benign breast cancer."
        captions.append(oriDict)
    else:
        oriDict["source"] = root + "us/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a malignant breast cancer."
        captions.append(oriDict)


with open("../train/metadata_us.json", 'w') as f: ###### 修改地方3: metadata_us.json; metadata_canny.json; metadata_laplacian.json
    for item in captions:
        f.write(json.dumps(item) + "\n")
        
###############################################################################################################################################    
        
import os
from natsort import natsorted
import json

imagepath = "../train/swe/" ###### 修改地方1: real image path: "../train/swe/"
imagelist = natsorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')]) # os.listdir生成列表，natsorted按照文件名从大到小排序

condpath = "../train/canny/" ###### 修改地方2: condition image path: "../train/us/"; "../train/canny/"; "../train/laplacian/"
condlist = natsorted([f for f in os.listdir(condpath) if f.lower().endswith('.png')])

captions = []
for index in range(len(imagelist)):
    oriDict = {}
    root = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/"
    if imagelist[index].split("_")[2] == "benign": # 得到列表
        oriDict["source"] = root + "canny/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a benign breast cancer."
        captions.append(oriDict)
    else:
        oriDict["source"] = root + "canny/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a malignant breast cancer."
        captions.append(oriDict)


with open("../train/metadata_canny.json", 'w') as f: ###### 修改地方3: metadata_us.json; metadata_canny.json; metadata_laplacian.json
    for item in captions:
        f.write(json.dumps(item) + "\n")
          
############################################################################################################################################### 

import os
from natsort import natsorted
import json

imagepath = "../train/swe/" ###### 修改地方1: real image path: "../train/swe/"
imagelist = natsorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')]) # os.listdir生成列表，natsorted按照文件名从大到小排序

condpath = "../train/laplacian/" ###### 修改地方2: condition image path: "../train/us/"; "../train/canny/"; "../train/laplacian/"
condlist = natsorted([f for f in os.listdir(condpath) if f.lower().endswith('.png')])

captions = []
for index in range(len(imagelist)):
    oriDict = {}
    root = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/"
    if imagelist[index].split("_")[2] == "benign": # 得到列表
        oriDict["source"] = root + "laplacian/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a benign breast cancer."
        captions.append(oriDict)
    else:
        oriDict["source"] = root + "laplacian/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a malignant breast cancer."
        captions.append(oriDict)


with open("../train/metadata_laplacian.json", 'w') as f: ###### 修改地方3: metadata_us.json; metadata_canny.json; metadata_laplacian.json
    for item in captions:
        f.write(json.dumps(item) + "\n")
###############################################################################################################################################
        
        
###############################################################################################################################################      

import os
from natsort import natsorted
import json

imagepath = "../test/swe/" ###### 修改地方1: real image path: "../train/swe/"
imagelist = natsorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')]) # os.listdir生成列表，natsorted按照文件名从大到小排序

condpath = "../test/us/" ###### 修改地方2: condition image path: "../train/us/"; "../train/canny/"; "../train/laplacian/"
condlist = natsorted([f for f in os.listdir(condpath) if f.lower().endswith('.png')])

captions = []
for index in range(len(imagelist)):
    oriDict = {}
    root = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/test/"
    if imagelist[index].split("_")[2] == "benign": # 得到列表
        oriDict["source"] = root + "us/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a benign breast cancer."
        captions.append(oriDict)
    else:
        oriDict["source"] = root + "us/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a malignant breast cancer."
        captions.append(oriDict)


with open("../test/metadata_us.json", 'w') as f: ###### 修改地方3: metadata_us.json; metadata_canny.json; metadata_laplacian.json
    for item in captions:
        f.write(json.dumps(item) + "\n")
        
###############################################################################################################################################

import os
from natsort import natsorted
import json

imagepath = "../test/swe/" ###### 修改地方1: real image path: "../train/swe/"
imagelist = natsorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')]) # os.listdir生成列表，natsorted按照文件名从大到小排序

condpath = "../test/canny/" ###### 修改地方2: condition image path: "../train/us/"; "../train/canny/"; "../train/laplacian/"
condlist = natsorted([f for f in os.listdir(condpath) if f.lower().endswith('.png')])

captions = []
for index in range(len(imagelist)):
    oriDict = {}
    root = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/test/"
    if imagelist[index].split("_")[2] == "benign": # 得到列表
        oriDict["source"] = root + "canny/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a benign breast cancer."
        captions.append(oriDict)
    else:
        oriDict["source"] = root + "canny/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a malignant breast cancer."
        captions.append(oriDict)


with open("../test/metadata_canny.json", 'w') as f: ###### 修改地方3: metadata_us.json; metadata_canny.json; metadata_laplacian.json
    for item in captions:
        f.write(json.dumps(item) + "\n")

###############################################################################################################################################
        
import os
from natsort import natsorted
import json

imagepath = "../test/swe/" ###### 修改地方1: real image path: "../train/swe/"
imagelist = natsorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')]) # os.listdir生成列表，natsorted按照文件名从大到小排序

condpath = "../test/laplacian/" ###### 修改地方2: condition image path: "../train/us/"; "../train/canny/"; "../train/laplacian/"
condlist = natsorted([f for f in os.listdir(condpath) if f.lower().endswith('.png')])

captions = []
for index in range(len(imagelist)):
    oriDict = {}
    root = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/test/"
    if imagelist[index].split("_")[2] == "benign": # 得到列表
        oriDict["source"] = root + "laplacian/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a benign breast cancer."
        captions.append(oriDict)
    else:
        oriDict["source"] = root + "laplacian/" + condlist[index] # source是condition
        oriDict["target"] = root + "swe/" + imagelist[index]
        oriDict["prompt_target"] = "a photo of a malignant breast cancer."
        captions.append(oriDict)


with open("../test/metadata_laplacian.json", 'w') as f: ###### 修改地方3: metadata_us.json; metadata_canny.json; metadata_laplacian.json
    for item in captions:
        f.write(json.dumps(item) + "\n")

###############################################################################################################################################  