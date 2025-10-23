from natsort import natsorted
import os

name = "BreastCA-img"
swepath = f"/root/autodl-tmp/ControlNet-main/data/{name}/train/all/swe/"
swelist = natsorted([f for f in os.listdir(swepath) if f.lower().endswith('.png')]) 

uspath = f"/root/autodl-tmp/ControlNet-main/data/{name}/train/all/us/"
uslist = natsorted([f for f in os.listdir(uspath) if f.lower().endswith('.png')]) 

swe = []
us = [] # two missing
for name in swelist:
    parts = name.split("_")
    swe.append(f"{parts[2]}_{parts[3]}")
    
# print(swe)
    
for name in uslist:
    parts = name.split("_")
    us.append(f"{parts[2]}_{parts[3]}")

# print(us)

# print("swe", len(swe))
# print("us", len(us))

print("swe", len(set(swe)))
print("us", len(set(us)))

missing_elements = list(set(swe) - set(us))
print("缺失的元素:", missing_elements)