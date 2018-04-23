# -*- coding: utf-8 -*-
'''
@author: hasee
'''

import pandas as pd
import os
import random
def label_dict(label):
    if label=="skirt_length_labels":
        return 0,6
    if label=="coat_length_labels":
        return 6,14
    if label=="collar_design_labels":
        return 14,19
    if label=="lapel_design_labels":
        return 19,24
    if label=="neck_design_labels":
        return 24,29
    if label=="neckline_design_labels":
        return 29,39
    if label=="pant_length_labels":
        return 39,45
    if label=="sleeve_length_labels":
        return 45,54

def change_label_binary(original_label,class_label):
    replace_n=original_label.replace("n","0;")
    replace_y=replace_n.replace("y","1;")
    replace_m=replace_y.replace("m","0;")
    first_index, end_index = label_dict(class_label)
    new_label=[]
    for _ in range(0,54):
        new_label.append(-1)
    new_label[first_index:end_index] = replace_m
    return new_label


def get_image_and_label(image_path,label_path,save_path):
    label_csv=pd.read_csv(label_path, header=None)
    label_csv.columns = ["image", "label_class", "label"]
    routes=os.listdir(image_path)
    input_image_Path = []
    output_label_code = []
    print("creat_preprocess_csv ing......")
    for route in routes:
        imagepath = "/".join([image_path, route])
        for _,_,files in os.walk(imagepath):
            for file in files:
                image = "/".join([imagepath, file])
                input_image_Path.append(image)
                label_info = label_csv[label_csv["image"]=="/".join(["Images", route, file])].reset_index(drop=True)
                label = change_label_binary(label_info['label'][0], label_info["label_class"][0])
                output_label_code.append(label)
    save_csv = pd.DataFrame({"image_path":input_image_Path, "label":output_label_code})
    save_csv.to_csv(save_path, index=False)        
    return input_image_Path, output_label_code

test_in_all_data_size=0.15
input_image_path = r'../../data/base/Images' 
output_label_path = r'../../data/base/Annotations/label.csv' 
imagelabel_path = r"imagelabel.csv" 
data_divide_nto_train_and_test = r"train_and_test_lable.csv"

get_image_and_label(image_path=input_image_path, label_path=output_label_path, save_path=imagelabel_path)
image_label = pd.read_csv(imagelabel_path)
image_label["test_in_all_data_size"] = test_in_all_data_size
image_label["flag"] = pd.Series(map(lambda x:0 if random()<x else 1, image_label["test_in_all_data_size"])) 
del image_label["test_in_all_data_size"]
image_label.to_csv(data_divide_nto_train_and_test, index=False)
print("code over.")