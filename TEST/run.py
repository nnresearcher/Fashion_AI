'''
Created on 2018��4��23��

@author: hasee
'''
from models import model_build
from basic_function import label_dict, predict
import pandas as pd
###�������������������������������������ڲ��Լ����ԡ�����������������������������������######

####������������������������·����������������������������#######
model_name = r"../TRAIN/my_model.h5"
predict_data_path = r"../data/rank"
question_path = r"../data/rank/Tests/question.csv"
output_path = r"../data/rank/Tests/rank_data_predicted.csv"
####������������������������·������������������������������#######
    
####�����������������������ò�����������������������������#######
pic_size = 448
class_num = 54
dropout = 0.5
####�����������������������ò�������������������������������#######

####������������������������ģ�͡�������������������������#######
model = model_build(input_shape=(pic_size, pic_size, 3), classes=class_num, dropout=dropout)
model.load_weights(model_name)
####������������������������ģ�͡���������������������������#######

####����������������������ȡ���ݡ�������������������������#######
test_data = pd.read_csv(question_path, header=None)
test_data.columns = ["image_path", "classes", "predict_output"]
####����������������������ȡ���ݡ���������������������������#######
    
####��������������������Ԥ�����ݡ�������������������������#######
for i in range(len(test_data)):
    each_image_path = "/".join([predict_data_path, test_data["image_path"][i]])
    predict_output = predict(each_image_path, model, (pic_size, pic_size))
    predict_output = predict_output.tolist()[0]
    first_index, end_index = label_dict(test_data["classes"][i])
    predict_output_keep = predict_output[first_index:end_index]
    predict_output_keep_norm = [x/sum(predict_output_keep) for x in predict_output_keep]
    predict_output_use = ";".join([str(round(x, 4)) for x in predict_output_keep_norm])
    test_data["predict_output"][i] = predict_output_use
test_data.to_csv(output_path, index=False, header=None)
    ####��������������������Ԥ�����ݡ���������������������������#######
###���������������������������������������������������ڲ��Լ����ԡ�������������������������������������������������######