# 房产行业聊天问答匹配
比赛网址 https://www.datafountain.cn/competitions/474/datasets
## 介绍
本项目基于pytorch和transformer包来进行问答匹配建模， 

data中为数据文件  

EDA.ipynb为数据处理文件，竞赛方给定的数据需要通过EDA代码处理之后保存之后才能用于后边的py代码  

run.py为单模型未五折交叉验证的  

run_cv.py是五折交叉验证的模型   

model_name 选择模型集合 ["bert-base-chinese", "hfl/chinese-roberta-wwm-ext", "hfl/chinese-bert-wwm", "hfl/chinese-bert-wwm-ext", "nghuyong/ernie-1.0"] 其中之一时仅需要将模型名字赋值给model_name即可  
当使用large模型时需要把hidden_size设为1024   

模型详情可见https://github.com/ymcui/Chinese-BERT-wwm  

"hfl/chinese-bert-wwm" 五折交叉验证可达到0.78+效果




