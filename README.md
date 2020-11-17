# 房产行业聊天问答匹配竞赛
比赛网址 https://www.datafountain.cn/competitions/474/datasets
## 介绍
本项目基于pytorch和transformer包来实现基于bert，roberta，ernie等模型实现问答匹配建模， 

data中为数据文件，原始test文件为gbk编码，本项目已将test文件改为utf-8编码  

EDA.ipynb为数据处理文件，竞赛方给定的数据需要通过EDA代码处理之后保存之后才能用于后边的py代码  

run.py为单模型未五折交叉验证的  

run_cv.py是五折交叉验证的模型   

run.py, run_cv.py均为端到端的，训练测试验证一条龙，最终提交文件保存在./data/下

model_name 选择模型集合 ["bert-base-chinese", "hfl/chinese-roberta-wwm-ext", "hfl/chinese-bert-wwm", "hfl/chinese-bert-wwm-ext", "nghuyong/ernie-1.0"] 其中之一时仅需要将模型名字赋值给model_name即可，模型文件会自动下载！！！！！但是下载时候需要翻墙，如果用google colab的gpu环境可以自动下载。若使用large模型时需要把hidden_size设为1024   

## 由于huggle face上的模型文件只能翻墙下载，所以我上传到微云上chinese-bert-wwm模型，链接：https://share.weiyun.com/akjxcHbL 密码：fcea6p  解压到项目下，然后将model_name改为解压后的文件夹名字即可。需要将其中的bert_config.json改为config.json 或者用这个链接： https://share.weiyun.com/6Sgi7PIA （密码：o0hz）

其他模型下载地址可见这个github https://github.com/ymcui/Chinese-BERT-wwm
 
"hfl/chinese-bert-wwm" 五折交叉验证可达到0.78+效果


- 有问题可以直接提issues,看到肯定回答




