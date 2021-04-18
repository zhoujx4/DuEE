# 项目说明  
百度2021年语言与智能技术竞赛多形态信息抽取赛道事件抽取部分Pytorch版baseline
比赛链接:https://aistudio.baidu.com/aistudio/competition/detail/65?isFromLuge=true

官方的baseline版本是基于paddlepaddle框架的,我把它改写成了Pytorch框架,其中大部分代码沿用的是官方提供的代码（如评测代码、保存预测文件代码等） ,只是对框架部分进行了修改,习惯用Pytorch版本的可以基于此进行优化.

# 环境
- python=3.6
- torch=1.7
- transformers=4.5.0




# 先训练生成训练数据脚本
```
python3 duee_1_data_prepare.py
python3 duee_fin_data_prepare.py
```

# DuEE-1.0
- Duee-1.0的trigger序列标注模型训练
```
python run_ner.py
--dataset=DuEE1.0
--event_type=trigger
--max_len=200
--per_gpu_train_batch_size=120
--per_gpu_eval_batch_size=180
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--linear_learning_rate=1e-4
--early_stop=2``
```
- Duee1.0的role序列标注模型训练
```
python run_ner.py
--dataset=DuEE1.0
--event_type=role
--max_len=200
--per_gpu_train_batch_size=120
--per_gpu_eval_batch_size=180
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--linear_learning_rate=1e-4
--early_stop=2
```

- Duee1.0的trigger序列标注模型预测
```
python predict_ner.py
--dataset=DuEE1.0
--event_type=trigger
--max_len=250
--per_gpu_eval_batch_size=180
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--fine_tunning_model_path=./output/DuEE1.0/trigger/best_model.pkl
--test_json=./data/DuEE1.0/duee_test1.json
```
- Duee1.0的role序列标注模型预测
```
python predict_ner.py
--dataset=DuEE1.0
--event_type=role
--max_len=250
--per_gpu_eval_batch_size=180
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--fine_tunning_model_path=./output/DuEE1.0/role/best_model.pkl
--test_json=./data/DuEE1.0/duee_test1.json
```
- Duee1.0的预测结果后处理，生成预测文件
```
python duee_1_data_prepare.py
--trigger_file=./output/DuEE1.0/trigger/test_result.json
--role_file=./output/DuEE1.0/role/test_result.json
--schema_file=./conf/DuEE1.0/event_schema.json
--save_path=./output/DuEE1.0/duee.json
```

# DuEE-Fin
- DuEE-Fin的trigger序列标注模型训练
```
python run_ner.py
--dataset=DuEE-Fin
--event_type=trigger
--max_len=400
--per_gpu_train_batch_size=50
--per_gpu_eval_batch_size=120
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--linear_learning_rate=1e-4
--early_stop=2
```
# 
- DuEE-Fin的role序列标注模型训练
```
python run_ner.py
--dataset=DuEE-Fin
--event_type=role
--max_len=400
--per_gpu_train_batch_size=50
--per_gpu_eval_batch_size=120
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--linear_learning_rate=1e-4
--early_stop=2
```

- DuEE-Fin的enum分类模型训练
```
python run_cls.py
--dataset=DuEE-Fin
--event_type=enum
--max_len=400
--per_gpu_train_batch_size=50
--per_gpu_eval_batch_size=120
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--linear_learning_rate=1e-4
--early_stop=2
```
- Duee-Fin的trigger预测
```
python predict_sequence_labeling.py
--dataset=DuEE-Fin
--event_type=trigger
--max_len=400
--per_gpu_eval_batch_size=250
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--fine_tunning_model_path=./output/DuEE-Fin/trigger/best_model.pkl
--test_json=./data/DuEE-Fin/sentence/test.json
```
- Duee-Fin的role预测
```
python predict_sequence_labeling.py
--dataset=DuEE-Fin
--event_type=role
--max_len=400
--per_gpu_eval_batch_size=180
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--fine_tunning_model_path=./output/DuEE-Fin/role/best_model.pkl
--test_json=./data/DuEE-Fin/sentence/test.json
```
- Duee-Fin的enum预测
```
python predict_cls.py
--dataset=DuEE-Fin
--event_type=enum
--max_len=400
--per_gpu_eval_batch_size=180
--model_name_or_path=/data/zhoujx/prev_trained_model/rbt3
--fine_tunning_model_path=./output/DuEE-Fin/enum/best_model.pkl
--test_json=./data/DuEE-Fin/sentence/test.json
```
- Duee-Fin的预测结果后处理，生成预测文件
```
python duee_fin_postprocess.py
--trigger_file=./output/DuEE-Fin/trigger/test_result.json
--role_file=./output/DuEE-Fin/role/test_result.json
--enum_file=./output/DuEE-Fin/enum/test_result.json
--schema_file=./conf/DuEE-Fin/event_schema.json
--save_path=./output/DuEE-Fin/duee-fin.json

```



# 效果

为了速度，用的都是rbt3模型（3层的roberta），用更大的模型效果肯定会有更多的提升。

![image-20210418145715313](https://raw.githubusercontent.com/zhoujx4/PicGo/main/img/image-20210418145715313.png)



# 后续优化

- 处理数据的方法
- 清洗数据
- 数据增广
- 模型架构