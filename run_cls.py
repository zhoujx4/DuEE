"""
@Time : 2021/4/158:09
@Auth : 周俊贤
@File ：run_ner.py.py
@DESCRIPTION:
"""

import copy
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, AdamW

from dataset.dataset import DuEEClsDataset, cls_collate_fn
from model.model import DuEECls_model
from utils.finetuning_argparse import get_argparse
from utils.utils import init_logger, seed_everything, logger, ProgressBar
from sklearn.metrics import f1_score

def evaluate(args, eval_iter, model):
    """evaluate"""
    batch_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(args.device)

    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_iter):
            for key in batch.keys():
                batch[key] = batch[key].to(args.device)
            logits = model(
                input_ids=batch['all_input_ids'],
                attention_mask=batch['all_attention_mask'],
                token_type_ids=batch['all_token_type_ids']
            )
            loss = criterion(logits, batch["all_labels"])
            batch_loss += loss.item()

            preds.extend(torch.argmax(logits, axis=-1).tolist())
            trues.extend(batch["all_labels"].tolist())

    f1 = f1_score(trues, preds, average="micro")

    return f1, batch_loss/(step+1)

def train(args, train_iter, model):
    logger.info("***** Running train *****")
    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.linear_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.linear_learning_rate},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(args.device)
    batch_loss = 0
    pbar = ProgressBar(n_total=len(train_iter), desc='Training')
    print("****" * 20)
    for step, batch in enumerate(train_iter):
        for key in batch.keys():
            batch[key] = batch[key].to(args.device)
        logits = model(
            input_ids=batch['all_input_ids'],
            attention_mask=batch['all_attention_mask'],
            token_type_ids=batch['all_token_type_ids']
        )
        # 正常训练
        loss = criterion(logits, batch["all_labels"])
        loss.backward()
        #
        batch_loss += loss.item()
        pbar(step,
             {
                 'batch_loss': batch_loss / (step + 1),
             })
        optimizer.step()
        model.zero_grad()

def main():
    args = get_argparse().parse_args()
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    init_logger(log_file="./log/{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    seed_everything(args.seed)

    args.output_model_path = os.path.join(args.output_dir, args.dataset, args.event_type, "best_model.pkl")
    # 设置保存目录
    if not os.path.exists(os.path.dirname(args.output_model_path)):
        os.makedirs(os.path.dirname(args.output_model_path))

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("/data/zhoujx/prev_trained_model/rbt3")

    # dataset & dataloader
    args.train_data = "./data/{}/{}/train.tsv".format(args.dataset, args.event_type)
    args.dev_data = "./data/{}/{}/dev.tsv".format(args.dataset, args.event_type)
    args.tag_path = "./conf/{}/{}_tag.dict".format(args.dataset, args.event_type)
    train_dataset = DuEEClsDataset(args,
                                   args.train_data,
                                   args.tag_path,
                                   tokenizer)
    eval_dataset = DuEEClsDataset(args,
                                  args.dev_data,
                                  args.tag_path,
                                  tokenizer)
    logger.info("The nums of the train_dataset features is {}".format(len(train_dataset)))
    logger.info("The nums of the eval_dataset features is {}".format(len(eval_dataset)))
    train_iter = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=args.per_gpu_train_batch_size,
                            collate_fn=cls_collate_fn,
                            num_workers=10)
    eval_iter = DataLoader(eval_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=cls_collate_fn,
                           num_workers=10)

    # 用于evaluate
    args.id2label = train_dataset.label_vocab
    args.num_classes = len(args.id2label)
    # metric = ChunkEvaluator(label_list=args.id2label.keys(), suffix=False)

    # model
    model = DuEECls_model(args.model_name_or_path, num_classes=args.num_classes)
    model.to(args.device)

    best_f1 = 0
    for epoch, _ in enumerate(range(int(args.num_train_epochs))):
        model.train()
        train(args, train_iter, model)
        eval_f1, eval_loss = evaluate(args, eval_iter, model)
        logger.info(
            "The F1-score is {}".format(eval_f1)
        )
        if eval_f1 > best_f1:
            early_stop = 0
            best_f1 = eval_f1
            logger.info("the best eval f1 is {:.4f}, saving model !!".format(best_f1))
            best_model = copy.deepcopy(model.module if hasattr(model, "module") else model)
            torch.save(best_model.state_dict(), args.output_model_path)
        else:
            early_stop += 1
            if early_stop == args.early_stop:
                logger.info("Early stop in {} epoch!".format(epoch))
                break

if __name__ == '__main__':
    main()
