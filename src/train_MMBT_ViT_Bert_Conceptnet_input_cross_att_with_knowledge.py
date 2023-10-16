# @title Image Encoder

import torch
import torch.nn as nn
import torchvision
import torch
import torch.nn as nn
# from pytorch_pretrained_bert.modeling import BertModel
from transformers import AutoConfig, AutoModel, BertConfig, BertModel, BertTokenizer, RobertaConfig, RobertaModel, \
    ViTModel
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
# from pytorch_pretrained_bert import BertAdam
import warnings
import logging
import time
from datetime import timedelta

import functools
import json
import os
from collections import Counter
import argparse

import torch
import torchvision.transforms as transforms
# from pytorch_pretrained_bert import BertTokenizer
from transformers.optimization import AdamW
from torch.utils.data import DataLoader

import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

import contextlib
import numpy as np
import random
import shutil
import os

import torch
# from neo4j import GraphDatabase
import pandas as pd
import json
import numpy as np


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


class VisionTransformerEncoder(nn.Module):
    def __init__(self, args):
        super(VisionTransformerEncoder, self).__init__()
        # pass
        self.l1 = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.n_classes = args.n_classes  # number of classes for my dataset
        self.vit_embd_dim = args.img_hidden_sz
        self.pre_classifier = torch.nn.Linear(self.vit_embd_dim, 256)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(256, self.n_classes)

    def forward(self, data):
        output_ = self.l1(data)
        # print('output_', output_)
        hidden_state = output_[0]
        # pooler = hidden_state[:, 0]
        # pooler = self.pre_classifier(pooler)
        # pooler = torch.nn.ReLU()(pooler)
        # pooler = self.dropout(pooler)
        # output = self.classifier(pooler)
        return hidden_state  # Bx3x224x224 -> BxNx786


# @title Image Classifier


class ImageClf(nn.Module):
    def __init__(self, args):
        super(ImageClf, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoder(args)
        self.clf = nn.Linear(args.img_hidden_sz * args.num_image_embeds, args.n_classes)

    def forward(self, x):
        x = self.img_encoder(x)
        x = torch.flatten(x, start_dim=1)
        out = self.clf(x)
        return out


# @title MMBERT ImageEmbeddings
class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_sz)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        bsz = input_imgs.size(0)
        seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token

        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        # cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cpu()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        # sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cpu()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        imgs_embeddings = self.img_embeddings(input_imgs)
        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1
        )

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        # position_ids = torch.arange(seq_length, dtype=torch.long).cpu()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        # print('position_ids.shape', position_ids.shape)
        position_embeddings = self.position_embeddings(position_ids)
        # print('token_type_ids.shape', token_type_ids.shape)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # print('token_embeddings.shape', token_embeddings.shape)
        # print('position_embeddings.shape', position_embeddings.shape)
        # print('token_type_embeddings.shape', token_type_embeddings.shape)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# @title MM BertEncoder

class MultimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder, self).__init__()
        self.args = args

        # ----------- for BERT Modelling -----------
        # model_config = BertConfig.from_pretrained(args.bert_model, output_hidden_states=True)
        # bert = BertModel.from_pretrained(args.bert_model, config=model_config)

        # # ------------ for Roberta Model ------------
        model_config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
        bert = RobertaModel.from_pretrained("roberta-base", config=model_config)
        model_config.type_vocab_size = 2
        bert.embeddings.token_type_embeddings = torch.nn.Embedding(2, model_config.hidden_size)
        bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
        # ----- -------------------------------------
        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        # self.img_encoder = ImageEncoder(args)
        self.vit_encoder = VisionTransformerEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, input_txt, attention_mask, segment, input_img):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
                # torch.ones(bsz, self.args.num_image_embeds + 2).long().cpu(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # print()
        img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )

        # img_tok = (
        #     torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
        #     .fill_(0)
        #     .cpu()
        # )
        # print('img_tok.shape', img_tok.shape)
        # img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048
        img = self.vit_encoder(input_img)  # BxNx3x224x224 -> BxNx786
        # print('img.shape', img.shape)
        img_embed_out = self.img_embeddings(img, img_tok)
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        # print('img_embed_out.shape', img_embed_out.shape)
        # print('txt_embed_out.shape', txt_embed_out.shape)

        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID

        # print('encoder_input.shape', encoder_input.shape)
        encoded_layers = self.encoder(encoder_input, extended_attention_mask)

        # print('self.pooler(encoded_layers[-1])', self.pooler(encoded_layers[-1]).shape)
        return self.pooler(encoded_layers[-1])


class CocatFusion(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super.__init__()
        self.dense_layer = torch.nn.Linear(input_channels, output_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, list_tensors):
        concat = torch.cat([list_tensors], dim=1)
        x = self.dense_layer(concat)
        x = self.relu(x)
        return x


class PoolFusion(object):
    def __init__(self):
        pass

    def __call__(self, list_tensors, pool_type='mean'):
        concat_dim_zero = torch.cat([list_tensors], dim=1)
        if pool_type == 'mean':
            mean_embedding = torch.mean(concat_dim_zero, 1, True)
        elif pool_type == 'max':
            mean_embedding = torch.max(concat_dim_zero, 1)
        return mean_embedding


class AttentionFusion(object):
    def __init__(self, input_dim, num_heads):
        self.attention_layer = torch.nn.MultiheadAttention(input_dim, num_heads)

    def __call__(self, text_embedding, knowledge_embedding):
        # concat_dim_zero = torch.cat([list_tensors], dim=1)
        attention_fused_tensors = self.attention_layer(text_embedding, knowledge_embedding,knowledge_embedding  need_weights=False)
        return attention_fused_tensors


# @title MultiModal BERT Classifier Layer

class MultimodalBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalBertClf, self).__init__()
        self.args = args
        self.enc = MultimodalBertEncoder(args)
        # self.clf = nn.Linear(args.hidden_sz, args.n_classes)
        self.linearlayer1 = nn.Linear(args.hidden_sz, 256)
        self.act1 = nn.GELU()
        self.linearlayer2 = nn.Linear(512, 256)
        self.act2 = nn.GELU()
        self.att_fusion_layer = AttentionFusion(256,4)
        self.clf = nn.Linear(128, args.n_classes)

    def forward(self, txt, mask, segment, img, concept_net_embeddings):
        x = self.enc(txt, mask, segment, img)
        x1 = concept_net_embeddings
        x = self.linearlayer1(x)
        x = self.act1(x)
        x = self.att_fusion_layer(x,x1)
        # x = self.act4(x)
        return self.clf(x)


# @title criterion
def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
            # criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cpu())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion


# @title Bert Optimizer


def get_optimizer(model, args):
    # if args.model in ["bert", "concatbert", "mmbt"]:
    total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
    )
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0, },
    ]
    # optimizer = BertAdam(
    #     optimizer_grouped_parameters,
    #     lr=args.lr,
    #     warmup=args.warmup,
    #     t_total=total_steps,
    # )
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr
    )
    # ------------ In Transformers, optimizer and schedules are splitted and instantiated like this--------
    # optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup,
                                                num_training_steps=total_steps)  # PyTorch scheduler

    # else:
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer, scheduler


# @title scheduler
def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


# @title loggers


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath, args):
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    logger.info(
        "\n".join(
            "%s: %s" % (k, str(v))
            for k, v in sorted(dict(vars(args)).items(), key=lambda x: x[0])
        )
    )

    return logger


# @title Helper Functions:
# @markdown get_data_loaders()
#


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    filename = os.path.join(checkpoint_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint["state_dict"])


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copied from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def store_preds_to_disk(tgts, preds, args):
    if args.task_type == "multilabel":
        with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
            fw.write(
                "\n".join([" ".join(["1" if x else "0" for x in p]) for p in preds])
            )
        with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
            fw.write(
                "\n".join([" ".join(["1" if x else "0" for x in t]) for t in tgts])
            )
        with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
            fw.write(" ".join([l for l in args.labels]))

    else:
        with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
            fw.write("\n".join([str(x) for x in preds]))
        with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
            fw.write("\n".join([str(x) for x in tgts]))
        with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
            fw.write(" ".join([str(l) for l in args.labels]))


def log_metrics(set_name, metrics, args, logger):
    if args.task_type == "multilabel":
        logger.info(
            "{}: Loss: {:.5f} | Macro F1 {:.5f} | Micro F1: {:.5f}".format(
                set_name, metrics["loss"], metrics["macro_f1"], metrics["micro_f1"]
            )
        )
    else:
        logger.info(
            "{}: Loss: {:.5f} | Acc: {:.5f}".format(
                set_name, metrics["loss"], metrics["acc"]
            )
        )


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path, encoding="utf8")]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        concept_net_embeddings = torch.from_numpy(np.array(self.data[index]["embeddings_2"]))
        concept_net_embeddings = concept_net_embeddings.float()
        if self.args.task == "vsnli":
            sent1 = self.tokenizer(self.data[index]["sentence1"])
            sent2 = self.tokenizer(self.data[index]["sentence2"])
            truncate_seq_pair(sent1, sent2, self.args.max_seq_len - 3)
            sentence = self.text_start_token + sent1 + ["[SEP]"] + sent2 + ["[SEP]"]
            segment = torch.cat(
                [torch.zeros(2 + len(sent1)), torch.ones(len(sent2) + 1)]
            )
        else:
            sentence = (
                    self.text_start_token
                    + self.tokenizer(self.data[index]["text"])[
                      : (self.args.max_seq_len - 1)
                      ]
            )
            segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"])]
            )

        image = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
            try:
                if self.data[index]["img"]:
                    image = Image.open(os.path.join(self.data_dir, self.data[index]["img"])).convert("RGB")
                else:
                    image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
                image = self.transforms(image)
            except:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
                image = self.transforms(image)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            # print('segment', segment)
            sentence = sentence[1:]
            # print('sentence', sentence)
            # The first segment (0) is of images.
            segment += 1

        # print('sentence.shape', sentence.shape)
        # print('segment.shape', segment.shape)
        # print('image.shape', image.shape)
        # print('label', label.shape)
        # print('concept_net_embeddings.shape', concept_net_embeddings.shape)

        return sentence, segment, image, concept_net_embeddings, label


# class Neo4jConnection:
#
#     def __init__(self, uri, user, pwd):
#         self.__uri = uri
#         self.__user = user
#         self.__pwd = pwd
#         self.__driver = None
#         try:
#             self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
#         except Exception as e:
#             print("Failed to create the driver:", e)
#
#     def close(self):
#         if self.__driver is not None:
#             self.__driver.close()
#
#     def query(self, query, db=None):
#         assert self.__driver is not None, "Driver not initialized!"
#         session = None
#         response = None
#         try:
#             session = self.__driver.session(database=db) if db is not None else self.__driver.session()
#             response = list(session.run(query))
#         except Exception as e:
#             print("Query failed:", e)
#         finally:
#             if session is not None:
#                 session.close()
#         return response


# conn = Neo4jConnection(uri="bolt://localhost:7687", user="trilok", pwd="trilok")


# def calculate_embeddings(text: str):
#     """
#
#     :param text:
#     :return:
#     """
#
#     words = text.split(' ')
#     # embedding = []
#     for i in range(len(words) - 1):
#         # print('Calculating embedding for ', words[i])
#         query = f'match (n) where n.id="{words[i].lower()}" return n'
#         response = conn.query(query, db='neo4j')
#         # print(response)
#         try:
#             df = pd.DataFrame.from_records(response[0])
#             if len(df['fastrp-embedding'].to_numpy()[0]) != 0:
#                 # print('embedding array', df['fastrp-embedding'].to_numpy()[0])
#                 if i == 0:
#                     # arr_row = np.append(arr, row, axis=1)
#                     arr = np.array(df['fastrp-embedding'].to_numpy()[0])
#                 else:
#                     # arr = np.append(arr, df['fastrp-embedding'].to_numpy()[0])
#                     arr = np.concatenate((arr, np.array(df['fastrp-embedding'].to_numpy()[0])))
#         except:
#             # print('Cant fetch embedding')
#             arr = np.zeros((32, 1))
#             pass
#     # print('arr.shape', arr.shape)
#     arr = arr.reshape(32, -1)
#     # print('arr.shape', arr.shape)
#     embedding = np.mean(arr, axis=1)
#     # embedding = embedding.reshape(32, 1)
#     # print('embedding.shape', embedding.shape)
#     return embedding


# text = 'Enjoy an unofficial cookbook full of delicious and unique recipes inspired by one of the greatest ' \
#        'entertainers of our generation '
# embedding = calculate_embeddings(text)
# print(embedding.shape)


class Vocab(object):
    def __init__(self, emptyInit=False):
        if emptyInit:
            self.stoi, self.itos, self.vocab_sz = {}, [], 0
        else:
            self.stoi = {
                w: i
                for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
            }
            self.itos = [w for w in self.stoi]
            self.vocab_sz = len(self.itos)

    def add(self, words):
        cnt = len(self.itos)
        for w in words:
            if w in self.stoi:
                continue
            self.stoi[w] = cnt
            self.itos.append(w)
            cnt += 1
        self.vocab_sz = len(self.itos)


def get_transforms(args):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    print('path', path)
    data_labels = [json.loads(line)["label"] for line in open(path, encoding="utf8")]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path, encoding="utf8"):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["bert", "mmbt", "concatbert"]:
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)

    return vocab


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        img_tensor = torch.stack([row[2] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[4] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[4] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    concept_net_embedding_tensor = torch.stack([row[3] for row in batch])
    return text_tensor, segment_tensor, mask_tensor, img_tensor, concept_net_embedding_tensor, tgt_tensor


def get_data_loaders(args):
    tokenizer = (
        BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert"]
        else str.split
    )

    transforms = get_transforms(args)

    args.labels, args.label_freqs = get_labels_and_frequencies(os.path.join(args.data_path, "train.jsonl"))
    # args.labels, args.label_freqs = get_labels_and_frequencies(r'{args.data_path}\\train.jsonl')
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    train = JsonlDataset(
        os.path.join(args.data_path, "train.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    args.train_data_len = len(train)

    # /content/hateful_memes/train.jsonl
    dev = JsonlDataset(
        os.path.join(args.data_path, "dev_seen.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    test_set = JsonlDataset(
        os.path.join(args.data_path, "test_seen.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    test_gt = JsonlDataset(
        os.path.join(args.data_path, "test_unseen.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    test_gt_loader = DataLoader(
        test_gt,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    test = {
        "test": test_loader,
        "test_gt": test_gt_loader,
    }

    return train_loader, val_loader, test


# @title model forward during training

def model_forward(i_epoch, model, args, criterion, batch):
    # print("batch",batch)
    txt, segment, mask, img, concept_net_embeddings, tgt = batch

    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt

    if args.model == "bow":
        txt = txt.cuda()
        out = model(txt)
    elif args.model == "img":
        img = img.cuda()
        out = model(img)
    elif args.model == "concatbow":
        txt, img = txt.cuda(), img.cuda()
        out = model(txt, img)
    elif args.model == "bert":
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        out = model(txt, mask, segment)
    elif args.model == "concatbert":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)
    else:
        assert args.model == "mmbt"
        for param in model.enc.vit_encoder.parameters():
            param.requires_grad = not freeze_img
        for param in model.enc.encoder.parameters():
            param.requires_grad = not freeze_txt

        txt, img, concept_net_embeddings = txt.cuda(), img.cuda(), concept_net_embeddings.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        # txt, img, concept_net_embeddings = txt.cpu(), img.cpu(), concept_net_embeddings.cpu()
        # mask, segment = mask.cpu(), segment.cpu()
        concept_net_embeddings[torch.isinf(concept_net_embeddings)] = 0
        concept_net_embeddings[torch.isnan(concept_net_embeddings)] = 0

        out = model(txt, mask, segment, img, concept_net_embeddings)

    tgt = tgt.cuda()
    # tgt = tgt.cpu()
    loss = criterion(out, tgt)
    return loss, out, tgt


# @title model_eval()

from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics


def get_model(args):
    return MultimodalBertClf(args)


# @title train function

from tqdm import tqdm


def train(args):
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)

    model = get_model(args)
    criterion = get_criterion(args)  # criteria for loss
    optimizer, scheduler = get_optimizer(model, args)  # adam optimizer
    # scheduler = get_scheduler(optimizer, args)  # Scheduler

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)
    model.cuda()
    use_cuda = True
    # use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Training..")

    for i_epoch in tqdm(range(start_epoch, args.max_epochs)):
        train_losses = []
        model.train()
        optimizer.zero_grad()

        # print('len(train_loader',len(train_loader))
        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, _, _ = model_forward(i_epoch, model, args, criterion, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion, store_preds=True
        )
        log_metrics(f"Test - {test_name}", test_metrics, args, logger)


# @title Arguments
# @markdown this is where you change your HyperParametres

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Models")

    parser.add_argument("--batch_sz", type=int, default=8)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="/content/hateful_memes")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=100)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="/path/to/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=20)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="mmbt",
                        choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--n_classes", type=int, default=2)

    parser.add_argument("--name", type=str, default="mmbt")

    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--savedir", type=str, default="/content/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="mmimdb", choices=["mmimdb", "vsnli", "food101"])
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)

    args, remaining_args = parser.parse_known_args()

    warnings.filterwarnings("ignore")
    train(args)

    """
    python mmbt/train.py --batch_sz 4 --gradient_accumulation_steps 40 \
     --savedir/result /path/to/savedir/ --name mmbt_model_run \
     --data_path /path/to/datasets/ \
     --task food101 --task_type classification \
     --model mmbt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3  \
     --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1
    """

"""python train_MMBT_ConceptNet_cuda.py --batch_sz 4 --gradient_accumulation_steps 40 --savedir results_9_6/ --name mmbt_model_run 
--data_path kickstarter_data --model mmbt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --max_epochs 5 """

# for windows machine gpu 6 - I4I
# python train_MMBT_cuda.py --batch_sz 64 --gradient_accumulation_steps 40 --savedir results_mmbt_12_9/ --name mmbt_model_run --data_path C:\Users\tpadhi1\Desktop\Adobe\mmbt-kg\kickstarter_dataset_processed --model mmbt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3 --max_epochs 50

# for vision transformer
# python train_MMBT_ViT_Bert.py --batch_sz 32 --img_hidden_sz 768 --gradient_acuumulation_steps 40 --gradient_accumulation_steps 40 --savedir test --name mmbt_model_run --data_path C:\Users\tpadhi1\Desktop\Adobe\mmbt-kg\data_prep_codes\Experiments\Transe --model mmbt --num_image_embeds 197 --freeze_txt 5 --freeze_img 3 --max_epochs 50
