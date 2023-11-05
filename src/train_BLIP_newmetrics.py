# @title Image Encoder

from transformers import BertTokenizer
from transformers import AutoProcessor, BlipForImageTextRetrieval
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
import warnings
import logging
import time
from datetime import timedelta
from sklearn.metrics import classification_report

import functools
from collections import Counter
import argparse
import torchvision.transforms as transforms
from transformers.optimization import AdamW
from torch.utils.data import DataLoader

from PIL import Image

from torch.utils.data import Dataset

import contextlib
import random
import shutil
import os

import torch
import json
import numpy as np
import jsonlines


class MultiModalBlipClf(nn.Module):
    def __init__(self, args):
        super(MultiModalBlipClf, self).__init__()
        # pass
        self.l1 = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        self.n_classes = args.n_classes  # number of classes for my dataset
        self.vit_embd_dim = args.hidden_sz
        self.pre_classifier = torch.nn.Linear(self.vit_embd_dim, 256)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(256, self.n_classes)

    def forward(self, input_ids, pixel_values):
        output_ = self.l1(input_ids=input_ids, pixel_values=pixel_values)
        # print('output_', output_)
        hidden_state = output_[1]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


# @title criterion
def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.CrossEntropyLoss(pos_weight=label_weights.cuda()) # new
            #criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
            # criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cpu())
        else:
            #criterion = nn.BCEWithLogitsLoss() #old
            criterion = nn.CrossEntropyLoss(pos_weight=label_weights.cuda()) #new
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

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    # ------------ In Transformers, optimizer and schedules are split and instantiated like this--------
    # optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup,
                                                num_training_steps=total_steps)  # PyTorch scheduler

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
            "{}: Loss: {:.5f} | Acc: {:.5f}  | Macro_rec: {:.5f} | Micro_rec: {:.5f} | Macro_prec: {:.5f} | Micro_prec: {:.5f} | Class_rep:".format(
                set_name, metrics["loss"], metrics["acc"], metrics["macro_precision"], metrics["micro_precision"], metrics["macro_recall"], metrics["micro_recall"],
            metrics['class_rep'] #new
            #"{}: Loss: {:.5f} | Macro F1 {:.5f} | Micro F1: {:.5f}".format(
             #   set_name, metrics["loss"], metrics["macro_f1"], metrics["micro_f1"]
            )
        )
   # else: #old
    #    logger.info(
     #       "{}: Loss: {:.5f} | Acc: {:.5f}".format(
      #          set_name, metrics["loss"], metrics["acc"]
      #      )
      #  )
    else: #new
        logger.info(
            "{}: Loss: {:.5f} | Acc: {:.5f}  | Macro_rec: {:.5f} | Micro_rec: {:.5f} | Macro_prec: {:.5f} | Micro_prec: {:.5f} | Class_rep:".format(
                set_name, metrics["loss"], metrics["acc"], metrics["macro_precision"], metrics["micro_precision"], metrics["macro_recall"], metrics["micro_recall"],
            metrics['class_rep']  
            ) # new
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


class MultiModalMarketingDataset(Dataset):
    def __init__(self, path, args):
        """
        path : path to the jsonl file input
        mode : accepts only two mode - 'train' , 'test' or 'val'
        """
        #self.root_dir = r'C:\Users\tpadhi1\Desktop\Adobe\mmbt-kg\data_prep_codes'#old
        self.root_dir = '' #new
        self.jsonl_file_input = path
        self.img_paths, self.texts, self.tgts = self.get_data_jsonl()
        self.mode = 'train'
        # define transforms
        self.transform = {
            'train': transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(50),
                                         transforms.ToTensor()]),

            'valid': transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()])
        }
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

    def __getitem__(self, idx):
        text = self.texts[idx]
        target_tensor = torch.tensor(self.tgts[idx])
        try:
            img_path = os.path.join(self.root_dir, self.img_paths[idx]).replace('\\', '/')
            # print('img_path', img_path)
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # print('cant load image')
            img = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        if self.mode.lower() == 'train':
            img = self.transform['train'](img)
        else:
            img = self.transform['valid'](img)
        encodings = self.processor(images=img, text=text, padding="max_length", return_tensors="pt")
        encodings['labels'] = target_tensor
        encodings = {k: v.squeeze() for k, v in encodings.items()}

        return encodings

    def __len__(self):
        return len(self.img_paths)

        # to read the jsonl format

    def get_data_jsonl(self):
        num_lines = sum(1 for line in jsonlines.open(self.jsonl_file_input, 'r'))
        img_paths = []
        texts = []
        tgts = []
        with jsonlines.open(self.jsonl_file_input) as f:
            for line in tqdm(f, total=num_lines):
                img_paths.append(line['img'])
                texts.append(line['text'])
                tgts.append(line['label'])
        return img_paths, texts, tgts


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




def get_data_loaders(args):

    args.labels, args.label_freqs = get_labels_and_frequencies(os.path.join(args.data_path, "train.jsonl"))
    args.n_classes = len(args.labels)

    train = MultiModalMarketingDataset(os.path.join(args.data_path, "train.jsonl"), args)
    train_loader = DataLoader(train, batch_size=args.batch_sz, shuffle=True, num_workers=args.n_workers, )

    args.train_data_len = len(train)
    dev = MultiModalMarketingDataset(os.path.join(args.data_path, "dev_seen.jsonl"), args)
    val_loader = DataLoader(dev,batch_size=args.batch_sz,shuffle=False,num_workers=args.n_workers    )

    test_set = MultiModalMarketingDataset(os.path.join(args.data_path, "test_seen.jsonl"), args)
    test_gt = MultiModalMarketingDataset(os.path.join(args.data_path, "test_unseen.jsonl"), args)

    test_loader = DataLoader(test_set,batch_size=args.batch_sz,shuffle=False,num_workers=args.n_workers)
    test_gt_loader = DataLoader(test_gt,batch_size=args.batch_sz,shuffle=False,num_workers=args.n_workers)

    test = {"test": test_loader,"test_gt": test_gt_loader}

    return train_loader, val_loader, test


# @title model forward during training

def model_forward(model, criterion, batch, device='cuda'):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)
    tgt = batch.pop("labels").to(device)
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    loss = criterion(outputs.cuda(), tgt)
    return loss, outputs, tgt


# @title model_eval()

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def model_eval(data, model, args, criterion, store_preds=True): # new, store_preds=False
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            loss, out, tgt = model_forward(model, criterion, batch)
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
        metrics["macro_recall"] = precision_score(tgts, preds, average='macro') # new
        metrics["macro_precision"] = recall_score(tgts, preds, average='macro') # new
        metrics["micro_recall"] = precision_score(tgts, preds, average='micro') # new
        metrics["micro_precision"] = recall_score(tgts, preds, average='micro') # new
        metrics["class_rep"] = classification_report(tgts, preds) # new

        
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)
        metrics["macro_recall"] = precision_score(tgts, preds, average='macro') # new
        metrics["macro_precision"] = recall_score(tgts, preds, average='macro') # new
        metrics["micro_recall"] = precision_score(tgts, preds, average='micro') # new
        metrics["micro_precision"] = recall_score(tgts, preds, average='micro') # new
        metrics["micro_precision"] = recall_score(tgts, preds, average='micro') # new
        metrics["class_rep"] = classification_report(tgts, preds, average='micro') # new

    #classification_report(y_true, , target_names=target_names)

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics


def get_model(args):
    # return MultimodalBertClf(args)
    return MultiModalBlipClf(args)


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

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, _, _ = model_forward(model, criterion, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics = model_eval(val_loader, model, args, criterion, store_preds=True)
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
    #parser.add_argument("--model", type=str, default="mmbt",
     #                   choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt"])
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
# python train_BLIP.py --batch_sz 16  --gradient_accumulation_steps 40 --savedir test --name mmbt_model_run --data_path C:\Users\tpadhi1\Desktop\Adobe\mmbt-kg\data_prep_codes\Experiments\Transe --model mmbt --max_epochs 50
