from datasets import load_dataset, load_metric
from datasets import ClassLabel, Audio
import random
import pandas as pd
import numpy as np
import re, time, math
import json, librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, HubertForCTC, Wav2Vec2ConformerForCTC
from transformers import Wav2Vec2CTCTokenizer, AutoTokenizer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import AutoModelForCTC
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM
from transformers import EarlyStoppingCallback, IntervalStrategy
import jiwer, os, sys
import time
from tqdm import tqdm
from torch.utils.data import Dataset
import argparse
import functools

def str_none(val):
    if val == 'None':
        return None
    else:
        return val

def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))

def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in vars(args).items():
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")

def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = strtobool if type == bool else type
    type = str_none if type == str else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' Default: %(default)s.',
                           **kwargs)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="/valleblob/v-shujiehu/Dbank/data-fully-aug/train/age_spk_together.txt.csv",       help="训练数据集的路径")
add_arg("test_data",     type=str, default="/valleblob/v-shujiehu/Dbank/data-fully-aug/evde/age_spk_together.txt.csv",        help="测试数据集的路径")
add_arg("output_dir",    type=str, default="output/",                  help="训练保存模型的路径")
add_arg("warmup_steps",  type=int, default=1000,      help="训练预热步数")
add_arg("logging_steps", type=int, default=3000,     help="打印日志步数")
add_arg("eval_steps",    type=int, default=3,    help="多少步数评估一次")
add_arg("save_steps",    type=int, default=3000,    help="多少步数保存模型一次")
add_arg("eval_delay",    type=int, default=9,    help="延迟多少步数开始评估")
add_arg("num_workers",   type=int, default=20,       help="读取数据的线程数量")
add_arg("learning_rate", type=float, default=3e-5,  help="学习率大小")
add_arg("num_train_epochs", type=int, default=30,      help="训练的轮数")
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复训练的检查点路径")
add_arg("per_device_train_batch_size", type=int, default=8,    help="训练的batch size")
add_arg("per_device_eval_batch_size",  type=int, default=4,    help="评估的batch size")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="梯度累积步数")
add_arg("use_lhuc", type=bool, default=False, help="是否使用LHUC")
add_arg("spk_num", type=int, default=688, help="有多少个说话人")
add_arg("spk_layer", type=bool, default=False, help="是否使用adapter")
add_arg("spk_intermediate_size", type=int, default=128, help="spk adapter的瓶颈层维度")

add_arg("seve_group", type=bool, default=False, help="是否使用Seve LHUC")
add_arg("seve_num", type=int, default=5, help="seve个数")
add_arg("seve_layer", type=bool, default=False, help="是否使用adapter")
add_arg("seve_intermediate_size", type=int, default=128, help="seve adapter的瓶颈层维度")

add_arg("add_lhuc_adapter_pos", type=int, default=-1, help="adapter的具体位置")

args = parser.parse_args()
print_arguments(args)

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).upper() + " "
    return batch

def prepare_dataset(batch):
    audio = batch["file"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["attention_mask"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).attention_mask[0]
    batch["input_length"] = len(batch["input_values"])
    batch["spk_id"] = batch["sid"]
    batch["seve_id"] = batch["aid"]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 mono=True,
                 sample_rate=16000):
        """
        Args:
            data_list_path: 数据列表文件的路径，或者二进制列表的头文件路径
            processor: Whisper的预处理工具，WhisperProcessor.from_pretrained获取
            mono: 是否将音频转换成单通道，这个必须是True
            sample_rate: 音频的采样率，默认是16000
        """
        super(CustomDataset, self).__init__()
        self.data_list_path = data_list_path
        self.processor = processor
        self.sample_rate = sample_rate
        self.mono = mono
        self.vocab = self.processor.tokenizer.get_vocab()
        self.data_list: List[dict] = []
        # 加载数据列表
        self._load_data_list()

    # 加载数据列表
    def _load_data_list(self):
        if self.data_list_path.endswith(".header"):
            # 获取二进制的数据列表
            self.dataset_reader = DatasetReader(data_header_path=self.data_list_path,
                                                min_duration=self.min_duration,
                                                max_duration=self.max_duration)
            self.data_list = self.dataset_reader.get_keys()
        else:
            # 获取数据列表
            with open(self.data_list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.data_list = []
            for line in tqdm(lines, desc='读取数据列表'):
                iterms = line.split(",")
                if iterms[0] == "id":
                    continue
                else:
                    self.data_list.append(line)
    
    # 从数据列表里面获取音频数据、采样率和文本
    def _get_list_data(self, idx):
        data_list = self.data_list[idx]
        # 分割音频路径和标签
        iterms = data_list.rstrip("\n").split(",")
        audio_file = iterms[1]
        transcript = iterms[2]
        transcript = re.sub(chars_to_ignore_regex, '', transcript).upper() + " "
        spk_id = iterms[3]
        seve_id = iterms[4]
        sample, sample_rate = sf.read(audio_file, dtype='float32')
        sample = sample.T
        # 转成单通道
        if self.mono:
            sample = librosa.to_mono(sample)
        return sample, sample_rate, transcript, spk_id, seve_id

    def __getitem__(self, idx):
        try:
            # 从数据列表里面获取音频数据、采样率和文本
            sample, sample_rate, transcript, spk_id, seve_id = self._get_list_data(idx=idx)
            data = self.processor(audio=sample, sampling_rate=self.sample_rate, text=transcript)
            data["input_length"] = len(data["input_values"][0])
            data["spk_id"] = int(spk_id)
            data["seve_id"] = int(seve_id)
            return data
        except Exception as e:
            print(f'读取数据出错，序号：{idx}，错误信息：{e}', file=sys.stderr)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_list)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        speaker_features = [feature["spk_id"] for feature in features]
        seve_features = [feature["seve_id"] for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        # batch["spk_id"] = torch.tensor(speaker_features)
        # batch["seve_id"] = torch.tensor(seve_features)

        return batch

def map_to_result(batch):
    model.eval()
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        attention_mask = torch.tensor(batch["attention_mask"], device="cuda").unsqueeze(0)
        spk_id =  [batch["spk_id"]]
        seve_id = [batch["seve_id"]]
        inputs = {"input_values": input_values, "spk_id": spk_id}
        # logits = model(**inputs, seve_id=seve_id, attention_mask=attention_mask).logits
        logits = model(input_values, attention_mask=attention_mask).logits

    model.train()
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    # label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    results = ua["evde"].map(map_to_result, remove_columns=ua["evde"].column_names)
    measures = jiwer.compute_measures(results["text"], results["pred_str"])
    return {"wer": measures["wer"]}

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


ua = load_dataset('csv', data_files={'evde':args.test_data}, cache_dir="large_cache_6")# .shuffle(seed=42)
ua = ua.map(remove_special_characters)
ua = ua.cast_column("file", Audio(sampling_rate=16000))

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rel-pos-large-960h-ft")

ua = ua.map(prepare_dataset, remove_columns=ua["evde"].column_names, num_proc=4)
'''
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
alphabet = list(sorted_vocab_dict.keys())
alphabet = [x if len(x) != 1 else x.upper() for x in alphabet]

decoder = build_ctcdecoder(labels=alphabet,kenlm_model_path="2gram_test.arpa")

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)
'''
# processor_with_lm.save_pretrained("processor_with_lm")

train_dataset = CustomDataset(data_list_path=args.train_data,
                                  processor=processor)
test_dataset = CustomDataset(data_list_path=args.test_data,
                                 processor=processor)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer", cache_dir="./models_large/wer")
cer_metric = load_metric("cer", cache_dir="./models_large/cer")

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

model_path = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"
model = Wav2Vec2ConformerForCTC.from_pretrained(
    model_path,
    cache_dir="./models_large/model",
    # spk_num=args.spk_num,
    # seve_num=args.seve_num,
    # add_lhuc_adapter_pos=args.add_lhuc_adapter_pos,
    # seve_group=args.seve_group,
    # use_lhuc=args.use_lhuc,
    # spk_layer=args.spk_layer,
    # spk_intermediate_size=args.spk_intermediate_size,
    # test_adapt=False,
    # skip_adapt=True,
    ignore_mismatched_sizes=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)
# feature_extractor -> feature_projection -> encoder -> lm_head

model.config.ctc_zero_infinity = True
model.freeze_feature_encoder()

training_args = TrainingArguments(
    output_dir=args.output_dir,
    # output_dir="tmp",
    # group_by_length=True,
    per_device_train_batch_size=args.per_device_train_batch_size,
    evaluation_strategy="steps",
    num_train_epochs=args.num_train_epochs,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    logging_steps=args.logging_steps,
    eval_delay=args.eval_delay,
    learning_rate=args.learning_rate,
    weight_decay=0.005,
    warmup_steps=args.warmup_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    dataloader_num_workers=20,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,
)
# print(model)
trainer.train()

