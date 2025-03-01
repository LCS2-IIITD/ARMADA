from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
import os
import random
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler

from copy import deepcopy as cp
from utils_superglue import (convert_examples_to_features,
                        output_modes, processors)

from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import cv2
import pickle

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

all_task_names = ['BOOLQ','CB','COPA','RTE','WIC','WSC']

class SimpleDataset(Dataset):
    def __init__(self, x1, x2, x3, x4, x5):
        self.__iter = None
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, key):
        val = self.x1[key], self.x2[key], self.x3[key], self.x4[key], self.x5[key]
        return val
    
class SimpleDataset2(Dataset):
    def __init__(self, x1, x2, x3, x4):
        self.__iter = None
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, key):
        val = self.x1[key], self.x2[key], self.x3[key], self.x4[key]
        return val
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_and_cache_examples(args, task, tokenizer, image_pipe, image_pipe2, evaluate=False, test=False):
    processor = processors[task.lower()]()
    output_mode = output_modes[task.lower()]
    if task.lower() == 'mnli-mm':
        task = 'MNLI'
        data_dir = os.path.join(args.data_dir,task)
    else:
        data_dir = os.path.join(args.data_dir,task)

    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    if evaluate:
        if test:
            examples = processor.get_test_examples(data_dir)
            folder = "images/{}/test".format(task)
        else:
            examples = processor.get_dev_examples(data_dir)
            folder = "images/{}/dev".format(task)
    else:
        examples = processor.get_train_examples(data_dir)
        folder = "images/{}/train".format(task)

    all_images = []
    all_embs = []

    for i, example in tqdm(enumerate(examples)):
        text = example.text_a.lower()
        if example.text_b:
            text += " " + example.text_b.lower()

        if example.text_c:
            text += " " + example.text_c.lower()

        if example.text_d:
            text += " " + example.text_d.lower()

        if example.text_e:
            text += " " + example.text_e.lower()

        if example.text_f:
            text += " " + example.text_f.lower()
        
        if example.text_g:
            text += " " + example.text_g.lower()



        image = image_pipe(text, num_inference_steps=20).images[0]
        image.save(os.path.join(folder, str(i+1)+".png"))
        im = cv2.imread(os.path.join(folder, str(i+1)+".png"))
        im = cv2.resize(im, (128,128))
        emb = image_pipe2(prompt=text, image=im, strength=0.75, guidance_scale=7.5, \
                    num_inference_steps=20, output_type="latent").images[0]
        emb = emb.detach().cpu().numpy().reshape(1,-1)
        im = cv2.imread(os.path.join(folder, str(i+1)+".png"))
        np.save(os.path.join(folder, str(i+1)+".npy"), emb)
        all_embs.append(emb)
        all_images.append(im[np.newaxis,:,:,:])
    
    all_embs = np.concatenate(all_embs, 0)
    print (all_embs.shape)



    features = convert_examples_to_features(examples, label_list, args.max_seq_length,
                                            tokenizer, output_mode,
                                            cls_token_at_end=False,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=1,
                                            pad_on_left=False,
                                            pad_token_segment_id=0)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_images = torch.tensor([im for im in all_images], dtype=torch.float)

    if test == False:
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = SimpleDataset(all_input_ids, all_input_mask, all_segment_ids, all_images, all_label_ids)
    else:
        dataset = SimpleDataset2(all_input_ids, all_input_mask, all_segment_ids, all_images)

    return dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default='CB', type=str,
                        help="Model type selected in the list: " + ", ".join(all_task_names))
    
    parser.add_argument("--data_dir", default='./superglue_data/', type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--teacher_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)

    args.n_gpu = 1

    set_seed(args)

    model_id = "CompVis/stable-diffusion-v1-4"
    image_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    image_pipe = image_pipe.to("cuda")

    image_pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32) 
    image_pipe2 = image_pipe2.to("cuda")

    task_name = args.task

    try:
        os.makedirs("images/{}/train/".format(task_name))
        os.makedirs("images/{}/dev/".format(task_name))
        os.makedirs("images/{}/test/".format(task_name))
    except:
        pass

    processor = processors[task_name.lower()]()
    output_mode = output_modes[task_name.lower()]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    print ("Loaded %s dataset" % (task_name))
    
    train_dataset = load_and_cache_examples(args, task_name, tokenizer, image_pipe, image_pipe2, evaluate=False)
    eval_dataset = load_and_cache_examples(args, task_name, tokenizer, image_pipe, image_pipe2, evaluate=True)
    test_dataset = load_and_cache_examples(args, task_name, tokenizer, image_pipe, image_pipe2, evaluate=True, test=True)
    
if __name__ == "__main__":
    main()
