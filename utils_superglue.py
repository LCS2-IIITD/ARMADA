from __future__ import absolute_import, division, print_function

import csv
import sys
import os
import logging
import numpy as np

os.environ["OPENCV_LOG_LEVEL"]="SILENT"

import cv2

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, 
                 text_d=None,text_e=None,text_f=None,text_g=None,
                 label=None,image=None):
        
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.text_d = text_d
        self.text_e = text_e
        self.text_f = text_f
        self.text_g = text_g
        self.label = label
        self.image = image

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid, image):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        self.image = image

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
        


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")))

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            try:
                image = np.load(os.path.join("images","RTE", set_type,"{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b,text_c=None,
                             text_d=None,text_e=None,text_f=None,
                             text_g=None,label=label,image=image))
        return examples

    def _create_test_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s" % (line[0])
            text_a = line[1]
            text_b = line[2]
            try:
                image = np.load(os.path.join("images","RTE", "test","{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b,text_c=None,
                             text_d=None,text_e=None,text_f=None,
                             text_g=None,label=None,image=image))
        return examples


class BoolqProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")))

    def get_labels(self):
        """See base class."""
        return ["False", "True"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            try:
                image = np.load(os.path.join("images","BOOLQ", set_type,"{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b,text_c=None,
                             text_d=None,text_e=None,text_f=None,
                             text_g=None,label=label,image=image))
        return examples

    def _create_test_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s" % (line[0])
            text_a = line[1]
            text_b = line[2]
            try:
                image = np.load(os.path.join("images","BOOLQ", "test","{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b,text_c=None,
                             text_d=None,text_e=None,text_f=None,
                             text_g=None,label=None,image=image))
        return examples


class CbProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")))
    

    def get_labels(self):
        """See base class."""
        return ["entailment","contradiction","neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            try:
                image = np.load(os.path.join("images","CB", set_type,"{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b,text_c=None,
                             text_d=None,text_e=None,text_f=None,
                             text_g=None,label=label,image=image))
        return examples

    def _create_test_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s" % (line[0])
            text_a = line[1]
            text_b = line[2]
            try:
                image = np.load(os.path.join("images","CB", "test","{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b,text_c=None,
                             text_d=None,text_e=None,text_f=None,
                             text_g=None,label=None,image=image))
        return examples


class CopaProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")))


    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            text_c = line[3]
            text_d = line[4]
            label = line[-1]
            try:
                image = np.load(os.path.join("images","COPA", set_type,"{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b,text_c=text_c,
                             text_d=text_d,text_e=None,text_f=None,
                             text_g=None,label=label,image=image))
        return examples

    def _create_test_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s" % (line[0])
            text_a = line[1]
            text_b = line[2]
            text_c = line[3]
            text_d = line[4]
            try:
                image = np.load(os.path.join("images","COPA", "test","{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b,text_c=text_c,
                             text_d=text_d,text_e=None,text_f=None,
                             text_g=None,label=None,image=image))
        return examples


class WicProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")))

    def get_labels(self):
        """See base class."""
        return ["False","True"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            text_c = line[3]
            text_d = line[4]
            text_e = line[5]
            text_f = line[6]
            text_g = line[7]

            label = line[-1]

            try:
                image = np.load(os.path.join("images","WIC", set_type,"{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))

            examples.append(
                InputExample(guid=guid,text_a=text_a,
                             text_b=text_b,text_c=text_c,
                             text_d=text_d,text_e=text_e,text_f=text_f,
                             text_g=text_g,label=label,image=image))
        return examples

    def _create_test_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s" % (line[0])
            text_a = line[1]
            text_b = line[2]
            text_c = line[3]
            text_d = line[4]
            text_e = line[5]
            text_f = line[6]
            text_g = line[7]

            try:
                image = np.load(os.path.join("images","WIC", "test","{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))

            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b,text_c=text_c,
                             text_d=text_d,text_e=text_e,text_f=text_f,
                             text_g=text_g,label=None,image=image))
        return examples


class WscProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")))


    def get_labels(self):
        """See base class."""
        return ["false","true"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])

            if(len(line)<6):
                continue

            text_a = line[1]
            text_b = line[2]
            text_c = line[3]
            text_d = line[4]
            text_e = line[5]

            label = line[-1]

            try:
                image = np.load(os.path.join("images","WSC", set_type,"{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))

            examples.append(
                InputExample(guid=guid,text_a=text_a,
                             text_b=text_b,text_c=text_c,
                             text_d=text_d,text_e=text_e,text_f=None,
                             text_g=None,label=label,image=image))
        return examples

    def _create_test_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s" % (line[0])

            if(len(line)<6):
                continue
            
            text_a = line[1]
            text_b = line[2]
            text_c = line[3]
            text_d = line[4]
            text_e = line[5]

            try:
                image = np.load(os.path.join("images","WSC", "test","{}.npy".format(i)))
            except:
                print (i)
                image = np.zeros((1,1024))

            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b,text_c=text_c,
                             text_d=text_d,text_e=text_e,text_f=None,
                             text_g=None,label=None,image=image))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):


    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a.lower())

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b.lower())

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            if example.label:
                label_id = label_map[example.label]
            else:
                label_id = None

        elif output_mode == "regression":
            if example.label:
                label_id = float(example.label)
            else:
                label_id = None
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if label_id:
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              guid = example.guid, image=example.image))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()
def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }
def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in ["boolq", "copa", "rte", "wic", "wsc"]:
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cb":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)
    
processors = {
    "rte": RteProcessor,
    "boolq":BoolqProcessor,
    "cb":CbProcessor,
    "copa":CopaProcessor,
    "wic":WicProcessor,
    "wsc":WscProcessor,
}

output_modes = {
    "rte": "classification",
    "boolq": "classification",
    "cb":"classification",
    "copa":"classification",
    "wic":"classification",
    "wsc":"classification",
}
SUPERGLUE_TASKS_NUM_LABELS = {
    "rte": 2,
    "boolq":2,
    "cb":3,
    "copa":2,
    "wic":3,
    "wsc":2,
}
