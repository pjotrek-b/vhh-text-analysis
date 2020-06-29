
import argparse
import os
import sys
import glob
from tqdm import tqdm
sys.path.append(os.getcwd())

import json
from collections import defaultdict

#
# config
#
parser = argparse.ArgumentParser()


parser.add_argument('--in-file-plain', action='store', dest='in_file_plain',
                    help='plain input file destination', required=True)

parser.add_argument('--start-model', action='store', dest='pretrained_model',
                    help='pre-trained fasttext model', required=True)

parser.add_argument('--ft', action='store', dest='fasttext_model',
                    help='trained fasttext model', required=True)

args = parser.parse_args()

#
# train fasttext
# 

from gensim.models.fasttext import *
from gensim.test.utils import datapath
from gensim.parsing.preprocessing import preprocess_string,strip_punctuation,strip_short,strip_multiple_whitespaces
import gensim

with open(args.in_file_plain,"r",encoding="utf8") as in_file_plain:
    corpus = in_file_plain.read().splitlines()
    clean_corpus = []
    for line in corpus:
        clean_corpus.append(strip_multiple_whitespaces(strip_short(strip_punctuation(line))).split())

def gen():
    for line in clean_corpus:
        yield line

#model = gensim.models.FastText(size=300,workers=50,min_count=3,window=7)
model = gensim.models.FastText.load_fasttext_format(args.pretrained_model)
model.workers = 50

# build the vocabulary
model.build_vocab(sentences=clean_corpus,update=True)

# train the model
model.train(
    sentences=clean_corpus, epochs=100,
    total_examples=len(clean_corpus), total_words=model.corpus_total_words,report_delay=15
)

# saving a model trained via Gensim's fastText implementation

model.save(args.fasttext_model, separately=[])