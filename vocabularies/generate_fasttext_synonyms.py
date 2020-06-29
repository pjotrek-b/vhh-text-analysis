
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

parser.add_argument('--out-file-plain', action='store', dest='out_file_plain',
                    help='same structure, but with synonyms', required=True)

parser.add_argument('--ft', action='store', dest='fasttext_model',
                    help='trained fasttext model', required=True)

args = parser.parse_args()


from gensim.models.fasttext import FastText as FT_gensim
from gensim.parsing.preprocessing import stem_text


loaded_model = FT_gensim.load(args.fasttext_model)
print(loaded_model)

vocab_flat_global = {}

with open(args.in_file_plain,"r",encoding="utf8") as in_file_plain,\
     open(args.out_file_plain,"w",encoding="utf8") as out_file_plain:
    for row in in_file_plain:
        row = row.strip().split("\t")
        out_file_plain.write(row[0] +"\t"+ row[1]+"\t" +"\t".join(["{} {:1.2f}".format(x,y) for x,y in loaded_model.wv.most_similar(row[1].split(),topn=50)])+"\n")