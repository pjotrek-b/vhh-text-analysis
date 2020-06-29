#
# merge ocr json output to a single .tsv file
# -------------------------------
#

import random
import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())
from collections import defaultdict
import numpy
import glob
import json
#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--files', action='store', dest='in_files', required=True)
parser.add_argument('--out', action='store', dest='out', required=True)

args = parser.parse_args()

stats = defaultdict(int)

all_files = glob.glob(args.in_files)
print("Found:",len(all_files))

with open(args.out,"w",encoding="utf8") as out:

    for f in tqdm(all_files):

        with open(f,"r",encoding="utf8") as in_file:
            #try:
                line = json.load(in_file) # in_file.read().split("\t")

                #text = line["ocrText"]
                text = line["ocrTextWithCorrections"]
                text = text.replace("\t"," ").replace("\n"," ").replace("\r"," ").replace("  "," ").strip()

                out.write(os.path.basename(f)[:-5]+"\t"+text+"\n")
                    
            #except BaseException as e:
            #    print("Error",e)