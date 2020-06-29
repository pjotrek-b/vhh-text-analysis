#
# filter out non-english and too short documents for analysis
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
from langdetect import detect

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--files', action='store', dest='in_files', required=True)
parser.add_argument('--out', action='store', dest='out', required=True)

args = parser.parse_args()

stats = defaultdict(int)
text_lengths=[]
langs=[]


with open(args.out,"w",encoding="utf8") as out:
        with open(args.in_files,"r",encoding="utf8") as in_file:

            for l in tqdm(in_file):

                line = l.split("\t")

                text = line[1]
                text_words = len(text.split())
                text_lengths.append(min(4_000,text_words))

                if text_words > 50:
                    lang = detect(text)
                    langs.append(lang)

                    if lang == "en":
                        stats["good"]+=1

                        out.write(line[0]+"\t"+text+"\n")
                    else:
                        stats["wrong-lang"]+=1
                else:
                    stats["too-short"]+=1
            

def crappyhist(a, bins=20, width=30,range_=(0,1)):
    h, b = numpy.histogram(a, bins,range_)

    for i in range (0, bins):
        print('{:12.5f}  | {:{width}s} {}'.format(
            b[i], 
            '#'*int(width*h[i]/numpy.amax(h)), 
            h[i],#/len(a), 
            width=width))
    print('{:12.5f}  |'.format(b[bins]))

print("Text")
crappyhist(text_lengths,range_=(0,4_000))

from collections import Counter
print(Counter(langs))

for key, val in stats.items():
    print(f"{key}\t{val}")