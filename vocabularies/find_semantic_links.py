import argparse
import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict
import statistics

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import csv
from tqdm import tqdm

#
# data
#

data_orig = {
        "In-Domain Only ": "C:\\Users\\sebas\\OneDrive\\University\\VHH\\en_longerthan50w_text_2020_06_300dim_synonyms.tsv",
        "Wikipedia + In-Domain ": "C:\\Users\\sebas\\OneDrive\\University\\VHH\\en_longerthan50w_text_2020_06_and_wikibase_300dim_synonyms.tsv",
}

data_path_docs = "C:\\Users\\sebas\\data\\vhh\\vocabularies\\citavi_sebastian_extract_2020_06.tsv"
out_path_prefix = "C:\\Users\\sebas\\data\\vhh\\vocabularies\\links_citavi_controlled_vocab_threshold_"

#
# work
#
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({'font.size': 16})
plt.rc('legend',**{'fontsize':14})
mpl.rcParams['axes.linewidth'] = 2 
mpl.rcParams['svg.fonttype'] = "none" 
mpl.rcParams['grid.linewidth'] = 1.5

mpl.rcParams['lines.linewidth'] = 2.5

#fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.8,7.8))
ax = fig.add_subplot(1, 1, 1)

colors = [
    #("dimgray","black"),
    ("indianred","firebrick"),
    ("sandybrown","darkorange"),
    ("steelblue","darkslateblue"),
    ("mediumseagreen","seagreen"),
    ("mediumorchid","purple"),
    ("darkturquoise","darkcyan"),
    ("yellow","yellow"),
    ("pink","pink")]

markers = [
    "o",
    "v",
    "^",
    "s",
    "p",
]

max_ms = 300
min_ms = 1
x_ticks = [1,50,100,150,200,250,300]

x_tick_labels = [str(x) for x in x_ticks]
x_range = list(range(min_ms,max_ms))

ax.set_clip_on(False)

min_value=1
max_value=0

links_per_label = {}
thresholds = np.arange(0.7,1.01,0.02)
thresholds[-1] = 1
for i,(label,(data_path)) in enumerate(data_orig.items()):

    with open(data_path,"r",encoding="utf8") as file:
        words = {}
        for line in tqdm(file):
            l = line.split("\t")
            words[l[1]] = 1.0
            for other in l[3:]:
                w,dist = other.split()
                if w in words:
                    dist = max(words[w],float(dist))
                words[w] = float(dist)
        print("sim: words",len(words))
    word_counts = defaultdict(int)
    word_link_to_doc = defaultdict(list)
    with open(data_path_docs,"r",encoding="utf8") as file:
        for i,line in tqdm(enumerate(file)):
            doc_id,text = line.split("\t")
            doc_words = text.split()
            for w in set(doc_words):
                if w in words:
                    word_counts[w]+=1
                    word_link_to_doc[w].append(doc_id)
            if i == 1000:
                break

    count_per_threshold = np.ndarray(len(thresholds))

    for t,th in enumerate(thresholds):
        with open(out_path_prefix+"{:.2f}".format(th)+"_"+label+".tsv","w",encoding="utf8") as file:
            for w,c in word_counts.items():
                if words[w] >= th:
                    count_per_threshold[t] += 1
                    file.write(w+"\t"+",".join("\""+a+"\"" for a in set(word_link_to_doc[w]))+"\n")

    links_per_label[label] = count_per_threshold


for i,(label,(data)) in enumerate(links_per_label.items()):
    ax.plot(thresholds, data,label=label,color=colors[i][1])#,zorder=zorder[label]) # 


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xticks(thresholds)
ax.set_xticklabels(["{:.2f}".format(t) for t in thresholds])
ax.xaxis.set_tick_params(width=2)
ax.yaxis.set_tick_params(width=2)

ax.set_ylim(bottom=500,top=1800)
ax.yaxis.grid(linestyle=":",zorder=0)

ax.set_ylabel("Number of Document Links")
ax.set_xlabel("Semantic Similarity",labelpad=7)

ax.margins(0.01)

plt.legend(loc= 'upper right' ,framealpha=1,labelspacing=0.1)
fig.tight_layout()

plt.show() 