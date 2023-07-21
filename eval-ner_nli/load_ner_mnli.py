# Load data script for NER and NLI.
# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import argparse
import json
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type = str, default = "fiction", help = "which domain corpus in use")
    parser.add_argument('--load_mnli', action='store_true', help = "whether to load MNLI")
    parser.add_argument('--load_ner', action='store_true', help = "whether to load NER")
    return parser.parse_args()

def store_mnli(domain, texts):
    file_path = "./data/nli/mnli_{}_train_sent_all.txt".format(domain)
    with open(file_path, 'w') as s:
        for element in texts:
            s.write(element + "\n")
    print("Done storing at: {}".format(file_path))
    
def store_ner(domain, texts):
    file_path = "./data/ner/ner_{}_train_sent_all.txt".format(domain)
    with open(file_path, 'w') as s:
        for element in texts:
            s.write(element + "\n")
    print("Done storing at: {}".format(file_path))
    
if __name__ == '__main__':
    #python load_ner_mnli.py --domain="fiction" --load_mnli
    #python load_ner_mnli.py --domain="fiction" --load_ner
    args = parse_args()
    if args.load_mnli:
        data_files = {}
        data_files["train"] = "./data/nli/mnli_{}_train.json".format(args.domain)
        extension = "./data/nli/mnli_{}_train.json".format(args.domain).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir="./cache/datasets")
        print("File loaded from: {}".format(data_files["train"]))
        print("Number of training samples: {}".format(len(raw_datasets["train"])))

        all_texts = []
        for i, d in enumerate(raw_datasets["train"]):
            all_texts.append(d["sentence1"])
            all_texts.append(d["sentence2"])

        print("Number of loaded texts: {}".format(len(all_texts)))
        store_mnli(args.domain, all_texts)
        
    if args.load_ner:
        data_files = {}
        data_files["train"] = "./data/ner/ner_{}_train.json".format(args.domain)
        extension = "./data/ner/ner_{}_train.json".format(args.domain).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir="./cache/datasets")
        print("File loaded from: {}".format(data_files["train"]))
        print("Number of training samples: {}".format(len(raw_datasets["train"])))

        all_texts = []
        for i, d in enumerate(raw_datasets["train"]):
            all_texts.append(" ".join(d["tokens"]))

        print("Number of loaded texts: {}".format(len(all_texts)))
        store_ner(args.domain, all_texts)