# Obtain Multiwoz training data.
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type = str, default = "taxi", help = "which domain corpus in use")
    parser.add_argument('--dataset_dir', type = str, default = "../downstream/dialog_datasets/MultiWOZ-2.1", help = "where the dialog_dataset dir is")
    return parser.parse_args()

def load_json(args):
    file_path = "{}/train/{}_train_dials.json".format(args.dataset_dir, args.domain)
    
    with open(file_path) as f:
        multiwoz_texts = json.load(f)
    
    print("File loaded from: {}".format(file_path))
    print("Number of domain dialog utterances: {}".format(len(multiwoz_texts)))
    
    return multiwoz_texts

def store_file(args, texts):
    file_path = "{}/train/{}_train_dials_all.txt".format(args.dataset_dir, args.domain)
    with open(file_path, 'w') as s:
        for element in texts:
            s.write(element + "\n")
    print("Done storing at: {}".format(file_path))

if __name__ == '__main__':
    #python load_multiwoz.py --domain="taxi"
    args = parse_args()
    multiwoz_texts = load_json(args)
    all_texts = []
    for i in range(len(multiwoz_texts)):
        for data in multiwoz_texts[i]['dialogue']:
            if len(data['system_transcript'])!=0:
                all_texts.append(data['system_transcript'])
            if len(data['transcript'])!=0:
                all_texts.append(data['transcript'])
                
    print("Number of loaded utterances: {}".format(len(all_texts)))
    store_file(args, all_texts)