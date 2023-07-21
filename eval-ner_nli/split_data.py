# Split data for train-test.
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

import random
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type = int, default = 10, help="set random seed")
    parser.add_argument('--train_size', type=int, default = 4500, help="size for training set") 
    parser.add_argument('--test_size', type=int, default = 300, help="size for testing set") 
    parser.add_argument('--input_domain_file', type=str, default = "./data/background/financial_background.txt", help="input domain file path")
    parser.add_argument('--save_train_file_name', type=str, default = "./data/background/train/financial_4_5K.txt", help = "file name for training data")
    parser.add_argument('--save_test_file_name', type=str, default = "./data/background/test/financial_0_3K.txt", help = "file name for testing data")
    return parser.parse_args()

def remove_puncts(text):
    return re.sub(r"\.+", ".", text)
            
def save_file(file_name, corpus_list):
    with open(file_name, 'a') as s:
        for i, element in enumerate(corpus_list):
            element = remove_puncts(element)
            s.write("{}\n".format(element))
            if i%10000==0:
                print(i)
                
if __name__ == '__main__':
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    with open(args.input_domain_file, 'r') as f:
        data = f.read().split('\n')
        data = data[0:500000]
    print("Original data size: {}".format(len(data)))
    random.shuffle(data)
    train, test = train_test_split(data, test_size=0.063, random_state=args.random_seed)
    train = train[0:args.train_size]
    test = test[0:args.test_size]
    print("Training data size: {}".format(len(train)))
    print("Testing data size: {}".format(len(test)))
    #print(test[-1])
    save_file(args.save_test_file_name, test)
    save_file(args.save_train_file_name, train)