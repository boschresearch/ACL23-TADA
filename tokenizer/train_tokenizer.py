# Train tokenizer script.
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

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name_or_path', type = str, default = "TODBERT/TOD-BERT-JNT-V1", help="initialize the pretrained tokenizer")
    parser.add_argument('--training_corpus_path', type = str, default = "./DomainCC", help="domain training corpus")
    parser.add_argument('--multiwoz_corpus_path', type = str, default = "./downstream/dialog_datasets/MultiWOZ-2.1", help="domain training corpus from MultiWOZ")
    parser.add_argument('--domain', type = str, default = "", help="domain in use")
    parser.add_argument('--batch_size', type=int, default = 1000, help="size for loading training corpus")
    parser.add_argument('--cache_dir', type=str, default = "./cache", help="cache directory") #"./domain/taxi_en.txt"
    parser.add_argument('--save_tokenizer_name', type=str, default = "taxi-cc", help="file name saved for trained tokenizer")
    parser.add_argument('--use_domaincc_only', action='store_true', help="Only use the resources from DomainCC to train domain-specialized tokenizer")
    parser.add_argument('--use_multiwoz', action='store_true', help="Merge the resources from both DomainCC and Domain-MultiWOZ to train domain-specialized tokenizer")
    parser.add_argument('--use_corpus', action='store_true', help="Use the text corpus")
    parser.add_argument('--use_corpus_x', action='store_true', help="Use the mix corpus")
    parser.add_argument('--shuffle', action='store_true', help="Whether to shuffle the merged dataset")
    parser.add_argument('--num_instances', type=int, default=40000, help="Number of instances for training domain-specialized tokenizer")
    return parser.parse_args()

def get_training_corpus(corpus, bs=1000, num_instances=40000):
    dataset = corpus["train"]
    for start_idx in range(0, num_instances, bs):
        samples = dataset[start_idx : start_idx + bs]
        yield samples["text"]

if __name__ == '__main__':
    args = parse_args()
    tod_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    print("Initialize tokenizer with vocab size: {}".format(len(tod_tokenizer.vocab)))
    
    if args.use_domaincc_only:
        raw_datasets = load_dataset("text", data_files=args.training_corpus_path+"/train/{}_200K_prep.txt".format(args.domain), cache_dir=args.cache_dir + "/datasets/DomainCC")
        training_corpus = get_training_corpus(raw_datasets, bs=args.batch_size, num_instances=args.num_instances)
        new_tokenizer = tod_tokenizer.train_new_from_iterator(training_corpus, len(tod_tokenizer.vocab), new_special_tokens=["[sys]", "[usr]", "[url]"])
        new_tokenizer.save_pretrained(args.cache_dir + "/transformer/tokenizer/" + args.save_tokenizer_name)
        print(raw_datasets)
        print("New tokenizer with vocab size: {}".format(len(new_tokenizer.vocab)))
        print("Difference vocab size: {}".format(len(list(set(tod_tokenizer.vocab) - set(new_tokenizer.vocab)))))
    
    if args.use_multiwoz:
        raw_datasets = load_dataset("text", data_files=args.training_corpus_path+"/train/{}_200K_prep.txt".format(args.domain), cache_dir=args.cache_dir + "/datasets/DomainCC")
        domain_datasets = load_dataset("text", data_files=args.multiwoz_corpus_path+"/train/{}_train_dials_all.txt".format(args.domain), 
                            cache_dir=args.cache_dir+"/datasets/MultiWOZ")
        #df = pd.DataFrame({"text": domain_datasets['train'][0:int(args.num_instances*0.5)]['text'] + raw_datasets['train'][0:int(args.num_instances*0.5)]['text']})
        df = pd.DataFrame({"text": domain_datasets['train'][0:20000]['text'] + raw_datasets['train'][0:20000]['text']})
        domain_dataset = DatasetDict({"train": Dataset.from_pandas(df)})
        if args.shuffle:
            domain_dataset = domain_dataset.shuffle(seed=34)
        training_corpus = get_training_corpus(domain_dataset, bs=args.batch_size, num_instances=len(domain_dataset['train']))
        print(domain_dataset)
        new_tokenizer = tod_tokenizer.train_new_from_iterator(training_corpus, len(tod_tokenizer.vocab), new_special_tokens=["[sys]", "[usr]", "[url]"])
    
    if args.use_corpus:
        raw_datasets = load_dataset("text", data_files="../background/train/{}_4_5K.txt".format(args.domain), cache_dir=args.cache_dir + "/datasets/background")
        training_corpus = get_training_corpus(raw_datasets, bs=args.batch_size, num_instances=args.num_instances)
        new_tokenizer = tod_tokenizer.train_new_from_iterator(training_corpus, len(tod_tokenizer.vocab))
        new_tokenizer.save_pretrained(args.cache_dir + "/transformer/tokenizer/" + args.save_tokenizer_name)
        print(raw_datasets)
        print("New tokenizer with vocab size: {}".format(len(new_tokenizer.vocab)))
        print("Difference vocab size: {}".format(len(list(set(tod_tokenizer.vocab) - set(new_tokenizer.vocab)))))
    
    if args.use_corpus_x:
        raw_datasets = load_dataset("text", data_files="../background/train/{}.txt".format(args.domain), cache_dir=args.cache_dir + "/datasets/background")
        #domain_datasets = load_dataset("text", data_files="./seq-eval/data/nli/mnli_{}_train_sent_all.txt".format(args.domain), cache_dir=args.cache_dir+"/datasets/background")
        domain_datasets = load_dataset("text", data_files="./data/ner/ner_{}_train_sent_all.txt".format(args.domain), cache_dir=args.cache_dir+"/datasets/background")
        #df = pd.DataFrame({"text": domain_datasets['train'][0:int(args.num_instances*0.5)]['text'] + raw_datasets['train'][0:int(args.num_instances*0.5)]['text']})
        df = pd.DataFrame({"text": domain_datasets['train'][0:20000]['text'] + raw_datasets['train'][0:20000]['text']})
        domain_dataset = DatasetDict({"train": Dataset.from_pandas(df)})
        if args.shuffle:
            domain_dataset = domain_dataset.shuffle(seed=34)
        training_corpus = get_training_corpus(domain_dataset, bs=args.batch_size, num_instances=len(domain_dataset['train']))
        print(domain_dataset)
        new_tokenizer = tod_tokenizer.train_new_from_iterator(training_corpus, len(tod_tokenizer.vocab))
        
    ##Financial
    new_tokenizer.save_pretrained(args.cache_dir + "/transformer/tokenizer/" + args.save_tokenizer_name)
    print("New tokenizer with vocab size: {}".format(len(new_tokenizer.vocab)))
    print("Difference vocab size: {}".format(len(list(set(tod_tokenizer.vocab) - set(new_tokenizer.vocab)))))
    print("Difference vocab size: {}".format(len(list(set(new_tokenizer.vocab) - set(tod_tokenizer.vocab)))))
    in_tod_not_in_new= set(tod_tokenizer.vocab) - set(new_tokenizer.vocab)
    in_new_not_in_tod= set(new_tokenizer.vocab) - set(tod_tokenizer.vocab)
    new_tokens = [item for item in in_tod_not_in_new if "unused" not in item and len(item)>3]
    print(len(in_tod_not_in_new))
    print(len(in_new_not_in_tod))
    print(len(new_tokens))
    new_size = len(tod_tokenizer.vocab)-len(new_tokenizer.vocab)
    #special_tokens_dict = {'additional_special_tokens': sorted(new_tokens)[:new_size]}
    #new_tokenizer.add_special_tokens(special_tokens_dict)
    with open('./tokenizer/background-x/financial/vocab_new.txt', 'w') as f:
        for line in sorted(new_tokens)[:new_size]:
            f.write(f"{line}\n")

    new_tokenizer.add_tokens(sorted(new_tokens)[:new_size])

    print("New tokenizer with vocab size: {}".format(len(new_tokenizer.vocab)))
    print("Difference vocab size: {}".format(len(list(set(tod_tokenizer.vocab) - set(new_tokenizer.vocab)))))
    print("Difference vocab size: {}".format(len(list(set(new_tokenizer.vocab) - set(tod_tokenizer.vocab)))))
