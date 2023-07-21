# Downstream task shell script for ner evaluation.
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

gpu=$1
add1=$2
add2=$3
add3=$4
add4=$5

# BERT
CUDA_VISIBLE_DEVICES=$gpu python run_ner.py \
  --model_name_or_path="bert-base-uncased" \
  --train_file="./data/ner/ner_all_train.json" \
  --validation_file="./data/ner/ner_all_dev.json" \
  --test_file="./data/ner/ner_fiction_test.json" \
  --output_dir="save/ner/fiction/BERT_ALL" \
  --cache_dir="./cache/transformer" \
  --num_train_epochs=10 --save_total_limit 2 \
  --max_seq_length=128 \
  --do_train \
  --do_eval \
  --do_predict \
  --return_entity_level_metrics --overwrite_cache --overwrite_output_dir \
  $add1 $add2 $add3 $add4

# BERT_MLM
CUDA_VISIBLE_DEVICES=$gpu python run_ner.py \
  --model_name_or_path="../meta-specialization/save/domain_eval/ner_all/BERT_MLM" \
  --train_file="./data/ner/ner_all_train.json" \
  --validation_file="./data/ner/ner_all_dev.json" \
  --test_file="./data/ner/ner_fiction_test.json" \
  --output_dir="save/ner/fiction/BERT_MLM_ALL" \
  --cache_dir="./cache/transformer" \
  --num_train_epochs=10 --save_total_limit 2 \
  --max_seq_length=128 \
  --do_train \
  --do_eval \
  --do_predict \
  --return_entity_level_metrics --overwrite_cache --overwrite_output_dir \
  $add1 $add2 $add3 $add4