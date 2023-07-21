# Downstream task shell script for mnli evaluation.
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
CUDA_VISIBLE_DEVICES=$gpu python run_mnli.py \
  --model_name_or_path="bert-base-uncased" \
  --cache_dir="./cache/transformer" \
  --language="en" \
  --train_language="en" \
  --do_train \
  --do_predict \
  --per_device_train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3 \
  --max_seq_length=128 \
  --save_total_limit=2 \
  --train_file="./data/nli/mnli_all_train.json" \
  --test_file="./data/nli/mnli_fiction_dev.json" \
  --output_dir="save/nli/fiction/BERT_ALL" \
  --save_steps=-1 --overwrite_cache --overwrite_output_dir \
  $add1 $add2 $add3 $add4


# BERT_MLM
CUDA_VISIBLE_DEVICES=$gpu python run_mnli.py \
  --model_name_or_path="../meta-specialization/save/domain_eval/nli_all/BERT_MLM" \
  --cache_dir="./cache/transformer" \
  --language="en" \
  --train_language="en" \
  --do_train \
  --do_predict \
  --per_device_train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3 \
  --max_seq_length=128 \
  --save_total_limit=2 \
  --train_file="./data/nli/mnli_all_train.json" \
  --test_file="./data/nli/mnli_fiction_dev.json" \
  --output_dir="save/nli/fiction/BERT_MLM_ALL" \
  --save_steps=-1 --overwrite_cache --overwrite_output_dir \
  $add1 $add2 $add3 $add4