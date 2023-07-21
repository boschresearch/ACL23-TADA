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
  --output_dir="save/nli/fiction/BERT_META_ALL_ATTENTION" \
  --model_name_or_path_2="../meta-specialization/save/domain_eval/government/BERT_MLM_EMB_ONLY" \
  --model_name_or_path_3="../meta-specialization/save/domain_eval/telephone/BERT_MLM_EMB_ONLY" \
  --model_name_or_path_4="../meta-specialization/save/domain_eval/fiction/BERT_MLM_EMB_ONLY" \
  --model_name_or_path_5="../meta-specialization/save/domain_eval/slate/BERT_MLM_EMB_ONLY" \
  --model_name_or_path_6="../meta-specialization/save/domain_eval/travel/BERT_MLM_EMB_ONLY" \
  --use_metaemb=True --use_average=True --use_attention=True --ignore_tod=False \
  --save_steps=-1 --overwrite_cache --overwrite_output_dir \
  $add1 $add2 $add3 $add4