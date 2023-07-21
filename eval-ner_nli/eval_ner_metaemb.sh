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

CUDA_VISIBLE_DEVICES=$gpu python run_ner.py \
  --model_name_or_path="bert-base-uncased" \
  --train_file="./data/ner/ner_all_train.json" \
  --validation_file="./data/ner/ner_all_dev.json" \
  --test_file="./data/ner/ner_fiction_test.json" \
  --output_dir="save/ner/fiction/BERT_MLM_EMB_ONLY_META_ALL_ATTENTION_IGNORE_DYNAMIC" \
  --cache_dir="./cache/transformer" \
  --num_train_epochs=10 --save_total_limit 2 \
  --max_seq_length=128 \
  --do_train \
  --do_eval \
  --do_predict \
  --model_name_or_path_2="../meta-specialization/save/domain_eval/financial/BERT_MLM_EMB_ONLY" \
  --model_name_or_path_3="../meta-specialization/save/domain_eval/fiction-ner/BERT_MLM_EMB_ONLY" \
  --model_name_or_path_4="../meta-specialization/save/domain_eval/news/BERT_MLM_EMB_ONLY" \
  --model_name_or_path_5="../meta-specialization/save/domain_eval/clinical/BERT_MLM_EMB_ONLY" \
  --model_name_or_path_6="../meta-specialization/save/domain_eval/science/BERT_MLM_EMB_ONLY" \
  --use_domain_metaemb=True --use_average=True --use_attention=True --ignore_tod=True --method="dynamic"\
  --return_entity_level_metrics --overwrite_cache --overwrite_output_dir \
  $add1 $add2 $add3 $add4