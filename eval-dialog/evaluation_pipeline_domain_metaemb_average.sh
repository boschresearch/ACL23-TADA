# Downstream task shell script for dialog task evaluation.
# The script is modified from TOD-BERT models.
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
model=$2
bert_dir=$3
output_dir=$4
add1=$5
add2=$6
add3=$7
add4=$8

# Response Retrieval
CUDA_VISIBLE_DEVICES=$gpu python main_domain_meta.py \
   --my_model=dual_encoder_ranking \
   --do_train \
   --task=nlg \
   --task_name=rs \
   --cache_dir="./cache/transformer" \
   --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/RR/MWOZ \
   --batch_size=24 --eval_batch_size=100 \
   --usr_token=[USR] --sys_token=[SYS] \
   --fix_rand_seed \
   --eval_by_step=1000 \
   --max_seq_length=256 --use_average --ignore_tod --overwrite\
   --model_name_or_path_2="../meta-specialization/save/domain_eval/restaurant/TOD-BERT_MLM_EMB_ONLY"\
   --model_name_or_path_3="../meta-specialization/save/domain_eval/train/TOD-BERT_MLM_EMB_ONLY"\
   --model_name_or_path_4="../meta-specialization/save/domain_eval/hotel/TOD-BERT_MLM_EMB_ONLY"\
   --model_name_or_path_5="../meta-specialization/save/domain_eval/taxi/TOD-BERT_MLM_EMB_ONLY"\
   --model_name_or_path_6="../meta-specialization/save/domain_eval/attraction/TOD-BERT_MLM_EMB_ONLY"\
   --domain_option="full" \
   --domain="restaurant" \
   $add1 $add2 $add3 $add4

# DST
CUDA_VISIBLE_DEVICES=$gpu python main_domain_meta.py \
 --my_model=BeliefTracker \
 --model_type=${model} \
 --dataset='["multiwoz"]' \
 --task_name="dst" \
 --earlystop="joint_acc" \
 --output_dir=${output_dir}/DST/MWOZ \
 --do_train \
 --task=dst \
 --cache_dir="./cache/transformer" \
 --example_type=turn \
 --model_name_or_path=${bert_dir} \
 --batch_size=6 --eval_batch_size=6 \
 --usr_token=[USR] --sys_token=[SYS] \
 --eval_by_step=1000 --use_average --ignore_tod --overwrite\
 --model_name_or_path_2="../meta-specialization/save/domain_eval/restaurant/TOD-BERT_MLM_EMB_ONLY"\
 --model_name_or_path_3="../meta-specialization/save/domain_eval/train/TOD-BERT_MLM_EMB_ONLY"\
 --model_name_or_path_4="../meta-specialization/save/domain_eval/hotel/TOD-BERT_MLM_EMB_ONLY"\
 --model_name_or_path_5="../meta-specialization/save/domain_eval/taxi/TOD-BERT_MLM_EMB_ONLY"\
 --model_name_or_path_6="../meta-specialization/save/domain_eval/attraction/TOD-BERT_MLM_EMB_ONLY"\
 --domain_option="full" \
 --domain="restaurant" \
 $add1 $add2 $add3 $add4

