# Downstream task shell script for dialog task evaluation.
# The script is modified from TOD-BERT, DS-TOD models.
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
adapter_dir=$5
add1=$6
add2=$7
add3=$8

## DST
CUDA_VISIBLE_DEVICES=$gpu python main_domain_adapter.py \
    --my_model=BeliefTracker \
    --model_type=${model} \
    --dataset='["multiwoz"]' \
    --task_name="dst" \
    --earlystop="joint_acc" \
    --output_dir=${output_dir}/DST/MWOZ \
    --do_train \
    --task=dst \
    --example_type=turn \
    --cache_dir="./save/transformers" \
    --model_name_or_path=${bert_dir} \
    --adapter_name_or_path=${adapter_dir} \
    --save_adapter_path=${output_dir}/DST/MWOZ/MLM \
    --batch_size=6 --eval_batch_size=6 \
    --usr_token=[USR] --sys_token=[SYS] \
    --eval_by_step=4000 \
    $add1 $add2 $add3
    
### Response Retrieval
#CUDA_VISIBLE_DEVICES=$gpu python main_domain_adapter.py \
#    --my_model=dual_encoder_ranking \
#    --do_train \
#    --task=nlg \
#    --task_name=rs \
#    --example_type=turn \
#    --model_type=${model} \
#    --model_name_or_path=${bert_dir} \
#    --adapter_name_or_path=${adapter_dir} \
#    --output_dir=${output_dir}/RR/MWOZ/ \
#    --save_adapter_path=${output_dir}/RR/MWOZ/RS \
#    --cache_dir="./save/transformers" \
#    --batch_size=24 --eval_batch_size=100 \
#    --usr_token=[USR] --sys_token=[SYS] \
#    --fix_rand_seed \
#    --eval_by_step=1000 \
#    --max_seq_length=256 \
#    $add1 $add2 $add3
