# MLM shell script for domain adaptation via embedding-based.
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
bert_dir=$2
output_dir=$3
train_file=$4
val_file=$5
add1=$6
add2=$7
add3=$8
add4=$9

CUDA_VISIBLE_DEVICES=$gpu python run_intermediate_mlm_emb.py \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir} \
    --train_file=${train_file} \
    --validation_file=${val_file} \
    --cache_dir="./save/transformers" \
    --line_by_line \
    --do_train \
    --do_eval \
    --load_best_model_at_end \
    --max_train_samples=200000 \
    --max_eval_samples=10000 \
    --learning_rate=1e-5 --evaluation_strategy="steps"\
    --save_steps=3000 --logging_steps=1000 --gradient_accumulation_steps=1\
    --num_train_epochs=30 --save_total_limit 2\
    --per_device_train_batch_size=32 --max_seq_length=256 --update_emb_only\
    ${add1} ${add2} ${add3} ${add4}
