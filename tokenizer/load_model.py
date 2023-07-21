# Load and save pretrained model and tokenizer from HuggingFace.
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

from transformers import AutoModel, AutoTokenizer
import os

cache_dir = "./cache/transformer/"
model_name = "bert-base-uncased"

#os.system("cd {}".format(cache_dir))
#os.system("mkdir {}".format(model_name))

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
tokenizer.save_pretrained(cache_dir + model_name)
#model=AutoModel.from_pretrained(model_name, cache_dir = cache_dir)
#model.save_pretrained(cache_dir + model_name)