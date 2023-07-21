# Load data script for NLI, where the data could be obtained from https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
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

import json
mnli_inp_dir = 'data/multinli_1.0/'
mnli_dev_inp_fname = mnli_inp_dir + 'multinli_1.0_dev_matched.txt'
mnli_train_inp_fname = mnli_inp_dir + 'multinli_1.0_train.txt'
mnli_dev_out_template = 'data/nli/mnli_$GENRE_dev.json'
mnli_train_out_template = 'data/nli/mnli_$GENRE_train.json'

def prepare_mnli_file(in_file, out_file_template):
    instances_grouped_by_genre = {}
    with open(in_file, 'r', encoding='utf-8') as fin:
        content = fin.read().splitlines()
        for line in content[1:]:
            if line.strip():
                s = line.split('\t')
                label, _, _, _, _, sent1, sent2, _, _, genre, _, _, _, _, _ = s
                if genre not in instances_grouped_by_genre:
                    instances_grouped_by_genre[genre] = []
                instances_grouped_by_genre[genre].append((label, sent1, sent2, genre))
    for genre, instances in instances_grouped_by_genre.items():
        out_file = out_file_template.replace('$GENRE', genre)
        with open(out_file, 'w', encoding='utf-8') as fout:
            for label, sent1, sent2, genre in instances:
                fout.write(json.dumps({
                    'sentence1': sent1,
                    'sentence2': sent2,
                    'label': label,
                    'genre': genre,
                }, ensure_ascii=False) + '\n')
prepare_mnli_file(mnli_dev_inp_fname, mnli_dev_out_template)
prepare_mnli_file(mnli_train_inp_fname, mnli_train_out_template)