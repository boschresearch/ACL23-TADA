<!---

    Copyright (c) 2023 Robert Bosch GmbH and its subsidiaries.

-->

# TADA: Efficient Task-Agnostic Domain Adaptation for Transformers

This is the companion code for the experiments reported in the paper

> "TADA: Efficient Task-Agnostic Domain Adaptation for Transformers"  by Chia-Chien Hung, Lukas Lange and Jannik Strötgen published at ACL 2023 Findings.

The paper can be found [here](https://aclanthology.org/2023.findings-acl.31.pdf)


## Introduction

Intermediate training of pre-trained transformer-based language models on domain-specific data leads to substantial gains for downstream tasks. To increase efficiency and prevent catastrophic forgetting alleviated from full domain-adaptive pre-training, approaches such as adapters have been developed. However, these require additional parameters for each layer, and are criticized for their limited expressiveness. In this work, we introduce **TADA**, a novel task-agnostic domain adaptation method which is modular, parameter-efficient, and thus, data-efficient. Within **TADA**, we retrain the embeddings to learn domain-aware input representations and tokenizers for the transformer encoder, while freezing all other parameters of the model. Then, task-specific fine-tuning is performed. We further conduct experiments with meta-embeddings and newly introduced meta-tokenizers, resulting in one model per task in multi-domain use cases. Our broad evaluation in 4 downstream tasks for 14 domains across single- and multi-domain setups and high- and low-resource scenarios reveals that **TADA** is an effective and efficient alternative to full domain-adaptive pre-training and adapters for domain adaptation, while not introducing additional parameters or complex training steps.

Overview of **TADA** framework:

<img src="/img/overview.png" width="1000"/>

## Citation

If you use any source codes, or datasets included in this repo in your work, please cite the following paper:
<pre>
@inproceedings{hung-etal-2023-tada,
    title = "{TADA}: Efficient Task-Agnostic Domain Adaptation for Transformers",
    author = {Hung, Chia-Chien  and
      Lange, Lukas  and
      Str{\"o}tgen, Jannik},
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.31",
    pages = "487--503"
}
</pre>

## Pretrained Models

The pre-trained models can be easily loaded using huggingface [Transformers](https://github.com/huggingface/transformers) or Adapter-Hub [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers) library using the **AutoModel** function. Following pre-trained versions are supported:
* `TODBERT/TOD-BERT-JNT-V1`: TOD-BERT pre-trained using both MLM and RCL objectives 
* `bert-base-cased`
* `bert-base-uncased`

The scripts for downstream tasks on DST and RR are mainly modified from [TOD-BERT](https://github.com/jasonwu0731/ToD-BERT) and [DS-TOD](https://github.com/umanlp/DS-TOD), where there might be slight version differences of the packages, which are noted down in the `requirements.txt` file.

## Background Data 

Please refer to the `background` folder for more infos.

## Downstream Data

The dialog_datasets in use of our paper for DST and RR are from MultiWOZ-2.1, which we further followed the preprocessing step from [here](https://github.com/jasonwu0731/ToD-BERT), and split the five domains into different subfiles. The full dialog_datasets can be found under [here](https://drive.google.com/file/d/1j8ZpC8Rl2GQPmMAgj1AHBZiYmRhjZdj3/view?usp=sharing).
For NER and NLI downstream task data, please refer to the descriptions in the paper for more information.


This repository is currently under the following structure:
```
.
└── background
└── eval-dialog
    └── models
    └── utils
    └── dialog_datasets
└── eval-ner_nli
    └── metrics
    └── data
└── img
└── meta-specialization
└── tokenizer
└── README.md
```

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication "TADA: Efficient Task-Agnostic Domain Adaptation for Transformers". 
It will neither be maintained nor monitored in any way.

## License

**TADA** is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.

For a list of other open source components, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

The software including its dependencies may be covered by third party rights, including patents. You should not execute this code unless you have obtained the appropriate rights, which the authors are not purporting to give.
