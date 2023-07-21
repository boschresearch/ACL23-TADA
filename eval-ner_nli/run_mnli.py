# Evaluate NLI script.
# The script is modified from transformers module from HuggingFace.
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

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
import transformers.adapters.composition as ac
from transformers import (
    AdapterConfig,
    AdapterTrainer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    MultiLingAdapterArguments,
    default_data_collator,
    set_seed,
)
#from trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from meta_model import BertEmbed, BertMetaEmbed, BertMetaDomainEmbed

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: Optional[str] = field(default="mnli", metadata={"help": "The name of the task (ner, pos...)."})
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    eval_language: str = field(
        default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
    )
    train_language: Optional[str] = field(
        default=None, metadata={"help": "Train language if it is different from the evaluation language."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_metaemb: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the meta embedding."
        },
    )
    use_domain_metaemb: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the domain meta embedding."
        },
    )
    use_average: bool = field(
        default=True,
        metadata={
            "help": "Meta Embedding method: average."
        },
    )
    use_attention: bool = field(
        default=False,
        metadata={
            "help": "Meta Embedding method: attention."
        },
    )
    ignore_tod: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore BERT."
        },
    )
    method: str = field(
        default="subword",
        metadata={
            "help": "Which subword aggregation method to use: subword, whitespace"
        },
    )
    model_name_or_path_2: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name_or_path_3: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name_or_path_4: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name_or_path_5: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name_or_path_6: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        
    extension = data_args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        #train_dataset = train_dataset.filter(lambda example: example["label"] in ["entailment", "neutral", "contradiction"])
    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.filter(lambda example: example["label"] in ["entailment", "neutral", "contradiction"])
    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        predict_dataset = predict_dataset.filter(lambda example: example["label"] in ["entailment", "neutral", "contradiction"])

    # Labels
    label_list = sorted(set(raw_datasets["train"]["label"]))
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        finetuning_task="xnli",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if model_args.use_metaemb:
        embedding_1 = BertEmbed.from_pretrained(model_args.model_name_or_path, cache_dir = model_args.cache_dir) #bert
        embedding_2 = BertEmbed.from_pretrained(model_args.model_name_or_path_2, cache_dir = model_args.cache_dir) #bert-MLMEMB-SAMEDOMAIN
        embedding_3 = None if model_args.model_name_or_path_3==None else BertEmbed.from_pretrained(model_args.model_name_or_path_3, cache_dir = model_args.cache_dir) #bert-MLMEMB--OTHERDOMAIN
        embedding_4 = None if model_args.model_name_or_path_4==None else BertEmbed.from_pretrained(model_args.model_name_or_path_4, cache_dir = model_args.cache_dir) #bert-MLMEMB--OTHERDOMAIN
        embedding_5 = None if model_args.model_name_or_path_5==None else BertEmbed.from_pretrained(model_args.model_name_or_path_5, cache_dir = model_args.cache_dir) #bert-MLMEMB--OTHERDOMAIN
        embedding_6 = None if model_args.model_name_or_path_6==None else BertEmbed.from_pretrained(model_args.model_name_or_path_6, cache_dir = model_args.cache_dir) #bert-MLMEMB--OTHERDOMAIN
        print("embedding 1: {}".format(model_args.model_name_or_path))
        print("embedding 2: {}".format(model_args.model_name_or_path_2))
        print("embedding 3: {}".format(model_args.model_name_or_path_3))
        print("embedding 4: {}".format(model_args.model_name_or_path_4))
        print("embedding 5: {}".format(model_args.model_name_or_path_5))
        print("embedding 6: {}".format(model_args.model_name_or_path_6))
        model.bert.embeddings = BertMetaEmbed(config = model.config, 
                                              embedding_1 = embedding_1, 
                                              embedding_2 = embedding_2,
                                              embedding_3 = embedding_3,
                                              embedding_4 = embedding_4,
                                              embedding_5 = embedding_5,
                                              embedding_6 = embedding_6,
                                              use_average = True if model_args.use_average else False,
                                              use_attention = True if model_args.use_attention else False,
                                              ignore_tod = True if model_args.ignore_tod else False
                                              )
        print("Use Attention: {}".format(model.bert.embeddings.use_attention))
        print("Use Average: {}".format(model.bert.embeddings.use_average))
        print("Ignore BERT: {}".format(model.bert.embeddings.ignore_tod))
        for param in model.bert.named_parameters():
            if "embedding_" in param[0]:
                param[1].requires_grad=False
        for param in model.bert.named_parameters():
            print("Param: {} Requires_grad: {}".format(param[0], param[1].requires_grad))
        print(model)
    elif model_args.use_domain_metaemb:
        embedding_1 = BertEmbed.from_pretrained(model_args.model_name_or_path, cache_dir = model_args.cache_dir) #bert
        embedding_2 = BertEmbed.from_pretrained(model_args.model_name_or_path_2, cache_dir = model_args.cache_dir) #bert-MLMEMB-SAMEDOMAIN
        embedding_3 = None if model_args.model_name_or_path_3==None else BertEmbed.from_pretrained(model_args.model_name_or_path_3, cache_dir = model_args.cache_dir) #bert-MLMEMB--OTHERDOMAIN
        embedding_4 = None if model_args.model_name_or_path_4==None else BertEmbed.from_pretrained(model_args.model_name_or_path_4, cache_dir = model_args.cache_dir) #bert-MLMEMB--OTHERDOMAIN
        embedding_5 = None if model_args.model_name_or_path_5==None else BertEmbed.from_pretrained(model_args.model_name_or_path_5, cache_dir = model_args.cache_dir) #bert-MLMEMB--OTHERDOMAIN
        embedding_6 = None if model_args.model_name_or_path_6==None else BertEmbed.from_pretrained(model_args.model_name_or_path_6, cache_dir = model_args.cache_dir) #bert-MLMEMB--OTHERDOMAIN

        tokenizer_1 = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir = model_args.cache_dir)
        tokenizer_2 = AutoTokenizer.from_pretrained(model_args.model_name_or_path_2, cache_dir = model_args.cache_dir)
        tokenizer_3 = None if model_args.model_name_or_path_3==None else AutoTokenizer.from_pretrained(model_args.model_name_or_path_3, cache_dir = model_args.cache_dir)
        tokenizer_4 = None if model_args.model_name_or_path_4==None else AutoTokenizer.from_pretrained(model_args.model_name_or_path_4, cache_dir = model_args.cache_dir)
        tokenizer_5 = None if model_args.model_name_or_path_5==None else AutoTokenizer.from_pretrained(model_args.model_name_or_path_5, cache_dir = model_args.cache_dir)
        tokenizer_6 = None if model_args.model_name_or_path_6==None else AutoTokenizer.from_pretrained(model_args.model_name_or_path_6, cache_dir = model_args.cache_dir)
        
        print("embedding/tokenizer 1: {}".format(model_args.model_name_or_path))
        print("embedding/tokenizer 2: {}".format(model_args.model_name_or_path_2))
        print("embedding/tokenizer 3: {}".format(model_args.model_name_or_path_3))
        print("embedding/tokenizer 4: {}".format(model_args.model_name_or_path_4))
        print("embedding/tokenizer 5: {}".format(model_args.model_name_or_path_5))
        print("embedding/tokenizer 6: {}".format(model_args.model_name_or_path_6))
        model.bert.embeddings = BertMetaDomainEmbed(config = model.config, 
                                                    embedding_1 = embedding_1, 
                                                    embedding_2 = embedding_2,
                                                    embedding_3 = embedding_3,
                                                    embedding_4 = embedding_4, 
                                                    embedding_5 = embedding_5,
                                                    embedding_6 = embedding_6,
                                                    tokenizer_1 = tokenizer_1,
                                                    tokenizer_2 = tokenizer_2,
                                                    tokenizer_3 = tokenizer_3,
                                                    tokenizer_4 = tokenizer_4,
                                                    tokenizer_5 = tokenizer_5,
                                                    tokenizer_6 = tokenizer_6,
                                                    method = model_args.method,
                                                    use_average = True if model_args.use_average else False,
                                                    use_attention = True if model_args.use_attention else False,
                                                    ignore_tod = True if model_args.ignore_tod else False
                                                    )
        print("Subword aggregation method: {}".format(model.bert.embeddings.method))
        print("Use Attention: {}".format(model.bert.embeddings.use_attention))
        print("Use Average: {}".format(model.bert.embeddings.use_average))
        print("Ignore BERT: {}".format(model.bert.embeddings.ignore_tod))
        for param in model.bert.named_parameters():
            if "embedding_" in param[0]:
                param[1].requires_grad=False
        for param in model.bert.named_parameters():
            print("Param: {} Requires_grad: {}".format(param[0], param[1].requires_grad))
        print(model)
        
    # Setup adapters
    if adapter_args.train_adapter:
        task_name = data_args.task_name or "xnli"
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                print("Load Adapter", adapter_args.load_adapter)
                print("Adapter Config", adapter_config)
                print("Task", task_name)
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, config=adapter_config)
                print("Add adapter")
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                load_as=adapter_args.language,
            )
        else:
            lang_adapter_name = None
        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
        else:
            model.set_active_adapters([task_name])
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode.Use --train_adapter to enable adapter training"
            )
    print(model)
    
    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
    
    def preprocess_function(examples):
        tokenized_inputs = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )
        labels = []
        for i, lab in enumerate(examples["label"]):
            labels.append(label_to_id[lab])
        #print(list(zip(examples["label"], labels)))
        tokenized_inputs["label"] = labels
        #print(tokenized_inputs["label"])
        return tokenized_inputs
    
    # def preprocess_function(examples):
    #     # Tokenize the texts
    #     return tokenizer(
    #         examples["sentence1"],
    #         examples["sentence2"],
    #         padding=padding,
    #         max_length=data_args.max_seq_length,
    #         truncation=True,
    #     )

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Get the metric function
    metric = load_metric("./metrics/xnli")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()