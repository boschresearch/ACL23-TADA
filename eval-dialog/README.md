# Downstream Evaluation for Dialog Tasks (DST, RR)

The evaluation scripts are mainly modified from [here](https://github.com/jasonwu0731/ToD-BERT), where there might be slight version differences of the packages, which are noted down in the `requirements.txt` file.

- **For model without adapter training**:
```
./evaluation_pipeline_domain.sh 0 todbert "../meta-specialization/save/domain_eval/taxi/TOD-BERT_MLM" save/domain_eval/taxi/BERT_MLM --domain "taxi" --overwrite
```

- **For model with adapter training**:
```
./evaluation_pipeline_domain_adapter.sh 0 todbert "TODBERT/TOD-BERT-JNT-V1" save/domain_eval/taxi/TOD-BERT_MLM_adapter "../meta-specialization/save/domain_eval/taxi/TOD-BERT_MLM_adapter/mlm" --domain "taxi" --overwrite
```

- **For model with meta-embeddings (attention, average)**:
```
./evaluation_pipeline_domain_metaemb_{attention, average}.sh 0 todbert "../meta-specialization/save/domain_eval/taxi/TOD-BERT_MLM" save/domain_eval/taxi/BERT_METAEMB --domain "taxi" --overwrite
```

- **For model with meta-tokenization meta-embeddings (attention, average)**:
```
./evaluation_pipeline_domain_metatok_{attention, average}.sh 0 todbert "../meta-specialization/save/domain_eval/taxi/TOD-BERT_MLM" save/domain_eval/taxi/BERT_METATOK --domain "taxi" --overwrite
```