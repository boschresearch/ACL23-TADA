# Domain Specialization

For domain specialization, here are multiple methods you can implement for your own usage:

- **MLM**
```
./run_intermediate_mlm.sh 0 "TODBERT/TOD-BERT-JNT-V1" save/domain_eval/taxi/TOD-BERT_MLM ../background/DomainCC/train/taxi_200K_prep.txt ../background/DomainCC/test/taxi_10K_prep.txt --overwrite_output_dir --patience 3
```

- **MLM-adapter**
```
./run_intermediate_mlm_adapter.sh 0 "TODBERT/TOD-BERT-JNT-V1" save/domain_eval/taxi/TOD-BERT_MLM_adapter ../background/DomainCC/train/taxi_200K_prep.txt ../background/DomainCC/test/taxi_10K_prep.txt --overwrite_output_dir --patience 3
```

- **MLM-EMB**
```
./run_intermediate_mlm_only.sh 0 "TODBERT/TOD-BERT-JNT-V1" save/domain_eval/taxi/TOD-BERT_MLM_adapter ../background/DomainCC/train/taxi_200K_prep.txt ../background/DomainCC/test/taxi_10K_prep.txt --overwrite_output_dir --patience 3
```

- **MLM-EMB-TOK**
```
./run_intermediate_mlm_only_tokenizer.sh 0 "TODBERT/TOD-BERT-JNT-V1" save/domain_eval/taxi/TOD-BERT_MLM_adapter ../background/DomainCC/train/taxi_200K_prep.txt ../background/DomainCC/test/taxi_10K_prep.txt --overwrite_output_dir --patience 3 --tokenizer_name="../tokenizer/DomainCCAll/taxi-cc"
```