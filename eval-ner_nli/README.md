# Downstream Evaluation for {NER, NLI} Tasks

- **For model without adapter training**:
```
./eval_{ner, nli}.sh 0
```

- **For model with meta-embeddings (attention, average)**:
```
./evaluation_{ner, nli}_metaemb.sh 0
```

- **For model with meta-tokenization meta-embeddings (attention, average)**:
```
./evaluation_{ner, nli}_metatok.sh 0
```