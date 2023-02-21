#!/bin/bash


input_file="data/database_sample.tsv"
ner_model="S1000-ner-model"
sentences_on_batch="100"
batch_size="32"
output_id="output"

python3 tagstringdb.py \
    --batch_size "$batch_size" \
    --input_data "$input_file" \
    --output_spans "$output_id" \
    --output_tsv "$output_id" \
    --sentences_on_batch "$sentences_on_batch" \
    --ner_model_dir "$ner_model" \

