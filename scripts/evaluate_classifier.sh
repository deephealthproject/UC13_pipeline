#!/bin/bash

#model_format='onnx'
model_format='eddl'

model_id='1a'
epoch='199'
model_filename="models/model_classifier_${model_id}-${epoch}.${model_format}"

#index_dataset="etc/index_training.txt"
index_dataset="etc/index_test.txt"
#index_dataset="etc/index_mini.txt"

python python/evaluate_classifier.py  --index ${index_dataset}  --model ${model_id}  --model-filename ${model_filename}
