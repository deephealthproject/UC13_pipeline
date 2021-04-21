#!/bin/bash

#model_format='onnx'
model_format='eddl'

#python python/evaluate_classifier.py  --index etc/index_training.txt --model 1a --model-filename models/model_classifier_1a-9.${model_format} 
#python python/evaluate_classifier.py  --index etc/index_test.txt     --model 1a --model-filename models/model_classifier_1a-9.${model_format} 
 python python/evaluate_classifier.py  --index etc/index_mini.txt     --model 1a --model-filename models/model_classifier_1a-18.${model_format} 
