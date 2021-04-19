#!/bin/bash


ls ../clean_signals/ch*/ch*-summary.txt | python python/generate_labels.py 
