#!/bin/bash


ls ../clean_signals/ch*/ch*-summary.txt | python3 python/old_python/generate_labels.py 
