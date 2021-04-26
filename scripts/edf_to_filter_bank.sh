#!/bin/bash

#
# we suggest to run this scripts in a similar way to:
#
# nohup scripts/edf_to_filter_bank.sh  >log/edf2fbank.out 2>log/edf2fbank.err &
#

file_list_1=$(find ../clean_signals/chb* -type f -name "*.edf.pbz2" | sort)

file_list_2=""
for file in ${file_list_1}
do
    target_file=${file/.edf.pbz2/.fbank.pbz2}
    if [ ! -f ${target_file} ]
    then
        file_list_2="${file_list_2} ${file}"
    else
        file_list_2="${file_list_2} #${file}"
    fi
done


for file in ${file_list_2}
do
    if [ ${file:0:1} != '#' ]
    then
        echo ${file}
    fi
done | python python/edf_to_filter_bank.py
