#!/bin/bash

find ../clean_signals/ -type f -name "*.fbank.pbz2" | egrep -v "chb17" | sed 's/\.fbank.pbz2//g' | sort >etc/index_all.txt


for n in {1..24}
do
    echo ${n}
    pattern=$(printf "chb%02d" ${n})
    grep ${pattern} etc/index_all.txt >etc/index_${pattern}.txt

    l=$(wc -l etc/index_${pattern}.txt | cut -f1 -d' ')
    if [ ${l} -lt 1 ]
    then
        rm -f etc/index_${pattern}.txt
    fi
done
