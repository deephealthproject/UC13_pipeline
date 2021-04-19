#!/bin/bash

for i in {1..24}
do
    n=$(printf "%02d" ${i})

    if [ -f ../physionet.org/files/chbmit/1.0.0/chb${n}/chb${n}-summary.txt ]
    then
        if [ ! -f ../clean_signals/chb${n}/chb${n}-summary.txt ]
        then
            if [ -d ../clean_signals/chb${n} ]
            then
                echo "Copying ../physionet.org/files/chbmit/1.0.0/chb${n}/chb${n}-summary.txt   TO   ../clean_signals/chb${n}/chb${n}-summary.txt"
                cp -p ../physionet.org/files/chbmit/1.0.0/chb${n}/chb${n}-summary.txt ../clean_signals/chb${n}/chb${n}-summary.txt
            else
                echo "Destination directory ../clean_signals/chb${n} does not exist!"
            fi
        fi
    fi
done
