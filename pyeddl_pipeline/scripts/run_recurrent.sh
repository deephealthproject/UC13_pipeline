# Recurrent raw signal experiments

for patient in  "chb05" "chb06" "chb08" "chb12" "chb14" "chb15" "chb24" "chb01" "chb03"
do
	python python/train_recurrent_detector.py --index ../indexes_detection/$patient/train.txt --index-val ../indexes_detection/$patient/validation.txt --id $patient --model lstm --epochs 10 --batch-size 64 --gpus 1 >log/${patient}_recurrent.out 2>log/${patient}_recurrent.err

done

