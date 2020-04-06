#!/bin/sh

echo "Arguments: list of all possible lambdas (immitation rates): $*"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
sleep 10

for t in 1 4 16 32 64 128 256
do
	for lmb in $*
	do
		python3 train_student.py --save_path ../data/student_data/search_optimal_kd_params/students_epoch_108.t_$t.lmb_$lmb.11per --train_epochs 108 --imitation_lmb $lmb --temperature $t 
	done
done
