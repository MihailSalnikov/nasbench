#!/bin/sh

echo ">>" $1

for t in 4 16 1
do
	for lmb in 0.0 0.5 1.0
	do
		python3 train_student.py --save_path ../data/student_data/sokd_epoch12_$1/students_epoch_lmb$lmb-t$t-11per --train_epochs 12 --imitation_lmb $lmb --temperature $t --optimizer $1
	done
done

# CUDA_VISIBLE_DEVICES=0 python3 train_student.py --save_path ../data/student_data/search_optimal_kd_params_small_exp/students_epoch_lmb1_t4_11per --train_epochs 108 --imitation_lmb 1 --temperature 4 &
# CUDA_VISIBLE_DEVICES=1 python3 train_student.py --save_path ../data/student_data/search_optimal_kd_params_small_exp/students_epoch_lmb1_t16_11per --train_epochs 108 --imitation_lmb 1 --temperature 16 &
# CUDA_VISIBLE_DEVICES=3 python3 train_student.py --save_path ../data/student_data/search_optimal_kd_params_small_exp/students_epoch_lmb1_t32_11per --train_epochs 108 --imitation_lmb 1 --temperature 32 &&
#	CUDA_VISIBLE_DEVICES=0 python3 train_student.py --save_path ../data/student_data/search_optimal_kd_params_small_exp/students_epoch_lmb0_t1_11per --train_epochs 108 --imitation_lmb 0 --temperature 1
