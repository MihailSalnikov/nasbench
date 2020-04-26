for t in 4 16 1
do
	for lmb in 0.0 0.5 1.0
	do
		python3 train_student.py \
            --save_path ../data/student_data/sop_epoch12_mnist/students_epoch_lmb$lmb-t$t-11per \
            --train_epochs 12 \
            --imitation_lmb $lmb \
            --temperature $t \
            --optimizer SGD \
            --train_data_files ../data/dataset/mnist/train_0.tfrecords,../data/dataset/mnist/train_1.tfrecords,../data/dataset/mnist/train_2.tfrecords,../data/dataset/mnist/train_3.tfrecords\
            --test_data_file ../data/dataset/mnist/test.tfrecords \
            --valid_data_file ../data/dataset/mnist/valid.tfrecords \
            --sample_data_file ../data/dataset/mnist/sample.tfrecords  \
            --num_train 54000 \
            --num_test 10000 \
            --num_valid 6000
	done
done