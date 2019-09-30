#!/bin/bash -v

OUTPUT_DIR=../../results/results_bert_twitter_earlyStopWithLoss_lower_10seeds/

BS_TRAIN=4
BS_EVAL=1
for DATASET in sentiment140; do
    echo $DATASET
    for EPOCH in 3; do
        echo "Training ${DATASET} dataset with ${PERC} missing for ${EPOCH} epochs"

        DATA_DIR="../../data/twitter_sentiment_data/${DATASET}_corrected_sentences/"

        for SEED in 1 2 3 4 5 6 7 8 9 10; do
            OUT_PATH="${OUTPUT_DIR}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_corr_seed${SEED}/"

            # Train
            CUDA_VISIBLE_DEVICES=0,1,2 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --save_best_model --do_train --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size $BS_TRAIN --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH                # Eval
            # Eval
            CUDA_VISIBLE_DEVICES=0,1,2 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_sentiment" --save_best_model --do_eval --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size $BS_TRAIN --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH
        done
    done
done
