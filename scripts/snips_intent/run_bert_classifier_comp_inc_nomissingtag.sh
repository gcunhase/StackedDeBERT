#!/bin/bash -v

OUTPUT_DIR=../../results/results_bert_comp_inc_earlyStopWithLoss_lower_noMissingTag/

BS_TRAIN=32
BS_EVAL=1
for DATASET in snips; do
    echo $DATASET
    for PERC in 0.1 0.2 0.3 0.4 0.5 0.8; do
        for EPOCH in 3; do
            echo "Training ${DATASET} dataset with ${PERC} missing for ${EPOCH} epochs"

            DATA_DIR="../../data/snips_intent_data/comp_with_incomplete_data_tfidf_lower_${PERC}_noMissingTag/"

            for SEED in 1 2 3 4 5 6 7 8 9 10; do
                OUT_PATH="${OUTPUT_DIR}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_${PERC}_seed${SEED}/"

                # Train
                CUDA_VISIBLE_DEVICES=0,1,2 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --save_best_model --do_train --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size $BS_TRAIN --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH
                # Eval
                CUDA_VISIBLE_DEVICES=0,1,2 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --save_best_model --do_eval --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size $BS_TRAIN --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH
            done
        done
    done
done
