#!/bin/bash -v

OUTPUT_DIR=../../results/results_bert_earlyStopWithLoss_lower_STTerror/

BS_TRAIN=8
BS_EVAL=1
for DATASET in chatbot; do
    echo $DATASET
    for TTS in "gtts" "macsay"; do
        for STT in "witai"; do
            for EPOCH in 3; do
                echo "Training ${DATASET} dataset with ${TTS}-${STT} missing for ${EPOCH} epochs"

                DATA_DIR="../data/intent_data/stterror_data/${DATASET}/${TTS}_${STT}/"

                for SEED in 1 2 3 4 5 6 7 8 9 10; do
                    OUT_PATH="${OUTPUT_DIR}/${DATASET}/${TTS}_${STT}/${DATASET}_ep${EPOCH}_bs${BS_TRAIN}_seed${SEED}/"

                    # Train
                    CUDA_VISIBLE_DEVICES=0,1 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --save_best_model --do_train --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size $BS_TRAIN --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH
                    # Eval
                    CUDA_VISIBLE_DEVICES=0,1 python ../../run_classifier.py --seed $SEED --task_name "${DATASET}_intent" --save_best_model --do_eval --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size $BS_TRAIN --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs $EPOCH --output_dir $OUT_PATH
                done
            done
        done
    done
done
