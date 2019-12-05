#!/bin/bash -v

OUTPUT_DIR=../../results/results_stacked_debert_dae_complete_earlyStopWithEvalLoss

DATASET=chatbot
EPOCH_1st=3
EPOCH=3
BS_TRAIN=8
BS_EVAL=1
for EPOCH_AE in 100 1000; do
    echo "Training ${DATASET} dataset for ${EPOCH} epochs and ${EPOCH_AE} ep autoencoder"

    DATA_DIR="../../data/intent_data/complete_data_with_target/${DATASET}/"

    for SEED in 1 2 3 4 5 6 7 8 9 10; do
        OUTPUT_DIR_1st_LAYER="${OUTPUT_DIR}/${DATASET}_ep${EPOCH_1st}_bs${BS_TRAIN}_seed${SEED}_first_layer_epae${EPOCH_AE}/"
        OUTPUT_DIR_2nd_LAYER="${OUTPUT_DIR}/${DATASET}_ep${EPOCH_1st}_bs${BS_TRAIN}_seed${SEED}_second_layer_epae${EPOCH_AE}/"

        # Train
        CUDA_VISIBLE_DEVICES=0,1 python ../../run_stacked_debert_dae_classifier.py --task_name "${DATASET}_intent" --save_best_model --do_train_autoencoder --do_train_second_layer --do_eval --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size $BS_TRAIN --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs_autoencoder $EPOCH_AE --num_train_epochs $EPOCH --output_dir_first_layer $OUTPUT_DIR_1st_LAYER --output_dir $OUTPUT_DIR_2nd_LAYER
        # Eval
        CUDA_VISIBLE_DEVICES=0,1 python ../../run_stacked_debert_dae_classifier.py --task_name "${DATASET}_intent" --save_best_model --do_eval --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size $BS_TRAIN --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs_autoencoder $EPOCH_AE --num_train_epochs $EPOCH --output_dir_first_layer $OUTPUT_DIR_1st_LAYER --output_dir $OUTPUT_DIR_2nd_LAYER
    done
done
