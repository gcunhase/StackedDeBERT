#!/bin/bash -v

OUTPUT_DIR=results/results_stacked_debert_dae_comp_inc_earlyStopWithEvalLoss_lower

EPOCH=3
BS_EVAL=1
for DATASET in chatbot askubuntu webapplications; do
    echo $DATASET
    for PERC in 0.1 0.2 0.3 0.4 0.5 0.8; do
        for EPOCH_AE in 100 1000 2000 5000; do
            echo "Training ${DATASET} dataset with ${PERC} missing for ${EPOCH} epochs and ${EPOCH_AE} ep autoencoder"

            DATA_DIR="/mnt/gwena/Gwena/IncompleteIntentionClassifier/data/comp_with_incomplete_data_tfidf_lower_${PERC}/nlu_eval_${DATASET}corpus/"
            OUTPUT_DIR_1st_LAYER="${OUTPUT_DIR}/${DATASET}_ep${EPOCH}_bs4_${PERC}/"
            OUTPUT_DIR_2nd_LAYER="${OUTPUT_DIR}/${DATASET}_ep${EPOCH}_bs4_${PERC}_second_layer_epae${EPOCH_AE}/"

            # Train
            CUDA_VISIBLE_DEVICES=0,1,2,3 python run_stacked_debert_dae_classifier.py --task_name "${DATASET}_intent" --save_best_model --do_train_autoencoder --do_train_second_layer --do_eval --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 4 --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs_autoencoder $EPOCH_AE --num_train_epochs $EPOCH --output_dir_first_layer $OUTPUT_DIR_1st_LAYER --output_dir $OUTPUT_DIR_2nd_LAYER
            # Evaluate
            CUDA_VISIBLE_DEVICES=0,1,2,3 python run_stacked_debert_dae_classifier.py --task_name "${DATASET}_intent" --save_best_model --do_eval --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 4 --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs_autoencoder $EPOCH_AE --num_train_epochs $EPOCH --output_dir_first_layer $OUTPUT_DIR_1st_LAYER --output_dir $OUTPUT_DIR_2nd_LAYER
        done
    done
done
