#!/bin/bash -v
# This script: Trains both BERT and Stacked DeBBERT together
#   Seed 1: BERT seed 1 followed by Stacked DeBERT seed 1
#   Seed 2: BERT seed 2 followed by Stacked DeBERT seed 2 ...
# Previous: Train BERT in separate script and select the best out of 10 seeds. Then train Stacked DeBERT with 10 seeds
#    considering the best BERT only. For example, consider the best BERT was at seed 2, then:
#   Seed 1: BERT seed 2 followed by Stacked DeBERT seed 1
#   Seed 2: BERT seed 2 followed by Stacjed DeBERT seed 2 ...

#OUTPUT_DIR=../../results_thesis/results_stacked_debert_dae_earlyStopWithLoss_lower_STTerror
OUTPUT_DIR=../../results_thesis/results_stacked_debert_dae_complete_earlyStopWithEvalLoss
EVAL_OUTPUT_ROOT=../../results_thesis/results_stacked_debert_dae_complete_earlyStopWithEvalLoss/test_with_incomplete

DATASET=chatbot
echo $DATASET
BS_TRAIN=8
BS_EVAL=1
for TTS in "gtts"; do
   for STT in "google" "witai" "sphinx"; do
      DATA_DIR="../../data/intent_data/stterror_data/${DATASET}/${TTS}_${STT}/"
      EPOCH_1st=3
      EPOCH=3
      for EPOCH_AE in 100 1000; do
           AUTOENCODER_LR=0.0001  # regular
           echo "Training ${DATASET} dataset with ${TTS}-${STT} for ${EPOCH} epochs and ${EPOCH_AE} ep autoencoder aelr ${AUTOENCODER_LR}"

           for SEED in 1 2 3 4 5 6 7 8 9 10; do
              EVAL_OUTPUT_DIR="${EVAL_OUTPUT_ROOT}/${DATASET}/bs${BS_TRAIN}_epae${EPOCH_AE}_lrae${AUTOENCODER_LR}/${DATASET}_ep${EPOCH_1st}_bs${BS_TRAIN}_seed${SEED}_epae${EPOCH_AE}/"

              # BERT (--do_train)
              OUTPUT_DIR_1st_LAYER="${OUTPUT_DIR}/${DATASET}/bs${BS_TRAIN}_epae${EPOCH_AE}_lrae${AUTOENCODER_LR}/${DATASET}_ep${EPOCH_1st}_bs${BS_TRAIN}_seed${SEED}_first_layer_epae${EPOCH_AE}/"
              # Stacked DeBERT (--do_train_autoencoder --do_train_second_layer)
              OUTPUT_DIR_2nd_LAYER="${OUTPUT_DIR}/${DATASET}/bs${BS_TRAIN}_epae${EPOCH_AE}_lrae${AUTOENCODER_LR}/${DATASET}_ep${EPOCH_1st}_bs${BS_TRAIN}_seed${SEED}_second_layer_epae${EPOCH_AE}/"

              # Eval: Test incomplete
              CUDA_VISIBLE_DEVICES=1 python ../../run_stacked_debert_dae_classifier.py --seed $SEED --autoencoder_lr $AUTOENCODER_LR --task_name "${DATASET}_intent" --save_best_model --do_eval --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size $BS_TRAIN --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs_autoencoder $EPOCH_AE --num_train_epochs $EPOCH --output_dir_first_layer $OUTPUT_DIR_1st_LAYER --output_dir $OUTPUT_DIR_2nd_LAYER --eval_output_filename "${TTS}_${STT}" --eval_output_dir $EVAL_OUTPUT_DIR
           done
      done
    done
done

for TTS in "macsay"; do
   for STT in "google"; do
      DATA_DIR="../../data/intent_data/stterror_data/${DATASET}/${TTS}_${STT}/"
      EPOCH_1st=3
      EPOCH=3
      for EPOCH_AE in 100 1000; do
           AUTOENCODER_LR=0.0001  # regular
           echo "Training ${DATASET} dataset with ${TTS}-${STT} for ${EPOCH} epochs and ${EPOCH_AE} ep autoencoder aelr ${AUTOENCODER_LR}"

           for SEED in 1 2 3 4 5 6 7 8 9 10; do
              EVAL_OUTPUT_DIR="${EVAL_OUTPUT_ROOT}/${DATASET}/bs${BS_TRAIN}_epae${EPOCH_AE}_lrae${AUTOENCODER_LR}/${DATASET}_ep${EPOCH_1st}_bs${BS_TRAIN}_seed${SEED}_epae${EPOCH_AE}/"

              # BERT (--do_train)
              OUTPUT_DIR_1st_LAYER="${OUTPUT_DIR}/${DATASET}/bs${BS_TRAIN}_epae${EPOCH_AE}_lrae${AUTOENCODER_LR}/${DATASET}_ep${EPOCH_1st}_bs${BS_TRAIN}_seed${SEED}_first_layer_epae${EPOCH_AE}/"
              # Stacked DeBERT (--do_train_autoencoder --do_train_second_layer)
              OUTPUT_DIR_2nd_LAYER="${OUTPUT_DIR}/${DATASET}/bs${BS_TRAIN}_epae${EPOCH_AE}_lrae${AUTOENCODER_LR}/${DATASET}_ep${EPOCH_1st}_bs${BS_TRAIN}_seed${SEED}_second_layer_epae${EPOCH_AE}/"

              # Eval: Test incomplete
              CUDA_VISIBLE_DEVICES=1 python ../../run_stacked_debert_dae_classifier.py --seed $SEED --autoencoder_lr $AUTOENCODER_LR --task_name "${DATASET}_intent" --save_best_model --do_eval --do_lower_case --data_dir $DATA_DIR --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size $BS_TRAIN --eval_batch_size $BS_EVAL --learning_rate 2e-5 --num_train_epochs_autoencoder $EPOCH_AE --num_train_epochs $EPOCH --output_dir_first_layer $OUTPUT_DIR_1st_LAYER --output_dir $OUTPUT_DIR_2nd_LAYER --eval_output_filename "${TTS}_${STT}" --eval_output_dir $EVAL_OUTPUT_DIR
           done
      done
    done
done
