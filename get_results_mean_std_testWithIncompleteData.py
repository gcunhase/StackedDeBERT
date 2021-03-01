import json
import numpy as np

# Use this script if the model was trained with complete data but you wish to test it with another set of test data,
#  in our case incomplete data.

root_name = './results_thesis/results_stacked_debert_dae_complete_earlyStopWithEvalLoss/test_with_incomplete/'

dataname, epoch, epae, lrae, bs, tts_stt_type = ["chatbot", 3, 100, 0.0001, 8, 'gtts_witai']

if lrae < 0.001:
    root_dir = '{root_name}/{dataname}/bs{bs}_epae{epae}_lrae{lrae}/{dataname}_ep{ep}_bs{bs}_'.\
        format(root_name=root_name, dataname=dataname, bs=bs, epae=epae, ep=epoch, lrae=lrae)
else:
    root_dir = '{root_name}/{dataname}/bs{bs}_epae{epae}/{dataname}_ep{ep}_bs{bs}_'. \
        format(root_name=root_name, dataname=dataname, bs=bs, epae=epae, ep=epoch)

f1_micro_str_all = ""
f1_micro_arr = []
f1_micro_str_all += "| {}    ".format(0)
for i in range(1, 10 + 1):
    tmp_dir = "{}seed{}_epae{}/".format(root_dir, i, epae)
    tmp_dir += "eval_results_test_{tts_stt_type}.json".format(tts_stt_type=tts_stt_type)

    # Load json file
    with open(tmp_dir, 'r') as f:
        datastore = json.load(f)
        f1_score = datastore['f1_micro']
        f1_micro_arr.append(f1_score)
        f1_micro_str_all += "|{:.2f}".format(f1_score*100)

f1_micro_str_all += "|{:.2f}|{:.2f}|\n".format(np.mean(f1_micro_arr)*100, np.std(f1_micro_arr)*100)

print(f1_micro_str_all)
