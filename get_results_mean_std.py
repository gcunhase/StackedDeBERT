import json
import numpy as np


# STTError
# Parameters:
#   is_incomplete_test: True if model was trained with complete data and tested with incomplete

root_name = './results_thesis/results_stacked_debert_dae_earlyStopWithLoss_lower_STTerror/'
#root_name = './results_thesis/results_stacked_debert_dae_complete_earlyStopWithEvalLoss/'
stt_error, dataname, epoch, epae, lrae, bs, tts_stt_type = [True, "chatbot", 3, 1000, '0.0001', 8, 'gtts_witai']

prefix = ''
if stt_error:
    prefix = tts_stt_type + '/'

if float(lrae) < 0.01:
    root_dir = '{root_name}/{dataname}/{prefix}bs{bs}_epae{epae}_lrae{lrae}/{dataname}_ep{ep}_bs{bs}_'.\
        format(root_name=root_name, dataname=dataname, prefix=prefix, bs=bs, epae=epae, ep=epoch, lrae=lrae)
else:
    root_dir = '{root_name}/{dataname}/{prefix}bs{bs}_epae{epae}/{dataname}_ep{ep}_bs{bs}_'. \
        format(root_name=root_name, dataname=dataname, prefix=prefix, bs=bs, epae=epae, ep=epoch)

f1_micro_str_all = ""
f1_micro_arr = []
f1_micro_str_all += "| {}    ".format(0)
for i in range(1, 10 + 1):
    #tmp_dir = "{}seed{}/".format(root_dir, i)
    tmp_dir = "{}seed{}_second_layer_epae{}/".format(root_dir, i, epae)
    tmp_dir += "eval_results_test.json"
    #tmp_dir += "eval_results.json"

    # Load json file
    with open(tmp_dir, 'r') as f:
        datastore = json.load(f)
        f1_score = datastore['f1_micro']
        f1_micro_arr.append(f1_score)
        f1_micro_str_all += "|{:.2f}".format(f1_score*100)

f1_micro_str_all += "|{:.2f}|{:.2f}|\n".format(np.mean(f1_micro_arr)*100, np.std(f1_micro_arr)*100)

print(f1_micro_str_all)
