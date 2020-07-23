import numpy as np


def get_avg(l, avg_type="micro avg"):
    classification_report = l.split(avg_type)[1]
    classification_report = classification_report.replace(" ", "")
    precision = classification_report[0:4]
    recall = classification_report[4:8]
    f1 = classification_report[8:12]
    return precision, recall, f1

dir_path = './results/results_trigram_hash_complete_768_10runs/'

runs = 10

#for data_type in ["inc_gtts_witai", "inc_macsay_witai"]:
for data_type in ["complete"]:
    str_write = ''
    for dataset_name in ['Chatbot']:
        str_write += dataset_name + '\n'

        for run in range(1, runs + 1):
            subdir_path = dir_path + '{}_run{}/'.format(data_type, run)
            filename = subdir_path + '{}_log.txt'.format(dataset_name)
            f = open(filename, 'r')
            f1_micro_str = ''
            precision_macro_str, recall_macro_str, f1_macro_str = '', '', ''
            lines = f.read().split('\n')
            # Get max acc from 1 run
            model_name = ''
            precision, recall, f1 = [], [], []
            for l in lines:
                if l is not "":
                    if "Training: " in l:
                        model_name = l.split("Training: ")[1]
                        model_name = model_name.split("(")[0]
                    elif "Departure Time" in l:
                        precision_tmp, recall_tmp, f1_tmp = get_avg(l, avg_type="Departure Time")
                        precision.append(float(precision_tmp))
                        recall.append(float(recall_tmp))
                        f1.append(float(f1_tmp))
                    elif "Find Connection" in l:
                        precision_tmp, recall_tmp, f1_tmp = get_avg(l, avg_type="Find Connection")
                        precision.append(float(precision_tmp))
                        recall.append(float(recall_tmp))
                        f1.append(float(f1_tmp))
                    elif "macro avg" in l:
                        precision_macro = np.sum(precision)/len(precision)
                        recall_macro = np.sum(recall)/len(recall)
                        f1_macro = np.sum(f1)/len(f1)
                        precision_macro_str += model_name + ': ' + str(precision_macro) + '\n'
                        recall_macro_str += model_name + ': ' + str(recall_macro) + '\n'
                        f1_macro_str += model_name + ': ' + str(f1_macro) + '\n'
                        precision, recall, f1 = [], [], []

            # Save accuracy in file - MACRO
            filename_precision_macro = subdir_path + '{}_precision_macro.txt.txt'.format(dataset_name)
            f_out = open(filename_precision_macro, 'w')
            f_out.write(precision_macro_str)

            filename_recall_macro = subdir_path + '{}_recall_macro.txt.txt'.format(dataset_name)
            f_out = open(filename_recall_macro, 'w')
            f_out.write(recall_macro_str)

            filename_f1_macro = subdir_path + '{}_f1_macro.txt.txt'.format(dataset_name)
            f_out = open(filename_f1_macro, 'w')
            f_out.write(f1_macro_str)

