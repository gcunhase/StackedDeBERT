import numpy as np


def get_acc(acc_type='f1'):
    acc_avg = 0
    acc_max = 0
    acc_arr = []
    for run in range(1, runs + 1):
        acc = 0
        subdir_path = dir_path + '{}_run{}/'.format(data_type, run)
        filename = subdir_path + '{}_{}.txt.txt'.format(dataset_name, acc_type)
        f = open(filename, 'r')
        lines = f.read().split('\n')
        # Get max acc from 1 run
        for l in lines:
            if l is not "":
                l_split = l.split(': ')
                l_acc = float(l_split[1])
                print(l_acc)
                if l_acc >= acc:
                    acc = l_acc
        # Get average acc
        acc_avg += acc
        acc_arr.append(acc)
        # Get max acc
        if acc >= acc_max:
            acc_max = acc
    # acc_avg /= runs
    acc_avg = np.mean(acc_arr)
    acc_std = np.std(acc_arr)
    acc_max = np.max(acc_arr)
    return acc_avg, acc_std, acc_max

#dir_path = './results/results_semhash_twitter_sentiment140_10runs/'
dir_path = './results/results_trigram_hash_twitter_sentiment140_768_10runs/'

runs = 10

for data_type in ["corr", "inc", "inc_with_corr"]:
    str_write = ''
    for dataset_name in ['sentiment140']:
        str_write += dataset_name + '\n'

        acc_avg, acc_std, acc_max = get_acc(acc_type='f1_micro')
        str_write += '  F1-Micro - Avg-{}: {:.2f}\pm{:.2f}\n'.format(runs, acc_avg * 100, acc_std * 100)
        str_write += '  F1-Micro - Best-{}: {:.2f}\n\n'.format(runs, acc_max * 100)

        acc_avg, acc_std, acc_max = get_acc(acc_type='f1_macro')
        str_write += '  F1-Macro - Avg-{}: {:.2f}\pm{:.2f}\n'.format(runs, acc_avg * 100, acc_std * 100)
        str_write += '  F1-Macro - Best-{}: {:.2f}\n\n'.format(runs, acc_max * 100)

        acc_avg, acc_std, acc_max = get_acc(acc_type='precision_macro')
        str_write += '  Precision-Macro - Avg-{}: {:.2f}\pm{:.2f}\n'.format(runs, acc_avg * 100, acc_std * 100)
        str_write += '  Precision-Macro - Best-{}: {:.2f}\n\n'.format(runs, acc_max * 100)

        acc_avg, acc_std, acc_max = get_acc(acc_type='recall_macro')
        str_write += '  Recall-Macro - Avg-{}: {:.2f}\pm{:.2f}\n'.format(runs, acc_avg * 100, acc_std * 100)
        str_write += '  Recall-Macro - Best-{}: {:.2f}\n\n'.format(runs, acc_max * 100)
    f_out = open(dir_path + data_type, 'w')
    f_out.write(str_write)
