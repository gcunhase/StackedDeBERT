import matplotlib.pyplot as plt
import numpy as np

"""
Grouped bar chart with labels: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
Gray color chart: https://colorswall.com/palette/25394/v
"""

ibleu = [0, 0.44, 0.50]
wer = [0, 2.39, 3.11]
plot_f1_improvement = False

labels = ['Complete (0.00/0.00)', 'gtts-witai (0.44/2.39)', 'macsay-witai (0.50/3.11)']
if plot_f1_improvement:
    f1_baseline = [0.08, 0.08, 0.08]
    f1_proposed = [0.08, 0.94, 1.89]
    f1_proposed_improvement = None
    labels_str1 = ['99.06', '96.23', '94.34']
    labels_str2 = ['99.06', '97.17', '96.23']
else:
    f1_baseline = [[92.45, 91.51, 86.79],
                   [99.06, 92.89, 91.51],
                   [96.23, 87.74, 81.13],
                   [95.24, 94.29, 94.29],
                   [99.06, 95.28, 91.51],
                   [98.11, 96.23, 94.34]]
    f1_proposed = [99.06, 97.17, 96.23]
    f1_proposed_improvement = [0.0, 0.94, 1.89]
    baseline_names = ['Baseline: Rasa (spacy)', 'Baseline: Rasa (tf)', 'Baseline: Dialogflow',
                      'Baseline: SAP', 'Baseline: Sem Hash', 'Baseline: BERT']

x = np.arange(len(labels))  # the label locations
#width = 0.35  # the width of the bars
width = 0.35

fig, ax = plt.subplots()
rects1 = []
if not plot_f1_improvement:
    location = x - width/len(f1_baseline)
    for i in range(len(f1_baseline), 0, -1):
        rects1.append(ax.bar(x - i*width/len(f1_baseline), f1_baseline[len(f1_baseline)-i], width, label=baseline_names[len(f1_baseline)-i]))  #, color=[.68, .85, .90]))  # lightblue
    rects2 = ax.bar(x + width / len(f1_baseline), f1_proposed, width, label='Proposed', color=[.42, .37, .73])  # purple
else:
    rects1 = ax.bar(x - width / 2, f1_baseline, width, label='Baseline (best)', color=[.68, .85, .90])  # lightblue
    rects2 = ax.bar(x + width/2, f1_proposed, width, label='Proposed', color=[.42, .37, .73])  # purple

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Noise level (iBLEU/WER score)')
if plot_f1_improvement:
    ax.set_ylabel('Improvement in F1-score (%)')
else:
    ax.set_ylabel('Accuracy in F1-score (%)')
ax.set_title('Robustness bar plot for the Chatbot NLU Corpus', fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
if plot_f1_improvement:
    ax.set_ylim([0, 2])
else:
    ax.set_ylim([80, 100])


def draw_diff_bars(f1_baseline, f1_proposed, plot_f1_improvement=False):
    for i in range(1, len(f1_baseline)):
        y_start = f1_baseline[i]+0.01
        y_end = f1_proposed[i]-0.01
        # Vertical line
        plt.plot([x[i] - 0.03, x[i] - 0.03], [y_start, y_end], color='#3f3d3b', linewidth=1)  # lighter gray = 808080, 696969, 3f3d3b
        # Horizontal lines
        plt.plot([x[i] - 0.05, x[i] - 0.01], [y_start, y_start], color='#3f3d3b', linewidth=1)
        plt.plot([x[i] - 0.05, x[i] - 0.01], [y_end, y_end], color='#3f3d3b', linewidth=1)
        # Text: difference between bars
        if plot_f1_improvement:
            plt.text(x[i] - 0.22, (y_end - y_start) / 2 + f1_baseline[i], str(f1_proposed[i]), color='k')
        else:
            plt.text(x[i]-0.22, (y_end - y_start)/2 + f1_baseline[i]-0.08, "{:.2f}".format(f1_proposed[i]-f1_baseline[i]), color='k')
            #plt.text(x[i] - 0.22, (y_end - y_start) / 2 + f1_baseline[i] + 0.2, "{:.2f}".format(f1_proposed[i] - f1_baseline[i]), color='k')


def autolabel(rects, labels_str):
    """Attach a text label above each bar in *rects*, displaying its height."""
    va = 'baseline'  # 'baseline', bottom
    for rect, l in zip(rects, labels_str):
        height = rect.get_height()
        ax.annotate('{}'.format(l),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va=va, color='#696969')


if plot_f1_improvement:
    autolabel(rects1, labels_str1)
    autolabel(rects2, labels_str2)

# Draw difference between bars
draw_diff_bars(f1_baseline[-1], f1_proposed, plot_f1_improvement=plot_f1_improvement)
if not plot_f1_improvement:
    plt.grid(axis='y', linestyle='--')
    ax.set_axisbelow(True)

fig.tight_layout()

plt.show()
