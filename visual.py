import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

default_dir = 'results/'


def display_stacking_dataframe(directory=None, adjusted=None):
    
    if directory is None:
        directory = default_dir
    
    if adjusted is not None:
        adjusted = '_adjusted'
    else:
        adjusted = ''

    df = pd.read_csv(default_dir + 'stacking_classifier_metrics' + adjusted + '.csv')
    display(df)


def display_stacking_metrics(directory=None, style=None, adjust_settings=None):
    
    if directory is None:
        directory = default_dir
    
    if style is None:
        style = 'barh'
    
    if adjusted is not None:
        adjusted = '_adjusted'
    else:
        adjusted = ''

    df = pd.read_csv(default_dir + 'stacking_classifier_metrics' + adjusted + '.csv')
    df_plot = df[["Classifier","Accuracy","Variance"]]
    df_plot.set_index(["Classifier"],inplace=True)
    df_plot.plot(kind=style, alpha=0.75, title='Classifier Metrics', figsize=(10, 10))
    plt.xlabel("Values")
    plt.show()

    # save metrics to .png for later use in pdf report
    plt.savefig(directory + 'classifier_metric_chart' + adjusted + '.png')

