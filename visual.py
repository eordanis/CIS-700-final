import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

default_dir = '/content/CIS-700-final/results/'


def display_stacking_dataframe(directory=None):
    
    if directory is None:
        directory = default_dir
    
    df = pd.read_csv('results/stacking_classifier_metrics.csv')
    display(df)


def display_stacking_metrics(directory=None, style=None):
    
    if directory is None:
        directory = default_dir
    
    if style is None:
        style = 'bar'
        
    df = pd.read_csv('results/stacking_classifier_metrics.csv')
    df_plot = df[["Classifier","Accuracy","Variance"]]
    df.set_index(["Classifier"],inplace=True)
    df.plot(kind=style,alpha=0.75, rot=45)
    plt.xlabel("")
    plt.show()

    # save metrics to .png for later use in pdf report
    plt.savefig(directory + 'classifier_metric_chart.png')
