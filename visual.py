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


def display_stacking_metrics(directory=None, style=None, adjusted=None):
    
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


def display_comparision_dataframe(directory=None):
    if directory is None:
        directory = default_dir
        
    df = pd.read_csv(directory + 'stacking_classifier_metrics.csv')
    df1 = df.rename(columns={"Accuracy": "Acc_Standard", "Variance": "Var_Standard"})
    df = pd.read_csv(directory + 'stacking_classifier_metrics_adjusted.csv')
    df2 = df.rename(columns={"Accuracy": "Acc_Adjusted", "Variance": "Var_Adjusted"})
    new_df = pd.merge(df1, df2,  how='left')
    new_df = new_df.reindex(sorted(new_df.columns), axis=1)
    display(new_df.set_index('Classifier'))
    
    
def display_comparision_metrics(directory=None, style=None):
    if directory is None:
        directory = default_dir
    
    if style is None:
        style = 'barh'
        
    df = pd.read_csv(directory + 'stacking_classifier_metrics.csv')
    df1 = df.rename(columns={"Accuracy": "Acc_Standard", "Variance": "Var_Standard"})
    df = pd.read_csv(directory + 'stacking_classifier_metrics_adjusted.csv')
    df2 = df.rename(columns={"Accuracy": "Acc_Adjusted", "Variance": "Var_Adjusted"})
    new_df = pd.merge(df1, df2,  how='left')
    new_df = new_df.reindex(sorted(new_df.columns), axis=1)
    df_plot = new_df[["Classifier","Acc_Standard","Acc_Adjusted","Var_Standard", "Var_Adjusted"]]
    df_plot.set_index(["Classifier"],inplace=True)
    df_plot.plot(kind=style, alpha=0.75, title='Classifier Metrics', figsize=(10, 10))
    plt.xlabel("Values")
    plt.show()       

