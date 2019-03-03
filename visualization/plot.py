import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels.api as st
from IPython.display import display

class DataVisualization:

    def describe_data(self, data):
        print ("This is the shape of the data. We have {} data Points with {} variables each.".format(*data.shape))
        print ("Description of the data: ")
        display(data.describe()) 

    def variables_correlation(self, data):
        corr = data.corr(method="pearson", min_periods=1)
        # Returns an array of zeros with the same shape as the corr
        # array
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sb.axes_style("white"):
            sb.heatmap(corr, mask=mask, vmax=1., square=True)
            plt.show()

    def data_correlation(self, data):
        plt.figure(figsize=(25, 13))
        s_ax = plt.subplot(211)
        sub_data_set = data.Close.values.squeeze()[:100000]
        st.graphics.tsa.plot_acf(sub_data_set, lags=60, ax=s_ax)
        s_ax = plt.subplot(212)
        st.graphics.tsa.plot_pacf(sub_data_set, lags=60, ax=s_ax)
        plt.show()

    def plot_data(self, data):
        # We will create a plot with the variable Close
        data_range = range(0,len(data),87600)
        index_values = data.index.values
        x_ticks = [index_values[i] for i in data_range]
        plt.figure(figsize = (15,6))
        plt.plot(range(len(data)), (data["Close"]))
        plt.xticks(data_range, pd.to_datetime(x_ticks, unit='s').strftime('%d/%m/%Y'), rotation=90)
        plt.xlabel("Date",fontsize=8)
        plt.ylabel("Close Price",fontsize=8)
        plt.show()
    
    def plot_model_loss(self, history):
        plt.figure(figsize=(25, 13))
        plt.subplot(311)
        plt.plot(history.epoch, history.history["loss"])
        plt.plot(history.epoch, history.history["val_loss"])
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")
        plt.title("Model Behaviour")
        plt.legend(["loss", "val_loss"])

    def plot_results_against_true_data(self, tested_data, true_data):
        plt.figure(figsize = (15,6))
        # style
        plt.style.use('seaborn-darkgrid')
        # create a color palette
        palette = plt.get_cmap('Set1')
        plt.plot(
            range(len(tested_data)), 
            tested_data, 
            marker='', 
            color=palette(1), 
            linewidth=0.5, 
            alpha=1, 
            label="Predicted Data"
        )

        plt.plot(
            range(len(true_data)), 
            true_data, 
            marker='', 
            color=palette(2), 
            linewidth=0.5, 
            alpha=0.3, 
            label="Real Data"
        )
        
        plt.legend(loc=2, ncol=2)
        plt.title("Results comparation", loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("Hours")
        plt.ylabel("Price")
        plt.show()




