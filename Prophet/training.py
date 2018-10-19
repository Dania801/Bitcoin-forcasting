import sys
sys.path.append('../ARIMA')
import warnings
import pandas
import numpy
import visualize
import matplotlib.pylab as plot
from matplotlib.pylab import rcParams
from fbprophet import Prophet

dataold = visualize.loadData('../../data/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
colNames = ['ds', 'y'];
df = pandas.DataFrame(columns=colNames);
df['y'] = dataold['Open'];
df['ds'] = dataold['Timestamp'];
print (df);
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
future.tail()



warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 15,6