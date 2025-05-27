# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/daily-minimum-temperatures-in-me.csv')
print(data.columns)
data['Date'] = pd.to_datetime(data['Date'])


data['Daily minimum temperatures'] = data['Daily minimum temperatures'].astype(str).str.replace('?', '', regex=False)
data['Daily minimum temperatures'] = pd.to_numeric(data['Daily minimum temperatures'], errors='coerce')

data.dropna(subset=['Daily minimum temperatures'], inplace=True)

plt.plot(data['Date'], data['Daily minimum temperatures'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series')
plt.show()

def check_stationarity(timeseries):
   
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['Daily minimum temperatures'])


plot_acf(data['Daily minimum temperatures'])
plt.show()
plot_pacf(data['Daily minimum temperatures'])
plt.show()


sarima_model = SARIMAX(data['Daily minimum temperatures'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['Daily minimum temperatures'][:train_size], data['Daily minimum temperatures'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()

```
### OUTPUT:
![image](https://github.com/user-attachments/assets/d9616b60-e243-4e29-b7b5-4f5b448aca4e)
![image](https://github.com/user-attachments/assets/4e79adc3-a5cc-436c-9069-bc681e7a9833)
![image](https://github.com/user-attachments/assets/dd18587a-4b37-4972-8702-d517b15d8de4)
![image](https://github.com/user-attachments/assets/518bcdfc-14b4-45d2-84ac-b0134c7a61d8)


### RESULT:
Thus the program run successfully based on the SARIMA model.
