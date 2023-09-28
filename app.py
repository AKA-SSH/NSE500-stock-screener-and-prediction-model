import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from copy import deepcopy

from sklearn.preprocessing import MinMaxScaler

# fetch historical data for each stock in the list for last D days
def fetch_data(D):
    # symbols list
    symbols = [symbol+'.NS' for symbol in listed_stocks.Symbol]

    dataframes = []
    # fetching data
    for symbol in symbols:
        try:
            data = yf.Ticker(symbol).history(period=str(D)+'d')
            data['Symbol'] = symbol.split('.')[0]
            dataframes.append(data)
        except IndexError:
            print(f'error fetching data for {symbol}: {IndexError}')

    if not dataframes:
        return None
    
    # putting collected data in dataframe
    historical_data= pd.concat(dataframes)
    historical_data.reset_index(inplace=True)

    # optimising dataframe
    historical_data = historical_data.round(2)
    historical_data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
    historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.strftime('%Y-%m-%d')
    historical_data.set_index('Date', inplace=True)
    
    return historical_data

# fetching price deviation in last D days
def fetch_price_band(D):
    data= fetch_data(D)
    grouped_data= data.groupby('Symbol')
    
    # values for calculations
    min_price= grouped_data.Low.min()
    max_price= grouped_data.High.max()
    avg_price= grouped_data.Close.mean()
    
    # calculation
    price_range= (max_price-min_price)/avg_price
    
    # assigning the calculated result to a dataframe
    price_range_data= pd.Series(price_range, name=f'Price Deviation ({D} D)').round(5)
    price_range_data= price_range_data.to_frame()
    price_range_data.reset_index(inplace= True)
    
    # creating merged dataframe
    merged_dataframe= pd.merge(listed_stocks, price_range_data, on= 'Symbol', how= 'inner')
    merged_dataframe.sort_values(by=f'Price Deviation ({D} D)', inplace=True)
    merged_dataframe.reset_index(inplace= True, drop= True)
    
    return merged_dataframe

# fetching the price action of respective symbol
def symbol_price_action(symbol):
    dataframe= pd.DataFrame()
    
    data= yf.Ticker(symbol+'.NS').history(period='1750d').Close.to_frame()
    data.index= data.index.date.astype('datetime64')
    data= data.round(2)

    dataframe['LTP']= data.iloc[-1]  # Latest data-point
    
    # short term
    if len(data) >= 125:
        dataframe['ST']= data.iloc[-125]   # 125 days ago
    else:
        dataframe['ST']= dataframe.LTP

    # medium term
    if len(data) >= 500:
        dataframe['MT']= data.iloc[-500]   # 500 days ago
    else:
        dataframe['MT']= dataframe.ST

    # long term
    if len(data) >= 1250:
        dataframe['LT']= data.iloc[-1250]  # 1250 days ago
    else:
        dataframe['LT']= dataframe.MT
    
    dataframe['ST %']= ((dataframe.LTP - dataframe.ST) * 100 / dataframe.LTP).round(3)
    dataframe['MT %']= ((dataframe.LTP - dataframe.MT) * 100 / dataframe.LTP).round(3)
    dataframe['LT %']= ((dataframe.LTP - dataframe.LT) * 100 / dataframe.LTP).round(3)

    dataframe['Symbol']= symbol
    dataframe.reset_index(inplace=True, drop=True)
    
    return dataframe

# fetch the price action details for each stock
def fetch_price_action():
    dataframe= pd.DataFrame()
    for symbol in listed_stocks['Symbol']:
        symbol_data= symbol_price_action(symbol)
        dataframe= pd.concat([dataframe, symbol_data], ignore_index= True)
    return dataframe

# fetch the final analysis
def fetch_data_details(D):
    price_band= fetch_price_band(D)
    price_action= fetch_price_action()

    merged_dataframe= pd.merge(price_band, price_action, on= 'Symbol', how='inner')

    return merged_dataframe

# calculate stoploss and target for the selected stock
def fetch_stop_loss_and_target(symbol, risk_reward_ratio=2, atr_period=14, atr_multiplier=3, rsi_period=14):
    symbol_ns = symbol + '.NS'
    data = yf.Ticker(symbol_ns).history(period=str(max(atr_period, rsi_period)) + 'd').round(2).iloc[:, :4]
    data.index = data.index.date.astype('datetime64')

    # calculating average true range
    C = data['Close'].to_numpy()
    H = data['High'].to_numpy()
    L = data['Low'].to_numpy()

    H_L= H - L
    H_PDC= np.abs(H - np.roll(C, 1))
    L_PDC= np.abs(L - np.roll(C, 1))

    TR= np.maximum.reduce([H_L, H_PDC, L_PDC])

    ATR= np.zeros(len(TR))
    ATR[atr_period - 1]= np.mean(TR[:atr_period])
    for i in range(atr_period, len(TR)):
        ATR[i]= (ATR[i-1]*(atr_period-1) + TR[i])/atr_period

    data[f'ATR ({atr_period})']= ATR
        
    # calculating relative strength index
    delta= np.diff(data.Close, prepend= np.nan)
    
    gain= np.where(delta > 0,  delta, 0)
    loss= np.where(delta < 0, -delta, 0)
    
    avg_gain= np.convolve(gain, np.ones(rsi_period)/rsi_period, mode= 'valid')
    avg_loss= np.convolve(loss, np.ones(rsi_period)/rsi_period, mode= 'valid')

    rs= avg_gain/avg_loss

    rsi= 100-(100/(1+rs))
    rsi= np.concatenate((np.full(rsi_period - 1, np.nan), rsi))

    data[f'RSI ({rsi_period})']= rsi
    
    # calculating stop loss and target
    data['SL'] = data.Close - data[f'ATR ({atr_period})'] * atr_multiplier
    data['Target'] = data.Close + data[f'ATR ({atr_period})'] * atr_multiplier * risk_reward_ratio

    resultant = data.tail(1).copy()
    resultant = resultant.rename_axis(f'R : R Ratio = {round(risk_reward_ratio, 1)}')
    resultant = resultant.round(2)
    resultant.drop(f'ATR ({atr_period})', axis= 1, inplace= True)
    resultant.index = resultant.index.strftime('%Y-%m-%d')

    return resultant

# web application
st.sidebar.markdown('### Upload listed stocks CSV file')
st.sidebar.markdown("##### reference: [nseindia listed stock](https://archives.nseindia.com/content/indices/ind_nifty500list.csv)")
uploaded_file= st.sidebar.file_uploader("or upload your own CSV file", type=["csv"])

if uploaded_file is not None:
    listed_stocks= pd.read_csv(uploaded_file).iloc[:, :3]

    D= st.sidebar.number_input('range:', min_value= 1, max_value= 15)

# fetch narrow range data for screening
    if st.sidebar.button('Fetch Stock Data'):
        details= fetch_data_details(int(D))

    if 'details' in locals():
        st.write(f'Narrow ranges created in the last {D} days:')
        st.write(details)

# downloading option for fetched data in CSV format
        csv= details.to_csv(index= False)
        date= datetime.today().strftime('%y-%m-%d')
        st.download_button(label= 'Download in CSV format',
                           data= csv,
                           file_name= f'{date}-deviation-range-{D}.csv',
                           key= f'download_data_{D}_days')

st.sidebar.markdown('## OR')

st.sidebar.write('### Upload screened stocks CSV file')
uploaded_details_file= st.sidebar.file_uploader('upload screened stock details file', type= ['csv'])
if uploaded_details_file is not None:
    details= pd.read_csv(uploaded_details_file)

    if 'details' in locals():
        st.write(f'Details from the uploaded file:')
        st.write(details)

# calculate stoploss and target
symbol_to_analyze= st.text_input('Enter NSE 500 listed stock:')

# ATR
col1, col2= st.columns(2)
with col1:
    atr_period= st.number_input('ATR Period:', min_value= 1, value= 14)
with col2:
    atr_multiplier= st.number_input('ATR Multiplier:', min_value=1.0, value= 3.0, step= 0.1)

col3, col4= st.columns(2)
# RSI
with col3:
    rsi_period= st.number_input('RSI Period:', min_value= 1, value= 14)

# risk-reward ratio
with col4:
    risk_reward_ratio= st.number_input('Risk-Reward Ratio:', min_value= 1.0, value= 2.0, step= 0.1)

if st.button('Calculate Stop-Loss and Target'):
    if symbol_to_analyze:
        st.markdown(f'#### Stop Loss and Target for {symbol_to_analyze}')
        stop_loss_target= fetch_stop_loss_and_target(symbol_to_analyze, risk_reward_ratio, atr_period, atr_multiplier, rsi_period)
        show_output= True
    else:
        st.warning('Please enter a stock symbol.')

if 'show_output' in locals() and show_output:
        st.write(stop_loss_target)

# stock price prediction using pytorch
device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
scaler= MinMaxScaler(feature_range= (-1, 1))

# download the last 5 years data of input stock symbol
def fetch_data(symbol, frames):
    data= yf.Ticker(symbol+'.NS').history(period= '5y').Close.to_frame()
    data.columns= [column.lower() for column in data.columns]
    data.index= data.index.date.astype('datetime64')

    for frame in range(1, frames+1):
        data[f'shift ({frame})']= data.close.shift(frame)
    
    data.dropna(inplace=True)
    data= data[data.columns[::-1]]
    data= data.apply(lambda x: x.astype('float32'))
    
    scaled_data= scaler.fit_transform(data)
    
    result= pd.DataFrame(data= scaled_data, columns= data.columns, index= data.index)

    return result

# convert the data into tensor and fetch date, features and label separately
def fetch_tensor(symbol, frames):
    dataframe= fetch_data(symbol, frames)
    x= dataframe.iloc[:, :-1].values
    y= dataframe.iloc[:, -1].values
    x= x.reshape((len(y), x.shape[1], 1))
    
    return x, y

# split the tensor into train, validation and test
def split_tensor(symbol, frames):
    x, y= fetch_tensor(symbol, frames)
    
    split_factor_2= int(len(y)*0.95) 

    x_train, y_train= x[:split_factor_2], y[:split_factor_2]
    x_test, y_test= x[split_factor_2:], y[split_factor_2:]

    x_train= x_train.reshape((-1, frames, 1))
    y_train= y_train.reshape((-1, 1))
    x_test= x_test.reshape((-1, frames, 1))
    y_test= y_test.reshape((-1, 1))

    x_train= torch.tensor(x_train).float()
    y_train= torch.tensor(y_train).float()
    x_test= torch.tensor(x_test).float()
    y_test= torch.tensor(y_test).float()

    return x_train, y_train, x_test, y_test

# custom class for splitting torch dataset
class TimeSeriesDataset(Dataset):
  def __init__(self, x, y):
    self.x= x
    self.y= y

  def __len__(self):
    return len(self.x)
  
  def __getitem__(self, i):
    return self.x[i], self.y[i]

# fetching datasets using class TimeSeriesDataset
def fetch_dataset(symbol, frames):
  x_train, y_train, x_test, y_test= split_tensor(symbol, frames)
  train_dataset= TimeSeriesDataset(x_train, y_train)
  test_dataset= TimeSeriesDataset(x_test, y_test)

  return train_dataset, test_dataset

# fetching the data loaders
def fetch_data_loader(symbol, frames, batch_size):
  train_dataset, test_dataset= fetch_dataset(symbol, frames)
  train_loader= DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
  test_loader= DataLoader(test_dataset, batch_size= batch_size, shuffle= False)

  return train_loader, test_loader

# splitting into batches for fitting
def fetch_data_batches(symbol, frames, batch_size):
  train_loader, test_loader= fetch_data_loader(symbol, frames, batch_size)
  for _, batch in enumerate(train_loader):
    x_batch, y_batch= batch[0].to(device), batch[1].to(device)

    return x_batch, y_batch
  
# making LSTM model for prediction
class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super().__init__()
    self.hidden_size= hidden_size
    self.num_stacked_layers= num_stacked_layers
    
    self.lstm= nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first= True)
    self.fc= nn.Linear(hidden_size, 1)
  
  def forward(self, x):
    batch_size= x.size(0)
    h0= torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    c0= torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    
    out, _ = self.lstm(x, (h0, c0))
    out= self.fc(out[:, -1, :])
    return out

# training process
def train_one_epoch(symbol, frames, batch_size):
  model.train(True)
  running_loss= 0.0
  
  train_loader, test_loader= fetch_data_loader(symbol, frames, batch_size)

  for batch_index, batch in enumerate(train_loader):
    x_batch, y_batch= batch[0].to(device), batch[1].to(device)

    output= model(x_batch)
    loss= loss_function(output, y_batch)
    running_loss+= loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % 100== 99:
      avg_loss_across_batches= running_loss/100
      running_loss= 0.0

# validation of training
def validate_one_epoch(symbol, frames, batch_size):
  model.train(False)
  running_loss= 0.0

  train_loader, test_loader= fetch_data_loader(symbol, frames, batch_size)

  for batch_index, batch in enumerate(test_loader):
    x_batch, y_batch= batch[0].to(device), batch[1].to(device)

    with torch.no_grad():
      output= model(x_batch)
      loss= loss_function(output, y_batch)
      running_loss += loss
    
    avg_loss_across_batches= running_loss/len(test_loader)

model= LSTM(1,4,1)
model.to(device)

# Add a button to the main panel for prediction
if st.button('Run Prediction'):
    if symbol_to_analyze:
        st.markdown(f'#### Price Prediction for {symbol_to_analyze}')

        # Add your code for prediction and chart generation here
        symbol = symbol_to_analyze
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

        for epoch in range(25):
            train_one_epoch(symbol, 5, 15)
            validate_one_epoch(symbol, 5, 15)

        x_train, y_train, x_test, y_test = split_tensor(symbol, 5)

        with torch.no_grad():
            predicted = model(x_train.to(device)).to('cpu').numpy()

        test_predictions = model(x_test.to(device)).detach().cpu().numpy().flatten()

        dummies = np.zeros((x_test.shape[0], 5 + 1))
        dummies[:, 0] = test_predictions
        dummies = scaler.inverse_transform(dummies)

        test_predictions = deepcopy(dummies[:, 0])

        dummies = np.zeros((x_test.shape[0], 5 + 1))
        dummies[:, 0] = y_test.flatten()
        dummies = scaler.inverse_transform(dummies)

        inverse_scaled_y_test = deepcopy(dummies[:, 0])

        plt.figure(figsize=(14, 6))
        plt.plot(inverse_scaled_y_test, label='True Price')
        plt.plot(test_predictions, label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'{symbol} Price Prediction')
        plt.xticks([])
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning('Please enter a stock symbol.')