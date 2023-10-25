import pandas as pd

sp500=pd.read_parquet('../data/sp500.parquet')

ff_factors=pd.read_parquet('../data/ff_factors.parquet')

data=pd.merge(sp500, ff_factors, on='Date', how='left')

data['Excess Return']=data['Monthly Returns'] - data['RF']

data.sort_index(inplace=True)
data['ex_ret_1']=data.groupby('Symbol')['Excess Return'].shift(-1)
data.sort_values(['Symbol', 'Date'], ascending=[True, True], inplace=True)

data.dropna(subset=['ex_ret_1'], inplace=True)
data.dropna(subset=['HML'], inplace=True)

data=data[data['Symbol']=='AMZN']
data.drop(columns=['Symbol'], inplace=True)