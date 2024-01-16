# python pip install requests numpy tensorflow pandas matplotlib seaborn openpyxl
import os
import sys

import requests
import numpy as np
from tensorflow import keras, reshape
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

from matplotlib.pyplot import figure

matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

import datetime

# Getting data from APIs to csv file
class DataModel:
    def __init__(self, TRAIN_CSV_FILENAME = "dataset.csv", BRENT_DATA_CSV_FILENAME = "Brent_oil_futures_historical_data.csv", PERCENT_FOR_DROP_COLUMN = 65, VERBOSE = 1):
        self.df = []
        self.TRAIN_CSV_FILENAME = TRAIN_CSV_FILENAME
        self.BRENT_DATA_CSV_FILENAME = BRENT_DATA_CSV_FILENAME
        self.PERCENT_FOR_DROP_COLUMN = PERCENT_FOR_DROP_COLUMN
        self.CURRENT_YEAR = datetime.date.today().year
        self.VERBOSE = VERBOSE
    
    class GetDataFromWBA:
        def __init__(self):
            self.API = 'http://api.worldbank.org/v2/country/%s/indicator/%s?format=json'
            
        def GetEnableCountries(self):
            page_count = int(requests.get('http://api.worldbank.org/v2/country/all?format=json').json()[0]['pages'])
            countries = []

            for page_number in range(1, page_count+1):
                _countries_dict_arr = requests.get(f'http://api.worldbank.org/v2/country/all?page={page_number}&format=json').json()[1]
                countries += [[country_dict['id'], country_dict['name']] for country_dict in _countries_dict_arr] # ['USA', 'RUS', 'SAU', 'IRQ', 'IND', 'CHN', 'CAN']

            return countries
        
        def GetDataFromWorldBankApi(self, countries):
            indicators = {'NY.GDP.MKTP.CD': 'GDP (current US$)',
                        'CRUDE_BRENT': 'Brent Crude Oil Price',
                        'EG.EGY.PROD.KT.OE': 'Energy production (kt of oil equivalent)',
                        'EG.ELC.PETR.KH': 'Electricity production from oil sources (kWh)',
                        'EG.ELC.PETR.ZS': 'Electricity production from oil sources (% of total)',
                        'EG.IMP.TOTL.KT.OE': 'Energy imports (kt of oil equivalent)',
                        'EG.USE.COMM.KT.OE': 'Energy use (kt of oil equivalent)',
                        'EG.USE.PCAP.KG.OE': 'Energy use (kg of oil equivalent per capita)',
                        'EN.EGY.PROD.KT.OE': 'Commercial energy production (kt of oil equivalent)',
                        'EN.TDF.CO': 'Traditional fuel use (kt of oil equivalent)',
                        'EU.EGY.USES.GD': 'GDP per unit of energy use (1987 US$ per kg of oil equivalent)',
                        'EU.EGY.USES.KG.OE.PC': 'Commercial energy use (kg of oil equivalent per capita)',
                        'EU.EGY.USES.KT.OE': 'Commercial energy use (kt of oil equivalent)',
                        'IS.ROD.DESL.KT': 'Road sector diesel fuel consumption (kt of oil equivalent)',
                        'IS.ROD.DESL.PC': 'Road sector diesel fuel consumption per capita (kg of oil equivalent)',
                        'IS.ROD.ENGY.KT': 'Road sector energy consumption (kt of oil equivalent)',
                        'IS.ROD.ENGY.PC': 'Road sector energy consumption per capita (kg of oil equivalent)',
                        'IS.ROD.SGAS.KT': 'Road sector gasoline fuel consumption (kt of oil equivalent)',
                        'IS.ROD.SGAS.PC': 'Road sector gasoline fuel consumption per capita (kg of oil equivalent)',
                        'NA.GDP.EXC.OG.CR': 'Total GDP excluding Oil and Gas (in IDR Million), Current Price',
                        'NA.GDP.EXC.OG.KR': 'Total GDP excluding Oil and Gas (in IDR Million), Constant Price',
                        'NA.GDP.INC.OG.CR': 'Total GDP including Oil and Gas (in IDR Million), Current Price',
                        'NA.GDP.INC.OG.KR': 'Total GDP including Oil and Gas (in IDR Million), Constant Price',
                        'NA.GDP.INC.OG.SNA08.CR': 'Total GDP including Oil and Gas (in IDR Million), SNA 2008, Current Price',
                        'NA.GDP.INC.OG.SNA08.KR': 'Total GDP including Oil and Gas (in IDR Million), SNA 2008, Constant Price',
                        'NRRV.SHR.PETR.CR': 'Total Natural Resources Revenue Sharing from Oil (in IDR, realization value)',
                        'NW.NCA.SAOI.PC': 'Natural capital per capita, nonrenewable assets: oil (constant 2018 US$)',
                        'NW.NCA.SAOI.TO': 'Natural capital, nonrenewable assets: oil (constant 2018 US$)',
                        'NY.GDP.PETR.RT.ZS': 'Oil rents (% of GDP)',
                        'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)',
                        'NY.GDP.DEFL.87.ZG': 'Inflation, GDP deflator (annual %)',
                        'NY.GDP.DEFL.KD.ZG': 'Inflation, GDP deflator (annual %)',
                        'NY.GDP.DEFL.KD.ZG.AD': 'Inflation, GDP deflator: linked series (annual %)',
                        'FM.LBL.BMNY.CN': 'Broad money (current LCU)',
                        'FM.LBL.BMNY.GD.ZS': 'Broad money (% of GDP)',
                        'FM.LBL.BMNY.IR.ZS': 'Broad money to total reserves ratio',
                        'FM.LBL.BMNY.ZG': 'Broad money growth (annual %)',
                        'FM.LBL.MONY.CN': 'Money (current LCU)',
                        'FM.LBL.MQMY.CN': 'Money and quasi money (M2) (current LCU)',
                        'FM.LBL.MQMY.CN.WB': 'Money Supply, Broadly Defined (local)',
                        'FM.LBL.MQMY.GD.ZS': 'Money and quasi money (M2) as % of GDP',
                        'FM.LBL.MQMY.GDP.ZS': 'Money and quasi money (M2) as % of GDP',
                        'FM.LBL.MQMY.IR.ZS': 'Money and quasi money (M2) to total reserves ratio',
                        'FM.LBL.MQMY.XD': 'Income velocity of money (GDP/M2)',
                        'FM.LBL.MQMY.ZG': 'Money and quasi money growth (annual %)',
                        'FM.LBL.QMNY.CN': 'Quasi money (current LCU)',
                        'FR.INR.MMKT': 'Money market rate (%)',
                        'FA.LBL.RCUR.CN': 'Currency Outside Banks  (local)',
                        'FB.AST.FRNO.ZS': 'Banking assets held by foreign-owned banks (% of total banking assets)',
                        'FB.AST.LOAN.CB.P3': 'Loan accounts, commercial banks (per 1,000 adults)',
                        'FB.AST.PUBO.ZS': 'Banking assets held by government-owned banks (% of total banking assets)',
                        'FM.LBL.NBNK.CN': 'Currency Ouside Banks (local)',
                        'FN.INR.CBIR': 'Central bank intervention rate (%)',
                        'GFDD.DI.06': 'Central bank assets to GDP (%)',
                        'FR.INR.RINR': 'Real interest rate (%)',
                        'SP.DYN.CBRT.IN': 'Birth rate, crude (per 1,000 people)',
                        'SP.DYN.CDRT.IN': 'Death rate, crude (per 1,000 people)',
                        'DT.CUR.CCVL.CD': 'Cross-currency valuation (current US$)',
                        'EG.EGY.PROD.KT.OE': 'Energy production (kt of oil equivalent)',
                        'EG.ELC.COAL.KH': 'Electricity production from coal sources (kWh)',
                        'EG.ELC.COAL.ZS': 'Electricity production from coal sources (% of total)',
                        'EG.ELC.FOSL.ZS': 'Electricity production from oil, gas and coal sources (% of total)',
                        'EG.ELC.HYRO.KH': 'Electricity production from hydroelectric sources (kWh)',
                        'EG.ELC.HYRO.ZS': 'Electricity production from hydroelectric sources (% of total)',
                        'EG.ELC.NGAS.KH': 'Electricity production from natural gas sources (kWh)',
                        'EG.ELC.NGAS.ZS': 'Electricity production from natural gas sources (% of total)',
                        'EG.ELC.NUCL.KH': 'Electricity production from nuclear sources (kWh)',
                        'EG.ELC.NUCL.ZS': 'Electricity production from nuclear sources (% of total)',
                        'EG.ELC.PETR.KH': 'Electricity production from oil sources (kWh)',
                        'EG.ELC.PETR.ZS': 'Electricity production from oil sources (% of total)',
                        'EG.ELC.PROD.KH': 'Electricity production (kWh)',
                        'EG.ELC.RNEW.KH': 'Electricity production from renewable sources (kWh)',
                        'EG.ELC.RNWX.KH': 'Electricity production from renewable sources, excluding hydroelectric (kWh)',
                        'EG.ELC.RNWX.ZS': 'Electricity production from renewable sources, excluding hydroelectric (% of total)'
                        }
            keys = list(indicators.keys())

            data = []

            def GetValuesByCountryAndIndicator(country, indicator, indicatorText):
                my_values = []

                try:
                    data = requests.get(self.API % (country, indicator)).json()
                    
                    for line in data[1]:
                        my_values.append({
                            'country_id': country,
                            'country': line['country']['value'],
                            'date': line['date'],
                            indicatorText: line['value']
                        })
                except Exception as err:
                    if(self.VERBOSE):
                        print(f'[ERROR] country, indicator ==> {country}, {indicator}, error ==> {err}')
                
                return my_values
            
            # For first set data by country and date
            for country in countries:
                values = GetValuesByCountryAndIndicator(country[0], keys[0], indicators[keys[0]])

                data.extend(values)

            for country in countries:
                for key in keys[1::]:
                    indicatorText = indicators[key]
                    values = GetValuesByCountryAndIndicator(country[0], key, indicatorText)

                    for value in values:
                        found = [i for i in data if i['country'] == country[1] and i['date'] == value['date']]

                        if len(found):
                            found[0][indicatorText] = value[indicatorText]
            return data

    class GetDataFromIMF:
        def __init__(self, current_year):
            self.API = 'https://www.imf.org/external/datamapper/api/v1/'
            self.CURRENT_YEAR = current_year

        def GetEnableCountries(self):
            _countries_dict_arr = requests.get(self.API + 'countries').json()['countries']
            countries = [[key, _countries_dict_arr[key]['label']] for key in _countries_dict_arr]

            return countries

        def GetDataFromIMFApi(self, countries):
            _indicators = requests.get(self.API + 'indicators').json()['indicators']
            indicators = [[key, _indicators[key]] for key in _indicators]
            
            data = []

            def GetValuesByCountryAndIndicator(countries, indicator, indicatorText):
                my_values = []

                try:
                    data = requests.get(self.API + indicator + '/' + str.join('/', countries)).json()['values'][indicator]
                    
                    for country_name in data:
                        try:
                            _country_full_name = [pair[1] for pair in countries if pair[0] == country_name]
                            country_full_name = _country_full_name[0] if len(_country_full_name) else 'None'
                            country_values = data[country_name]

                            for year in country_values:
                                if(int(year) > self.CURRENT_YEAR):
                                    continue

                                my_values.append({
                                    'country_id': country_name,
                                    'country': country_full_name,
                                    'date': year,
                                    indicatorText['label']: country_values[year]
                                })
                        except Exception as err:
                            if(self.VERBOSE):
                                print(f'[ERROR] country, indicator ==> {country_name}, {indicator}, error ==> {err}')
                except Exception as err:
                    if(self.VERBOSE):
                        print(f'[ERROR] indicator ==> {indicator}, error ==> {err}')
                
                return my_values

            for indicator in indicators:
                indicatorText = indicator[1]
                key = indicator[0]
                values = GetValuesByCountryAndIndicator(countries, key, indicatorText)

                data.extend(values)            
            return data

    def MakeMaskForLoc(self, df, date_column_name, start_date, end_date):
        return (df[date_column_name] >= pd.to_datetime(str(start_date))) & (df[date_column_name] < pd.to_datetime(str(end_date)))

    def GetDataFromAPIsToData(self):
        data_model_from_wba = self.GetDataFromWBA()
        data_model_from_imf = self.GetDataFromIMF(self.CURRENT_YEAR)

        countries = []

        countries_from_wba = data_model_from_wba.GetEnableCountries()
        countries_from_imf = data_model_from_imf.GetEnableCountries()

        for country_data_pair_imf in countries_from_imf:
            if country_data_pair_imf in countries_from_wba:
                countries.append(country_data_pair_imf)

        if(self.VERBOSE):
            print('Getting data from services')
            print('For countries: ', str.join(', ', [f'{country[1]} ({country[0]})' for country in countries]))

        data_from_imf = data_model_from_imf.GetDataFromIMFApi(countries)
        data_from_wba = data_model_from_wba.GetDataFromWorldBankApi(countries)

        data = data_from_imf
        data.extend(data_from_wba)

        self.df = pd.DataFrame(data).sort_values(['country', 'date'], ascending=True)
        self.countries = countries

    def GetDataFromBrentCsvToData(self):
        # If you haven't got file please download it from 'https://www.investing.com/commodities/brent-oil-historical-data' with max dates by weekly
        if(self.VERBOSE):
            print('Reading data from', self.BRENT_DATA_CSV_FILENAME)

        self.brent_data = pd.read_csv(self.BRENT_DATA_CSV_FILENAME)

        self.brent_data['Date'] = pd.to_datetime(self.brent_data['Date'])

        if(self.VERBOSE):
            print('Have read data from', self.BRENT_DATA_CSV_FILENAME)

    def SaveDataToCSV(self):
        if(self.VERBOSE):
            print('Exporitng to csv:', self.TRAIN_CSV_FILENAME)

        # Writing to csv
        self.df.to_csv(self.TRAIN_CSV_FILENAME, index=False)

        if(self.VERBOSE):
            print('Exported to csv:', self.TRAIN_CSV_FILENAME)
    
    def ReadDataFromCSV(self):
        if len(self.df) == 0:
            if os.path.exists(self.TRAIN_CSV_FILENAME):
                if(self.VERBOSE):
                    print('Importing data from csv')

                try:
                    self.df = pd.read_csv(self.TRAIN_CSV_FILENAME)

                    self.df['date'] = pd.to_datetime(self.df['date'].apply(str))

                    self.countries = self.df['country'].unique().tolist()

                    self.GetDataFromBrentCsvToData()

                    if(self.VERBOSE):
                        print('Data from csv is imported')
                except:
                    if(self.VERBOSE):
                        print('Importing from file is failed...')
            else:
                if(self.VERBOSE):
                    print('Dataset file is not exists')
                
                if(self.VERBOSE):
                    print('Start getting data from services')
                self.GetDataFromAPIsToData();
                self.SaveDataToCSV()

                if(self.VERBOSE):
                    print('Restarting process')
                os.execv(sys.executable, ['python'] + sys.argv)
    
    def ClearingDataModel(self):
        if(self.VERBOSE):
            print('Clearing data')
        df = self.df

        cols = df.columns
        if(self.VERBOSE):
            # определяем цвета 
            # желтый - пропущенные данные, синий - не пропущенные
            colours = ['#000099', '#ffff00'] 
            sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))

        col_missings = []
        for col in df.columns:
            pct_missing = np.mean(df[col].isnull())
            col_missings += [[col, pct_missing * 100]]
            if(self.VERBOSE):
                print('{} - {}%'.format(col, round(pct_missing*100)))

        # Отбрасываем признаки, отсутствующие больше определенного процента
        if(self.VERBOSE):
            print('Dropping columns equal or greater than ', self.PERCENT_FOR_DROP_COLUMN, 'percent')

        for col_missing in col_missings:
            if(col_missing[1] >= self.PERCENT_FOR_DROP_COLUMN):
                df = df.drop(col_missing[0], axis=1)
                if(self.VERBOSE):
                    print('Have dropped the column \'' + col_missing[0] + '\' with missing', col_missing[1])

        cols = df.columns

        # Filling empty cells by less than minimum number
        if(self.VERBOSE):
            print('Filling empty celling')
        for i in cols[3:]:
            df[i] = df[i].fillna(0)
        
        self.df = df
    
    def DisplayGraps(self):
        print('Displaying data model graps')

        mask_equal = lambda df, column_name, value: (df[column_name] == value)

        # For brent datas
        brent_figure = plt.figure()
        plt.title('Drent prices by years')
        plt.plot(self.brent_data['Date'], self.brent_data['Price'])

        # For df
        df_figures = []
        columns = self.df.columns.tolist()[3:]
        country_dfs = []

        for country_index in range(len(self.countries)):
            country_name = self.countries[country_index]

            country_df = self.df.loc[mask_equal(self.df, 'country', country_name)]
            
            country_dfs += [[country_name, country_df]]

        for column_index in range(len(columns)):
            column_name = columns[column_index]

            df_figures += [plt.figure()]

            plt.title(column_name)

            for country_index in range(len(country_dfs)):
                country_name = country_dfs[country_index][0]
                country_df = country_dfs[country_index][1]
                country_date = country_df['date']

                plt.plot(country_date, country_df[column_name], label=country_name)
            plt.legend()

        plt.show()
        

    def GetTrainTestXY(self, YEARS_BEFORE_BRENT_PRICES = 2, make_test_data = True, only_Price_column = False):
        brent_data = self.brent_data

        # Determine start year for brent prices
        brent_start_year = brent_data['Date'].min()
        brent_start_year = brent_start_year.year if brent_start_year.month == 1  else brent_start_year.year + 1 

        df = self.df.loc[self.MakeMaskForLoc(self.df, 'date', brent_start_year - YEARS_BEFORE_BRENT_PRICES, self.CURRENT_YEAR - YEARS_BEFORE_BRENT_PRICES)]

        ## TODO: Write code for cutting data from brent_data if brent_start_year - YEARS_BEFORE_BRENT_PRICES date doesn't exists in df 

        x_train = []
        y_train = []

        x_test = []
        y_test = []

        general_train_test_pair_count = (self.CURRENT_YEAR - brent_start_year) * 4
        test_pair_count = int(general_train_test_pair_count * 0.2)
        train_years_for_a_test = int((general_train_test_pair_count - test_pair_count) / test_pair_count)
        train_year_for_a_test_need_now = train_years_for_a_test

        for brent_year in range(brent_start_year, self.CURRENT_YEAR - YEARS_BEFORE_BRENT_PRICES + 1):
            # Making data for x
            df_start_year = brent_year - YEARS_BEFORE_BRENT_PRICES
            year_mask = self.MakeMaskForLoc(df, 'date', str(df_start_year), str(df_start_year + YEARS_BEFORE_BRENT_PRICES)) # range: [start_year, start_year + YEARS_BEFORE_BRENT_PRICES)

            df_year_array = df.loc[year_mask].drop(df.columns[[0, 1, 2]],axis=1).values.ravel()

            # Making data for y
            y_year_price_means = []
            y_year_open_means = []
            y_year_high_means = []
            y_year_low_means = []

            for brent_year_month in range(1, 13):
                year_month_str = str(brent_year) + '-' + str(brent_year_month)
                year_month_mask = self.MakeMaskForLoc(brent_data, 'Date', year_month_str, pd.to_datetime(year_month_str) + pd.DateOffset(months=1))

                year_month_brent_data = brent_data.loc[year_month_mask]

                y_year_price_means.extend([year_month_brent_data['Price'].mean()])
                y_year_open_means.extend([year_month_brent_data['Open'].mean()])
                y_year_high_means.extend([year_month_brent_data['High'].mean()])
                y_year_low_means.extend([year_month_brent_data['Low'].mean()])

            if (train_year_for_a_test_need_now == 0 and make_test_data):
                if(only_Price_column):
                    x_test.extend([df_year_array])
                    y_test.extend([y_year_price_means])
                else:
                    x_train.extend([df_year_array, df_year_array, df_year_array])
                    y_train.extend([y_year_open_means, y_year_high_means, y_year_low_means])
                    x_test.extend([df_year_array])
                    y_test.extend([y_year_price_means])

                train_year_for_a_test_need_now = train_years_for_a_test
            else:
                if(only_Price_column):
                    x_train.extend([df_year_array])
                    y_train.extend([y_year_price_means])
                else:
                    x_train.extend([df_year_array, df_year_array, df_year_array, df_year_array])
                    y_train.extend([y_year_price_means, y_year_open_means, y_year_high_means, y_year_low_means])

                train_year_for_a_test_need_now -= 1

        if(make_test_data):
            return x_train, x_test, y_train, y_test
        else:
            return x_train, y_train, [brent_start_year, self.CURRENT_YEAR - YEARS_BEFORE_BRENT_PRICES]

    def GetTrainTestXYForThisYear(self, YEARS_BEFORE_BRENT_PRICES):
        df = self.df.loc[self.MakeMaskForLoc(self.df, 'date', self.CURRENT_YEAR - YEARS_BEFORE_BRENT_PRICES, self.CURRENT_YEAR)]

        x_data = df.drop(df.columns[[0, 1, 2]],axis=1).values.ravel()

        y_data = []

        for brent_year_month in range(1, 13):
            year_month_str = str(self.CURRENT_YEAR) + '-' + str(brent_year_month)
            year_month_mask = self.MakeMaskForLoc(self.brent_data, 'Date', year_month_str, pd.to_datetime(year_month_str) + pd.DateOffset(months=1))

            year_month_brent_data = self.brent_data.loc[year_month_mask]

            y_data += [year_month_brent_data['Price'].mean()]

        return x_data, y_data # y_data - isn't full of data about all months of the current year


class DeepMindModel:
    def __init__(self, data_model, MY_MODEL_FILENAME = 'my_model.keras',EPOCHS = 200, OPTIMIZER = "adam", LOSS = "mse", METRICS = ['accuracy'], VERBOSE = 1, YEARS_BEFORE_BRENT_PRICES = 2):
        self.DATA_MODEL = data_model
        self.MY_MODEL_FILENAME = MY_MODEL_FILENAME
        self.EPOCHS = EPOCHS
        self.OPTIMIZER = OPTIMIZER
        self.LOSS = LOSS
        self.METRICS = METRICS
        self.VERBOSE = VERBOSE
        self.YEARS_BEFORE_BRENT_PRICES = YEARS_BEFORE_BRENT_PRICES
    
    def Get_x_y_koffs(self, x, y):
        get_max_num_2d_list = lambda list_2d: max([max(sub_list) for sub_list in list_2d])
        
        return get_max_num_2d_list(x), get_max_num_2d_list(y)

    def TeachModel(self, show_plot = True):
        # Разделяем данные на обучающую и тестовую выборки
        print('Teaching started')

        x_train, x_test, y_train, y_test = self.DATA_MODEL.GetTrainTestXY(YEARS_BEFORE_BRENT_PRICES=self.YEARS_BEFORE_BRENT_PRICES)

        x_koff, y_koff = self.Get_x_y_koffs(x_train + y_test, y_train + y_test)

        self.x_koff = x_koff
        self.y_koff = y_koff

        print(f'Calculated x and y koffs: x koff={self.x_koff}, y koff={self.y_koff}')

        x_train, y_train = x_train / x_koff, y_train / y_koff
        # x_test, y_test = x_test / x_koff, y_test / y_koff
        
        # Создаем модель
        c = len(x_train[0])
        cc = int(c / 2)

        model = keras.Sequential([
            keras.layers.Dense(c, activation="relu"),
            keras.layers.Dense(cc, activation="relu"),
            keras.layers.Dense(cc, activation="relu"),
            keras.layers.Dense(cc, activation="relu"),
            keras.layers.Dense(len(y_train[0]))
        ])

        # Компилируем модель
        if(self.VERBOSE):
            print('Компилиуем модель')
        model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)

        # Обучаем модель
        if(self.VERBOSE):
            print('Обучаем модель...')
        model.fit(x_train, y_train,
                  batch_size=20,
                  epochs=self.EPOCHS,
                  verbose=self.VERBOSE)

        # Оцениваем модель
        y_test = np.array([np.array(y_test_i) for y_test_i in y_test])        

        score = model.evaluate(x_test / x_koff, y_test / y_koff, verbose=self.VERBOSE)
        print("Точность модели:", score[1])

        self.model = model

        # Предсказываем выходные данные
        predicted_data = self.PredictData(x_test)

        # Строим график предсказанных и фактических данных
        if(show_plot):
            plt.plot((y_test * y_koff).ravel(), label="Фактические данные")
            plt.plot((predicted_data * y_koff).ravel(), label="Предсказанные данные")
            plt.plot((abs(predicted_data - y_test) * y_koff).ravel(), label="Разность между факт. и предск. данными")
            plt.legend()
            plt.show()

        return score[1]
    
    def SaveModel(self):
        self.model.save(self.MY_MODEL_FILENAME)

        print('Model have saved to', self.MY_MODEL_FILENAME)
    
    def LoadModel(self):
        self.model = keras.models.load_model(self.MY_MODEL_FILENAME)

        print('ML model loaded from', self.MY_MODEL_FILENAME)

        print('Calculating x and y koffs...')

        x_data, x_2_data, y_data, y_2_data = self.DATA_MODEL.GetTrainTestXY(YEARS_BEFORE_BRENT_PRICES=self.YEARS_BEFORE_BRENT_PRICES)

        self.x_koff, self.y_koff = self.Get_x_y_koffs(x_data + x_2_data, y_data + y_2_data)

        print(f'Calculating has finished: x koff = {self.x_koff}, y_koff = {self.y_koff}.')

    def PredictData(self, x_test):
        return self.model.predict(x_test / self.x_koff, self.YEARS_BEFORE_BRENT_PRICES) * self.y_koff # reshape(x_test, (-1, len(x_test[0][0]))) | x_test
    
    def PredictDataForEnableYears(self, only_Price_column = False):
        x_data, y_data, start_end_years_date = self.DATA_MODEL.GetTrainTestXY(YEARS_BEFORE_BRENT_PRICES=self.YEARS_BEFORE_BRENT_PRICES, make_test_data=False, only_Price_column=only_Price_column)

        predicted_data = self.PredictData(x_data)

        return {"predicted_data": predicted_data, 'actual_data': np.array([np.array(y_data_i) for y_data_i in y_data]), 'start_end_years_date': start_end_years_date}

    def PredictDataForCurrentYear(self):
        x_data, y_data = self.DATA_MODEL.GetTrainTestXYForThisYear(self.YEARS_BEFORE_BRENT_PRICES)

        predicted_data = self.PredictData(x_data)

        return {'predicted_data': predicted_data, 'actual_data': y_data}

    def DisplayPredictedActualGraph(self):
        predicted_actual = self.PredictDataForEnableYears(only_Price_column=True)

        predicted_data = predicted_actual['predicted_data']
        actual_data = predicted_actual['actual_data']
        start_end_years_date = predicted_actual['start_end_years_date']
        years_by_months = pd.date_range(start=datetime.date(start_end_years_date[0], 1, 1), end=datetime.date(start_end_years_date[1] + 1, 1, 1), freq='m')

        predicted_actual_figure = plt.figure()

        plt.plot(years_by_months, predicted_data.ravel(), label='predicted_data')
        plt.plot(years_by_months, actual_data.ravel(), label='actual_data')

        plt.legend()
        plt.show()
    
    def DisplayXYTestGraph(self):
        x_train, x_test, y_train, y_test = self.DATA_MODEL.GetTrainTestXY(YEARS_BEFORE_BRENT_PRICES=self.YEARS_BEFORE_BRENT_PRICES)

        y_test = np.array([np.array(y_test_i) for y_test_i in y_test])

        # Предсказываем выходные данные
        predicted_data = self.PredictData(x_test)

        # Строим график предсказанных и фактических данных
        plt.plot((y_test * self.y_koff).ravel(), label="Фактические данные")
        plt.plot((predicted_data * self.y_koff).ravel(), label="Предсказанные данные")
        plt.plot((abs(predicted_data - y_test) * self.y_koff).ravel(), label="Разность между факт. и предск. данными")
        plt.legend()
        plt.show()
        

class TestingDeepMindModels():
    def __init__(self, DATA_MODEL_VERBOSE = 1, ML_VERBOSE = 1):
        self.CSV_FILENAME = 'testing_result.csv'
        self.EPOCHS_ARR = [200, 100, 50]
        self.OPTIMIZERS = ['adam', 'rmsprop', 'adadelta', 'nadam']
        self.LOSSes = ["mse", 'mae']
        self.METRICS = [['accuracy']]
        self.DATA_MODEL_VERBOSE = DATA_MODEL_VERBOSE
        self.ML_VERBOSE = ML_VERBOSE
        self.YEARS_BEFORE_BRENT_PRICES = [3, 2, 1]

    def StartTesting(self):
        start = datetime.datetime.now()
        
        print('Testing started at', start)

        model_testing_data = []
        max_accuracy = 0.0

        data_model = DataModel(VERBOSE=self.DATA_MODEL_VERBOSE)

        data_model.ReadDataFromCSV()
        data_model.ClearingDataModel()


        for EPOCHS in self.EPOCHS_ARR:
            for YEARS_BEFORE_BRENT_PRICES in self.YEARS_BEFORE_BRENT_PRICES:
                for OPTIMIZER in self.OPTIMIZERS:
                    for LOSS in self.LOSSes:
                        for METRICS in self.METRICS:
                            print('Эпох:', EPOCHS, '. Optimizer:', OPTIMIZER, '. Loss:', LOSS, '. Metrics:', str(METRICS), '. Years_before_brent_prices: ', YEARS_BEFORE_BRENT_PRICES)

                            try:
                                deep_mind_model = DeepMindModel(data_model=data_model, EPOCHS=EPOCHS, OPTIMIZER=OPTIMIZER, LOSS=LOSS, METRICS=METRICS, VERBOSE=self.ML_VERBOSE, YEARS_BEFORE_BRENT_PRICES=YEARS_BEFORE_BRENT_PRICES)
                                
                                accuracy = deep_mind_model.TeachModel(show_plot=False)

                                if(accuracy > max_accuracy):
                                    # deep_mind_model.SaveModel()
                                    max_accuracy = accuracy

                                model_testing_data += [{'Точность модели': accuracy, 'Эпох': EPOCHS, 'optimizer': OPTIMIZER, 'loss': LOSS, 'metrics': str(METRICS), 'years_before_brent_prices': YEARS_BEFORE_BRENT_PRICES}]

                                print(model_testing_data[-1])
                            except Exception as err:
                                print('\nTesting error: ', err, '\n')
        
        df_testing = pd.DataFrame(model_testing_data)

        df_testing.to_csv(self.CSV_FILENAME)

        end = datetime.datetime.now()

        print('Testing has finished at', end)        

        print(f"Time taken in (dd:hh:mm:ss.ms) is {end - start}")


testing_deep_mind_models = TestingDeepMindModels(DATA_MODEL_VERBOSE=0,ML_VERBOSE=0)

testing_deep_mind_models.StartTesting()


'''
data_model = DataModel(VERBOSE=False)
data_model.ReadDataFromCSV()
data_model.ClearingDataModel()

#data_model.DisplayGraps()

dmm = DeepMindModel(data_model=data_model, MY_MODEL_FILENAME="my_model_2.keras", EPOCHS=200, OPTIMIZER='rmsprop', LOSS='mae', METRICS=['accuracy'], YEARS_BEFORE_BRENT_PRICES=1)

#dmm.TeachModel(show_plot=False)
#dmm.SaveModel()
dmm.LoadModel()

dmm.DisplayXYTestGraph()
#dmm.DisplayPredictedActualGraph()
'''