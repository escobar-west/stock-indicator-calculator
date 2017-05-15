#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 15:39:53 2017

@author: EscobarWest
"""
import datetime
import pandas_datareader.data as web
import numpy as np

class StockData:
    """
    members:
        self.symbol
        self.start
        self.end
        self.std_window
        self.alpha
        self.horizon
        self.df
    methods:
        none
    """
    def __init__(self,
                 symbol     = 'SPY',
                 start      = datetime.datetime(2017, 1, 1),
                 end        = datetime.datetime.today(),
                 std_window = 5,
                 alpha      = 1,
                 horizon    = 1):
        
        ########## pandas datareader ##########
        self.symbol = symbol
        self.start = start
        self.end = end
        self.std_window = std_window
        self.alpha = alpha
        self.horizon = horizon
        
        self.df = web.DataReader(symbol,
            'google', start, end)[['Close', 'High', 'Low', 'Volume']]
                                
        self.df.Close = np.log(self.df.Close)
        self.df.High  = np.log(self.df.High)
        self.df.Low   = np.log(self.df.Low)
        
        self.df['Diff'] = self.df.Close.diff()
        
        self.df['Rolling_STD'] = self.df.Diff.rolling(window=std_window).std()
        
        self.df['GLD_Diff'] = np.log(web.DataReader('GLD',
            'google', start, end)['Close']).diff()

        if symbol != 'SPY':
            self.df['SPY_Diff'] = np.log(web.DataReader('SPY',
                'google', start, end)['Close']).diff()
                
        ########## MACD ##########
        EMA12 = [np.NaN] * 11
        EMA12.append( self.df.Close[:12].mean() )
        
        multiplier = 2/13
        
        for x in self.df.Close[12:]:
            EMA12.append( multiplier*x + (1-multiplier)*EMA12[-1] )
            
        EMA26 = [np.NaN] * 25
        EMA26.append( self.df.Close[:26].mean() )
        
        multiplier = 2/27
        
        for x in self.df.Close[26:]:
            EMA26.append( multiplier*x + (1-multiplier)*EMA26[-1] )
            
        self.df['MACD'] = np.array(EMA12) - np.array(EMA26)
        
        MACD = self.df.MACD.dropna()
        MACD_EMA = [np.NaN] * 33
        MACD_EMA.append(MACD[:9].mean())
        
        multiplier = 2/10
        
        for x in MACD[9:]:
            MACD_EMA.append( multiplier*x + (1-multiplier)*MACD_EMA[-1] )
            
        self.df['MACD_Signal'] = self.df.MACD.as_matrix() - MACD_EMA

        ########## CMF ##########
        self.df['MFM'] = ( (2*self.df.Close - self.df.High - self.df.Low) /
            (self.df.High-self.df.Low) )
        
        self.df['MFV'] = self.df.MFM * self.df.Volume

        CMF = [np.NaN] * 19

        for i in range(20, len(self.df)+1):
            CMF.append( self.df.MFV[i-20:i].sum() / self.df.Volume[i-20:i].sum() )
        
        self.df['CMF'] = CMF

        ######### Target #########
        weights = [alpha] ** np.arange(0, horizon)
        weights =  weights/weights.sum()
        
        target = []

        for i in range(1, len(self.df)-horizon+1):
            target.append( (weights*self.df.Diff[i:i+horizon]).sum() )
    
        target = target + [np.NaN] * (horizon)
        
        self.df['Target'] = target

        ######### Clean up #########        
        self.df.drop(['Close', 'High', 'Low', 'MFM', 'MFV'],inplace=True,axis=1)