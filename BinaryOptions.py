#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:05:22 2019

@author: varunkumaryadav
"""
import numpy as np
import scipy.stats as sp


class SimulatedBinaryOption:
    def __init__(self, strike, spot, rate, sigma, time,sims,steps):
        self.strike = strike
        self.spot = spot
        self.rate = rate
        self.sigma = sigma
        self.time = time
        self.sims = sims
        self.steps = steps
        self.dt = self.time / self.steps

    def Simulations(self):
        
        total = np.zeros((self.sims,self.steps+1),float)
        pathwiseS= np.zeros((self.steps+1),float)
        for j in range(self.sims):
            pathwiseS[0] =self.spot
            total[j,0] = self.spot
            for i in range(1,self.steps+1):
                phi = np.random.normal()
                pathwiseS[i] = pathwiseS[i-1]*(1+self.rate*self.dt+self.sigma*phi*np.sqrt(self.dt))
                total[j,i]= pathwiseS[i]
            
        return total.reshape(self.sims, self.steps+1)
    
    def CashOrNothingCall(self):
        
        getpayoff = self.Simulations()
        callpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if getpayoff[j,self.steps-1]>self.strike:
                callpayoff[j] = 1
            else:
                callpayoff[j] = 0
                
        return np.exp(-self.rate*self.time)*np.average(callpayoff)
    

    def CashOrNothingPut(self):
        
        getpayoff = self.Simulations()
        Putpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if getpayoff[j,self.steps-1]<self.strike:
                Putpayoff[j] = 1
            else:
                Putpayoff[j] = 0
                
        return np.exp(-self.rate*self.time)*np.average(Putpayoff)
    
    def AssetOrNothingCall(self):
        
        getpayoff = self.Simulations()
        callpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if getpayoff[j,self.steps-1]>self.strike:
                callpayoff[j] = getpayoff[j,self.steps-1]
            else:
                callpayoff[j] = 0
                
        return np.exp(-self.rate*self.time)*np.average(callpayoff)
    

    def AssetOrNothingPut(self):
        
        getpayoff = self.Simulations()
        Putpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if getpayoff[j,self.steps-1]<self.strike:
                Putpayoff[j] = getpayoff[j,self.steps-1]
            else:
                Putpayoff[j] = 0
                
        return np.exp(-self.rate*self.time)*np.average(Putpayoff)


class CloseFormBinary:
    def __init__(self, strike, spot, rate, sigma, time):
        self.strike = strike
        self.spot = spot
        self.rate = rate
        self.sigma = sigma
        self.time = time
        self.sigtime = self.sigma * np.sqrt(self.time)
        self.d1 = ((np.log(self.spot / self.strike)) + (self.rate + 0.5 * self.sigma ** 2) * self.time) / (self.sigtime)
        self.d2 = self.d1 - self.sigtime

    def CashOrNothingCall(self):
        value = np.exp(-self.rate * self.time) * sp.norm.cdf(self.d2)
        return value

    def CashOrNothingPut(self):
        value = np.exp(-self.rate * self.time) * sp.norm.cdf(-self.d2)
        return value
    
    def AssetOrNothingCall(self):
        value = self.spot * sp.norm.cdf(self.d1)
        return value
    
    def AssetOrNothingPut(self):
        value = self.spot * sp.norm.cdf(-self.d1)
        return value