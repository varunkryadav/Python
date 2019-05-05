#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:48:42 2019

@author: varunkumaryadav
"""
import numpy as np
import scipy.stats as sp


class PricingSimulatedLookback:
    def __init__(self, spot, strike,rate, sigma, time, sims, steps):
        self.spot = spot
        self.strike = strike
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

    def CallFloatingStrike(self):
        
        getpayoff = self.Simulations()
        minprice = np.zeros((self.sims),float)
        priceatmaturity = np.zeros((self.sims),float)
        callpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            minprice[j] = min(getpayoff[j,])
            priceatmaturity[j] = getpayoff[j,self.steps-1]
            callpayoff[j] = max(priceatmaturity[j]-minprice[j],0)
        
        return np.exp(-self.rate*self.time)*np.average(callpayoff)

    def PutFloatingStrike(self):
        
        getpayoff = self.Simulations()
        maxprice = np.zeros((self.sims),float)
        priceatmaturity = np.zeros((self.sims),float)
        Putpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            maxprice[j] = max(getpayoff[j,])
            priceatmaturity[j] = getpayoff[j,self.steps-1]
            Putpayoff[j] = max(maxprice[j]-priceatmaturity[j],0)
        
        return np.exp(-self.rate*self.time)*np.average(Putpayoff)
    
    def CallFixedStrike(self):
        
        getpayoff = self.Simulations()
        maxprice = np.zeros((self.sims),float)
        callpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            maxprice[j] = max(getpayoff[j,])
            callpayoff[j] = max(maxprice[j]-self.strike,0)
        
        return np.exp(-self.rate*self.time)*np.average(callpayoff)
    
    def PutFixedStrike(self):
        
        getpayoff = self.Simulations()
        minprice = np.zeros((self.sims),float)
        Putpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            minprice[j] = min(getpayoff[j,])
            Putpayoff[j] = max(self.strike-minprice[j],0)
        
        return np.exp(-self.rate*self.time)*np.average(Putpayoff)


class ClosedFormFloatingLookbackCall:
    def __init__(self,  smin, spot, rate, sigma, time):
        self.spot = spot
        self.rate = rate
        self.sigma = sigma
        self.time = time
        self.smin = smin
        self.sigtime = self.sigma * np.sqrt(self.time)
        self.d1 = ((np.log(self.spot / self.smin)) + (self.rate + 0.5 * self.sigma ** 2) * self.time) / (self.sigtime)
        self.d2 = self.d1 - self.sigtime
        self.sigrinv = -2*self.rate/(self.sigma**2)
        self.sigr = (self.sigma*self.sigma)/(2*self.rate)
        self.sigrdt = 2*self.rate*np.sqrt(self.time)/self.sigma

    def CallFloatingStrike(self):
        value = self.spot*sp.norm.cdf(self.d1)-self.smin*np.exp(-self.rate*self.time)*sp.norm.cdf(self.d2)+\
                self.spot*np.exp(-self.rate*self.time)*self.sigr*\
                (((self.spot/self.smin)**self.sigrinv)*sp.norm.cdf(-self.d1 + self.sigrdt)-np.exp(self.rate*self.time)*
                 sp.norm.cdf(-self.d1))
        return value
    
class ClosedFormFloatingLookbackPut:
    def __init__(self,  smax, spot, rate, sigma, time):
        self.spot = spot
        self.rate = rate
        self.sigma = sigma
        self.time = time
        self.smax = smax
        self.sigtime = self.sigma * np.sqrt(self.time)
        self.d1 = ((np.log(self.spot / self.smax)) + (self.rate + 0.5 * self.sigma ** 2) * self.time) / (self.sigtime)
        self.d2 = self.d1 - self.sigtime
        self.sigrinv = -2*self.rate/(self.sigma**2)
        self.sigr = (self.sigma**2)/(2*self.rate)
        self.sigrdt = 2*self.rate*np.sqrt(self.time)/self.sigma

    def PutFloatingStrike(self):
        value = -self.spot*sp.norm.cdf(-self.d1)+self.smax*np.exp(-self.rate*self.time)*sp.norm.cdf(-self.d2)+\
                self.spot*np.exp(-self.rate*self.time)*self.sigr*\
                (-(self.spot/self.smax)**self.sigrinv*sp.norm.cdf(self.d1 - self.sigrdt)+np.exp(self.rate*self.time)*
                 sp.norm.cdf(self.d1))
        return value


class ClosedFormFixedLookbackCall:
    def __init__(self,  smax, spot, strike,rate, sigma, time):
        self.spot = spot
        self.rate = rate
        self.strike = strike
        self.sigma = sigma
        self.time = time
        self.smax = smax
        self.sigtime = self.sigma * np.sqrt(self.time)
        self.d1 = ((np.log(self.spot / self.smax)) + (self.rate + 0.5 * self.sigma ** 2) * self.time) / (self.sigtime)
        self.d2 = self.d1 - self.sigtime
        self.sigrinv = -2*self.rate/(self.sigma**2)
        self.sigr = (self.sigma**2)/(2*self.rate)
        self.sigrdt = 2*self.rate*np.sqrt(self.time)/self.sigma
        
    def CallFixedStrike(self):
        if self.strike > self.smax :
            value = self.spot*sp.norm.cdf(self.d1) - self.strike*np.exp(-self.rate*self.time)*sp.norm.cdf(self.d2)+\
            self.spot*np.exp(-self.rate*self.time)*self.sigr*\
            (-(self.spot/self.strike)**self.sigrinv*sp.norm.cdf(self.d1-self.sigrdt)+\
             np.exp(self.rate*self.time)*sp.norm.cdf(self.d1))
            return value
        else:
            value = (self.smax - self.strike)*np.exp(self.rate*self.time) + self.spot*sp.norm.cdf(self.d1)-\
            self.smax*np.exp(-self.rate*self.time)*sp.norm.cdf(self.d2)+\
            self.spot*np.exp(-self.rate*self.time)*self.sigr*\
            (-(self.spot/self.smax)**self.sigrinv*sp.norm.cdf(self.d1-self.sigrdt)+\
             np.exp(self.rate*self.time)*sp.norm.cdf(self.d1))
            return value

class ClosedFormFixedLookbackPut:
    def __init__(self,  smin, spot, strike,rate, sigma, time):
        self.spot = spot
        self.rate = rate
        self.strike = strike
        self.sigma = sigma
        self.time = time
        self.smin = smin
        self.sigtime = self.sigma * np.sqrt(self.time)
        self.d1 = ((np.log(self.spot / self.smin)) + (self.rate + 0.5 * self.sigma ** 2) * self.time) / (self.sigtime)
        self.d2 = self.d1 - self.sigtime
        self.sigrinv = -2*self.rate/(self.sigma**2)
        self.sigr = (self.sigma**2)/(2*self.rate)
        self.sigrdt = 2*self.rate*np.sqrt(self.time)/self.sigma
        
    def PutFixedStrike(self):
        if self.strike < self.smin :
            value = -self.spot*sp.norm.cdf(-self.d1) + self.strike*np.exp(-self.rate*self.time)*sp.norm.cdf(-self.d2)+\
            self.spot*np.exp(-self.rate*self.time)*self.sigr*\
            ((self.spot/self.strike)**self.sigrinv*sp.norm.cdf(-self.d1+self.sigrdt)-\
             np.exp(self.rate*self.time)*sp.norm.cdf(-self.d1))
            return value
        else:
            value = (self.smin - self.strike)*np.exp(self.rate*self.time) - self.spot*sp.norm.cdf(-self.d1)+\
            self.smin*np.exp(-self.rate*self.time)*sp.norm.cdf(-self.d2)+\
            self.spot*np.exp(-self.rate*self.time)*self.sigr*\
            ((self.spot/self.smin)**self.sigrinv*sp.norm.cdf(-self.d1+self.sigrdt)-\
             np.exp(self.rate*self.time)*sp.norm.cdf(-self.d1))
            return value
        