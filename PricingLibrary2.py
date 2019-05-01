# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from math import log as ln


class pricing_bsm:
    def __init__(self, strike, spot, rate, sigma, time):
        self.strike = strike
        self.spot = spot
        self.rate = rate
        self.sigma = sigma
        self.time = time
        self.sigtime = self.sigma * np.sqrt(self.time)
        self.d1 = ((np.log(self.spot / self.strike)) + (self.rate + 0.5 * self.sigma ** 2) * self.time) / (self.sigtime)
        self.d2 = self.d1 - self.sigtime

    def bsm_call(self):
        value = self.spot * sp.norm.cdf(self.d1) - self.strike * np.exp(-self.rate * self.time) * sp.norm.cdf(self.d2)
        return value

    def bsm_put(self):
        value = -self.spot * sp.norm.cdf(-self.d1) + self.strike * np.exp(-self.rate * self.time) * sp.norm.cdf(
            -self.d2)
        return value


class PricingFloatingLookback:
    def __init__(self, spot, rate, sigma, time, sims, steps):
        self.spot = spot
        self.rate = rate
        self.sigma = sigma
        self.time = time
        self.sims = sims
        self.steps = steps+1
        self.dt = self.time / self.steps

    def CallFloatingStrike(self):

        SimPriceMin = np.array([])
        SimPriceAtMaturity = np.array([])
        total = np.zeros((self.sims,self.steps),float)
        callminimum = np.array([])
        for j in range(self.sims):
            pathwiseS= np.zeros((self.steps,),float)
            pathwiseS[0] =self.spot
            total[j,0] = self.spot
            for i in range(self.steps-1):
                phi = np.random.normal()
                pathwiseS[i+1] = pathwiseS[i]*(1+self.rate*self.dt+self.sigma*phi*np.sqrt(self.dt))
                total[j,i]= pathwiseS[i+1]            
            SimPriceMin = np.append(SimPriceMin, min(pathwiseS))
            SimPriceAtMaturity = np.append(SimPriceAtMaturity, pathwiseS[self.steps - 1])
            callminimum = np.append(callminimum,max( np.average(SimPriceAtMaturity)- np.average(SimPriceMin), 0))

            

        call = np.average(callminimum[1:self.sims])*np.exp(-self.rate*self.time)
        return call #total.reshape(self.sims, self.steps), SimPriceMin

    def PutFloatingStrike(self):

#        SimPriceMax = np.array([])
#        SimPriceAtMaturity = np.array([])
        put2 = np.array([])
        total = np.zeros((self.sims,self.steps),float)     
        for j in range(self.sims):
            pathwiseS= np.zeros((self.steps,),float)
            pathwiseS[0] =self.spot
            total[j,0] = self.spot
            for i in range(self.steps-1):
                phi = np.random.normal()
                pathwiseS[i+1] = pathwiseS[i]*(1+self.rate*self.dt+self.sigma*phi*np.sqrt(self.dt))
                total[j,i]= pathwiseS[i+1]
            
#            SimPriceAtMaturity = np.append(SimPriceAtMaturity, pathwiseS[self.steps - 1])
            put2 = np.append(put2,max(pathwiseS)-pathwiseS[self.steps - 1])
#            SimPriceMax = np.append(SimPriceMax, max(pathwiseS))
            

        put = np.average(put2)*np.exp(-self.rate*self.time)
        return put #total.reshape(self.sims, self.steps), SimPriceMax

class PricingFixedLookback:
    def __init__(self, strike, rate, sigma, time, sims, steps):
        self.strike = strike
        self.rate = rate
        self.sigma = sigma
        self.time = time
        self.sims = sims
        self.steps = steps+1
        self.dt = self.time / self.steps

    def CallFixedStrike(self):

#        SimPriceMax = np.array([])
#        SimErrorCheck = np.array([])
        call2 = np.array([])
        total = np.zeros((self.sims,self.steps),float)
        pathwiseS= np.zeros((self.steps,),float)
        for j in range(self.sims):
            pathwiseS[0] =self.strike
            total[j,0] = self.strike
            for i in range(self.steps-1):
                phi = np.random.normal()
                pathwiseS[i+1] = pathwiseS[i]*(1+self.rate*self.dt+self.sigma*phi*np.sqrt(self.dt))
                total[j,i]= pathwiseS[i+1]
                
            call2 = np.append(call2,max(max(pathwiseS)-self.strike,0))
#            SimErrorCheck = np.append(SimErrorCheck,pathwiseS[self.steps - 1])
#            SimPriceMax = np.append(SimPriceMax, max(pathwiseS))

        call = np.average(call2)*np.exp(-self.rate*self.time)
        return call#, SimErrorCheck #total.reshape(self.sims, self.steps), SimPriceMax

    def PutFixedStrike(self):
#        SimErrorCheck = np.array([])
#        SimPriceMin = np.array([])
        put2 = np.array([])
        total = np.zeros((self.sims,self.steps),float)
        pathwiseS= np.zeros((self.steps,),float)
        for j in range(self.sims):
            pathwiseS[0] =self.strike
            total[j,0] = self.strike
            for i in range(self.steps-1):
                phi = np.random.normal()
                pathwiseS[i+1] = pathwiseS[i]*(1+self.rate*self.dt+self.sigma*phi*np.sqrt(self.dt))
                total[j,i]= pathwiseS[i+1]
                
#            SimErrorCheck = np.append(SimErrorCheck,pathwiseS[self.steps - 1])
            put2 = np.append(put2,max(self.strike-min(pathwiseS),0))
#            SimPriceMin = np.append(SimPriceMin, min(pathwiseS))


        put = np.average(put2)*np.exp(-self.rate*self.time)
        return put#, SimErrorCheck#total.reshape(self.sims, self.steps), SimPriceMin

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

class SimulatedBinaryOption:
    def __init__(self, strike, spot, rate, sigma, time,sims,steps):
        self.strike = strike
        self.spot = spot
        self.rate = rate
        self.sigma = sigma
        self.time = time
        self.sims = sims
        self.steps = steps+1
        self.dt = self.time / self.steps
    

    def CashOrNothingCall(self):

        SimPriceAtMaturity = np.array([])
        call2 = np.array([])
        pathwiseS= np.zeros((self.steps,),float)
        total = np.zeros((self.sims,self.steps),float)       
        for j in range(self.sims):
            pathwiseS[0] =self.spot
            total[j,0] = self.spot
            for i in range(self.steps-1):
                phi = np.random.normal()
                pathwiseS[i+1] = pathwiseS[i]*(1+self.rate*self.dt+self.sigma*phi*np.sqrt(self.dt))
                total[j,i]= pathwiseS[i+1]
            
            SimPriceAtMaturity = np.append(SimPriceAtMaturity, pathwiseS[self.steps - 1])
            if pathwiseS[self.steps - 1]>self.strike:
                call2 = np.append(call2,1)
            else:
                call2 = np.append(call2,0)


        callbsm = np.average(call2)
        return  callbsm*np.exp(-self.rate*self.time) #total.reshape(self.sims, self.steps)
    
    def CashOrNothingPut(self):

        SimPriceAtMaturity = np.array([])
        put2 = np.array([])
        pathwiseS= np.zeros((self.steps,),float)
        total = np.zeros((self.sims,self.steps),float)       
        for j in range(self.sims):
            pathwiseS[0] =self.spot
            total[j,0] = self.spot
            for i in range(self.steps-1):
                phi = np.random.normal()
                pathwiseS[i+1] = pathwiseS[i]*(1+self.rate*self.dt+self.sigma*phi*np.sqrt(self.dt))
                total[j,i]= pathwiseS[i+1]
            
            SimPriceAtMaturity = np.append(SimPriceAtMaturity, pathwiseS[self.steps - 1])
            if pathwiseS[self.steps - 1]<self.strike:
                put2 = np.append(put2,1)
            else:
                put2 = np.append(put2,0)


        putbinary = np.average(put2)
        return  putbinary*np.exp(-self.rate*self.time) #total.reshape(self.sims, self.steps)


    def AssetOrNothingCall(self):

        SimPriceAtMaturity = np.array([])
        call2 = np.array([])
        pathwiseS= np.zeros((self.steps,),float)
        total = np.zeros((self.sims,self.steps),float)       
        for j in range(self.sims):
            pathwiseS[0] =self.spot
            total[j,0] = self.spot
            for i in range(self.steps-1):
                phi = np.random.normal()
                pathwiseS[i+1] = pathwiseS[i]*(1+self.rate*self.dt+self.sigma*phi*np.sqrt(self.dt))
                total[j,i]= pathwiseS[i+1]
            
            SimPriceAtMaturity = np.append(SimPriceAtMaturity, pathwiseS[self.steps - 1])
            if pathwiseS[self.steps - 1]>self.strike:
                call2 = np.append(call2,pathwiseS[self.steps - 1])
            else:
                call2 = np.append(call2,0)


        callbsm = np.average(call2)
        return callbsm*np.exp(-self.rate*self.time) #, total.reshape(self.sims, self.steps)
    
    def AssetOrNothingPut(self):

        SimPriceAtMaturity = np.array([])
        put2 = np.array([])
        pathwiseS= np.zeros((self.steps,),float)
        total = np.zeros((self.sims,self.steps),float)       
        for j in range(self.sims):
            pathwiseS[0] =self.spot
            total[j,0] = self.spot
            for i in range(self.steps-1):
                phi = np.random.normal()
                pathwiseS[i+1] = pathwiseS[i]*(1+self.rate*self.dt+self.sigma*phi*np.sqrt(self.dt))
                total[j,i]= pathwiseS[i+1]
            
            SimPriceAtMaturity = np.append(SimPriceAtMaturity, pathwiseS[self.steps - 1])
            if pathwiseS[self.steps - 1]<self.strike:
                put2 = np.append(put2,pathwiseS[self.steps - 1])
            else:
                put2 = np.append(put2,0)


        putbinary = np.average(put2)
        return putbinary*np.exp(-self.rate*self.time) #,total.reshape(self.sims, self.steps)

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
    

