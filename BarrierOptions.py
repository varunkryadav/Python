#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:01:48 2019

@author: varunkumaryadav
"""

import numpy as np

class PricingSimulatedBarrierOption:
    def __init__(self, spot, strike, barrier, rate, sigma, time, sims, steps):
        self.spot = spot
        self.strike = strike
        self.barrier = barrier
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

    def CallUpAndOut(self):
        
        getpayoff = self.Simulations()
        callpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if max(getpayoff[j,])>=self.barrier:
                callpayoff[j] = 0
            else:
                callpayoff[j] = max(getpayoff[j,self.steps-1]-self.strike,0)  
        
        return np.exp(-self.rate*self.time)*np.average(callpayoff)

    def CallDownAndOut(self):
        
        getpayoff = self.Simulations()
        callpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if min(getpayoff[j,])<=self.barrier:
                callpayoff[j] = 0
            else:
                callpayoff[j] = max(getpayoff[j,self.steps-1]-self.strike,0)
        
        return np.exp(-self.rate*self.time)*np.average(callpayoff)
      
    def PutUpAndOut(self):
        
        getpayoff = self.Simulations()
        Putpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if max(getpayoff[j,])>=self.barrier:
                Putpayoff[j] = 0
            else:
                Putpayoff[j] = max(self.strike-getpayoff[j,self.steps-1],0)  
        
        return np.exp(-self.rate*self.time)*np.average(Putpayoff)

    def PutDownAndOut(self):
        
        getpayoff = self.Simulations()
        Putpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if min(getpayoff[j,])<=self.barrier:
                Putpayoff[j] = 0
            else:
                Putpayoff[j] = max(self.strikegetpayoff-[j,self.steps-1],0)
        
        return np.exp(-self.rate*self.time)*np.average(Putpayoff)
    
    def CallUpAndIn(self):
        
        getpayoff = self.Simulations()
        callpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if max(getpayoff[j,])>=self.barrier:
                callpayoff[j] = max(getpayoff[j,self.steps-1]-self.strike,0)
            else:
                callpayoff[j] = 0  
        
        return np.exp(-self.rate*self.time)*np.average(callpayoff)

    def CallDownAndIn(self):
        
        getpayoff = self.Simulations()
        callpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if min(getpayoff[j,])<=self.barrier:
                callpayoff[j] = max(getpayoff[j,self.steps-1]-self.strike,0)
            else:
                callpayoff[j] = 0
        
        return np.exp(-self.rate*self.time)*np.average(callpayoff)
      
    def PutUpAndIn(self):
        
        getpayoff = self.Simulations()
        Putpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if max(getpayoff[j,])>=self.barrier:
                Putpayoff[j] = max(self.strike-getpayoff[j,self.steps-1],0)
            else:
                Putpayoff[j] = 0  
        
        return np.exp(-self.rate*self.time)*np.average(Putpayoff)

    def PutDownAndIn(self):
        
        getpayoff = self.Simulations()
        Putpayoff = np.zeros((self.sims),float)
        for j in range(self.sims):
            if min(getpayoff[j,])<=self.barrier:
                Putpayoff[j] = max(self.strikegetpayoff-[j,self.steps-1],0)
            else:
                Putpayoff[j] = 0
        
        return np.exp(-self.rate*self.time)*np.average(Putpayoff)







#c = PricingSimulatedBarrierOption(100,100,170,0.05,0.2,1,100,252)
##print(c.CallUpAndOut())
##print(c.CallDownAndOut())
#a,b,d = c.CallDownAndOut()
    
#print(t)
#print(max(t[1,]))
