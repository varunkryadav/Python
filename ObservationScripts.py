#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:15:53 2019

@author: varunkumaryadav
"""

####PLEASE NOTE THAT BELOW SCRIPTS WERE MODIFIED TO RUN VARIOUS TESTS#####
import PricingLibrary2
import matplotlib.pyplot as plt
import numpy as np



#b = PricingFixedLookback(100,0.05,0.2,1,100000,1008)
#v = b.PutFixedStrike()
#v2 = b.CallFixedStrike()
#d = ClosedFormFixedLookbackCall(110,100,100,0.05,0.2,1)
#v3 = d.CallFixedStrike()
#e = ClosedFormFixedLookbackPut(100,100,100,0.05,0.2,1)
#v4 = e.PutFixedStrike()
#print(v,v2)
vac = np.array([])
vap = np.array([])
for x in np.arange(0.05,0.15,0.01):
    b = CloseFormBinary(100,100,x,0.2,1)
    v = b.CashOrNothingCall()
    v2 = b.CashOrNothingPut()
    vac = np.append(vac,v)
    vap = np.append(vap,v2)
c = np.arange(0.05,0.15,0.01)
plt.title('Cash or Nothing Closed Form Options')
plt.plot(c,vac,label = 'Closed Form Call')
plt.plot(c,vap,label = 'Closed Form Put')
plt.xlabel('Interest Rate of the Option')
plt.ylabel('Value of the Option')
plt.legend()
plt.show()




#a = PricingFloatingLookback(100,0.05,0.2,1,10000,252)
#value, totalputmatrix, minprice = a.CallFloatingStrike()
#b = ClosedFormFloatingLookbackCall(100,100,0.05,0.2,1)
#c = np.array([])
#d = np.array([])
#for x in range(200,10000,500):
#    a = PricingFixedLookback(100,0.05,0.2,1,x,500)
#    value,error = a.PutFixedStrike()
#    for i in range(x):
#        b = error[i]
#        c = np.append(c,ln(b))
#    d = np.append(d,np.std(c)/np.sqrt(x))

#v = np.array([])
#for x in range(40,200,5):
#    c = ClosedFormFixedLookbackPut(x,40,60,0.05,0.2,1)
#    v = np.append(v,c.PutFixedStrike())   
#a = range(40,200,5)
#plt.xlabel('Minimum Values of Fixed Put Lookback Option')
#plt.ylabel('Payoff of the option')
#plt.plot(a,v)

#c = PricingFloatingLookback(100,0.05,0.2,1,10000,252)
#v = c.CallFloatingStrike()
#d = PricingFloatingLookback(100,0.05,0.2,1,10000,252)
#e = d.PutFloatingStrike()
#print(v,e)


#c = SimulatedBinaryOption(100,100,0.05,0.2,1,800,1008)
#vm,v = c.AssetOrNothingPut()


#cash= CloseFormBinary(100,100,0.05,0.2,1)
#v1 = cash.AssetOrNothingCall()
#v = np.array([])
#for x in range (10000,20000,1000):
#    c = SimulatedBinaryOption(100,100,0.05,0.2,1,x,230)
#    v = np.append(v,c.AssetOrNothingCall())
#v2 = np.copy(v)
#v2.fill(v1)
#plt.xlabel('Number of simulations')
#plt.ylabel('Payoff of the Assetor Nothing Call option')
#x = range (10000,20000,1000)
#plt.plot(x,v2,label = 'Closed Form value')
#plt.plot(x,v, Label = 'Simulated Value')
#plt.legend()
#plt.label()

#for i in range(25600):
#    b = error[i]
#    c = np.append(c,ln(b))
#
#print(value)
#print(np.std(c)/np.sqrt(25600))


#b = np.array([])
#for x in range(40,160,10):
#    a = PricingFixedLookback(x,0.4,0.2,1,1000,252)
#    value = a.PutFixedStrike()
#    b = np.append(b,value[0])
#y = range(40,160,10)
#plt.ylabel('Fixed Put Option Price')
#plt.xlabel('Put Strike Values')
#plt.plot(y,b)

#plt.xlabel('Number of simulated steps')
#plt.ylabel('Payoff of the option')
#plt.plot(totalmatrix.T)


 