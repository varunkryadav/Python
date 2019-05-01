import random as r
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from babel.numbers import format_currency
from matplotlib.patches import Circle
import pandas as pd


def question1():
    ##Creates empty arrays for calculations##
    x = np.array([])
    y = np.array([])
    z = np.array([])
    nuport = np.array([])
    sigmaport = np.array([])
    ratio = np.array([])
    var = np.array([])
    esvar = np.array([])

    ##Below values are given in the question##
    sigma = [0.2, 0.15, 0.1]
    nu = [0.3, 0.2, 0.1]
    rho = [[1, -0.3, -0.5], [-0.3, 1, -0.6], [-0.5, -0.6, 1]]
    rho = np.array(rho).reshape(3, 3)
    s1 = np.diag([0.2, 0.15, 0.1])
    vcv = s1.dot(rho).dot(s1)

    ##Below code created random weights for the portfolio##
    for i in range(1000):
        a = r.uniform(0, 1)
        b = r.uniform(0, 1)
        c = r.uniform(0,1)
        su1 = a+b+c
        a /= su1
        b /= su1
        c /= su1
        x = np.append(x, a)
        y = np.append(y, b)
        z = np.append(z, c)

    ##Below code calculates Mean Portfolio, SD portfolio, VaR, ES VaR and Min Variance Portfolio
    for j in range(1000):
        XT = np.array([x[j], y[j], z[j]])
        nport = x[j] * nu[0] + y[j] * nu[1] + z[j] * nu[2]
        nuport = np.append(nuport, nport)
        varpart1 = np.dot(-XT.T, nu)
        varpart2 = norm.ppf(0.99) * np.sqrt(XT.T.dot(vcv).dot(XT))
        var = np.append(var, varpart1 + varpart2)
        espart2 = (np.sqrt(XT.T.dot(vcv).dot(XT)) / 0.01) * norm.pdf(norm.ppf(0.99))
        esvar = np.append(esvar, varpart1 + espart2)
        sp1 = np.sqrt(XT.T.dot(vcv).dot(XT))
        sigmaport = np.append(sigmaport, sp1)
        p = nport / sp1
        ratio = np.append(ratio, p)


    ##Below Code generates weights with minimum var and esvar
    h = np.argmin(var)
    minvarx = np.array([x[h], y[h], z[h]])
    e = np.argmin(esvar)
    minesvarx = np.array([x[e], y[e], z[e]])
    f = np.argmax(ratio)
    mvp = np.array(nuport[f])
    mvp = np.append(mvp,sigmaport[f])


    ##Finding eigenvalues for positive definite determination##
    eigenvalues = np.linalg.eigvals(rho)
    return nuport, sigmaport, mvp, minvarx, minesvarx, eigenvalues,rho


d1, d2, mvp, d3, d4, d5, rho = question1()


## The code below plots graph##
plt.style.use('seaborn')
plt.scatter(d2, d1,)
plt.xlabel('Portfolio Standard Deviation')
plt.ylabel('Portfolio Returns')
plt.xlim(0.0,0.2)
plt.ylim(0.0,0.3)
plt.title('Markowitz Portfolio Question 1a')
plt.show()

print('\n'+"The Answers to Question 1 are as below:")
## The code below gives the location of MVP##
print("The location of Minimum Variance Portfolio will have return: {} and the standard deviation {}".format(mvp[0],mvp[1]))
## The Code below gives the allocation with minimum VaR##
print("The weights with minimum VaR are:{} ".format(d3))
## The Code below gives the allocation with minimum ESVaR##
print("The weights with minimum ES VaR are:{} ".format(d4))
##Check if matrix is Positive Definite##
if all(i > 0 for i in d5):
    print("The Matrix is Positive Definite"),
else:
    print("The Matrix is not Position Definite")

# Drawing Gershgorin Circles##
fig = plt.figure()
ax = fig.add_subplot(111)

# Circle: |A[i,i]-z| <= sum(|A[i,j]| for j in range(n) and j != i)
for i in range(rho.shape[0]):
    real = rho[i, i].real    # each complex's real part
    imag = rho[i, i].imag    # each complex's image part

    # calculate the radius of each circle
    radius = -np.sqrt(rho[i, i].real**2+rho[i, i].imag**2)
    for j in range(rho.shape[0]):
        radius += np.sqrt(rho[i, j].real**2+rho[i, j].imag**2)

    # add the circle to the  figure and plot the center of the circle
    cir = Circle(xy = (real,imag), radius=radius, alpha=0.5, fill=False)
    ax.add_patch(cir)
    x, y = real, imag
    ax.plot(x, y, 'ro')
plt.title('Gershgorin Circles Question 1')
plt.show()

##Printing new lines for question ##
print('\n')

def question2():

    ##Below values given in the question##
    x2 = np.array([0.5,0.2,0.3])
    rho2 =([1,0.8,0.5],[0.8,1,0.3],[0.5,0.3,1])
    rho2 = np.array(rho2).reshape(3, 3)
    sigdiag = np.diag([0.3,0.2,0.15])
    vcv2 = sigdiag.dot(rho2).dot(sigdiag)
    varpart2q2 = np.array([])
    esvarpartq2 = np.array([])
    sigmaport2 = np.sqrt(x2.T.dot(vcv2).dot(x2))

    ##Below code calculates dVaR and dESVaR as specified in the question"
    for i in range(3):
        vartoppart = np.matmul(x2, vcv2)
        v2 = norm.ppf(0.99) * (vartoppart[i]/(sigmaport2)) ## Since return is a zero matrix, it has been excluded
        varpart2q2 = np.append(varpart2q2,v2)
        e2 = (vartoppart[i]/(0.01*sigmaport2))*norm.pdf(norm.ppf(0.99))
        esvarpartq2 = np.append(esvarpartq2,e2)
    return varpart2q2,esvarpartq2,rho2

a1,a2,a3 = question2()

print("The Answers to Question 2 are as below:")
##Sensitivites to Var##
print("The Sensitivities to VaR are:{}".format(a1))

##Sensitivites to Var##
print("The Sensitivities to Expected Shortfall VaR are: {}".format(a2))

##Eigenvalues of Rho##
print("The Eigenvalues of Rho are: {}".format(np.linalg.eigvals(a3)))


# Drawing Gershgorin Circles## THIS HAS BEEN EXCEPTIONALLY CHALLENGING IN PYTHON
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
a4 = np.array([[1,0.8,0.5],[0.8,1,0.3],[0.5,0.3,1]],dtype='complex')
# Circle: |A[i,i]-z| <= sum(|A[i,j]| for j in range(n) and j != i)
for i in range(a4.shape[0]):
    real = a4[i, i].real    # each complex's real part
    imag = a4[i, i].imag    # each complex's image part

    # calculate the radius of each circle
    radius = -np.sqrt(a4[i, i].real**2+a4[i, i].imag**2)
    for j in range(a4.shape[0]):
        radius += np.sqrt(a4[i, j].real**2+a4[i, j].imag**2)

    # add the circle to the  figure and plot the center of the circle
    cir1= Circle(xy = (real,imag), radius=radius, alpha=0.5, fill=False)
    ax1.add_patch(cir1)
    x, y = real, imag
    ax1.plot(x, y, 'ro')
plt.title('Gershgorin Circles Question 2')
plt.show()



def question5():
    ## Part A of question 5##
    Wealth1 = 16000000
    mu1 = 0.01
    alpha = norm.ppf(0.99)
    sd = 0.03
    muspread = 35*0.0001
    sigmaspread = 150*0.0001
    LVAR = (Wealth1*((alpha*sd)-mu1)+Wealth1*0.5*(muspread+(alpha*sigmaspread)))
    VAR1 = (Wealth1*((alpha*sd)-mu1))
    DeltaL1 = Wealth1*0.5*(muspread+(alpha*sigmaspread))
    VarPer1 = VAR1/LVAR
    DeltaPer1 = DeltaL1/LVAR


    ##Part B of the question 5##
    Wealth2 = 40000000
    sigport = 0.03
    spread = 55*0.0001
    spreadchanged = 255*0.0001
    alpha2 = norm.ppf(0.95)
    LVAR2 = (Wealth2*((alpha2*sigport)+0.5*spread))
    VAR2 = Wealth2*(alpha2*sigport)
    DeltaL2 = Wealth2*(0.5*spread)
    VarPer2 = VAR2/LVAR2
    DeltaPer2 = DeltaL2/LVAR2
    LVAR3 = (Wealth2*((alpha2*sigport)+0.5*spreadchanged))
    VarPer3 = VAR2/LVAR3
    DeltaL3 = Wealth2*(0.5*spreadchanged)
    DeltaPer3 = DeltaL3/LVAR3

    return LVAR,LVAR2,LVAR3,VAR1,VarPer1,DeltaL1,DeltaPer1,VAR2,VarPer2,DeltaL2,DeltaPer2,VarPer3,DeltaL3,DeltaPer3

LVAR,LVAR2,LVAR3,VAR1,VarPer1,DeltaL1,DeltaPer1,VAR2,VarPer2,DeltaL2,DeltaPer2,VarPer3,DeltaL3,DeltaPer3 = question5()
LVAR = format_currency(LVAR,'USD', locale='en_US')
VAR1 = format_currency(VAR1,'USD',locale='en_US')
DeltaL1 = format_currency(DeltaL1,'USD',locale='en_US')
LVAR2 = format_currency(LVAR2,'GBP',locale='en_GB')
VAR2 = format_currency(VAR2,'GBP',locale='en_GB')
DeltaL2 = format_currency(DeltaL2,'GBP',locale='en_GB')
LVAR3 = format_currency(LVAR3,'GBP',locale='en_GB')
DeltaL3 = format_currency(DeltaL3,'GBP',locale='en_GB')

print('\n'+"The Answers to question 5 are as below")
print("PARTA: The LVaR of the Portfolio is : {}. Out of which the VaR is: {} and it is {:.1%} of LVaR. \n The Spread is {} and it is {:.1%} of LVaR".format(LVAR,VAR2,VarPer1,DeltaL1,DeltaPer1))
print("PARTB: The LVaR of portfolio at 55 BPS Bid-Ask Spread is {} and the LVaR when the Spread widens to 255 BPS the LVAR changes to {}".format(LVAR2,LVAR3))
print("The VaR with 55 BPS Bid-Ask Spread is {} and it is {:.1%} of LVaR, When the Spread widens the VaR is lowered to {:.1%} of LVaR ".format(VAR2,VarPer2,VarPer3))
print("The Spread with 55 BPS is {} and it is {:.1%} of LVaR, When the Spread widens the Spread Changes to {} and it increases to {:.1%} of LVaR ".format(DeltaL2,DeltaPer2,DeltaL3,DeltaPer3))