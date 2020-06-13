#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:40:23 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""
import numpy as np
#from testConnectivity import lps_csd
from dyconnmap.fc import corr, coherence, pli, plv
import matplotlib.pyplot as plt
from dyconnmap.analytic_signal import analytic_signal
#import mlab_Fanny as mlab
from scipy import signal
import matplotlib.mlab as mlab
import seaborn as sns

# {}
# []

##############################################################################
def lps_plv(data, fb, fs):
    _, u_phase = analytic_signal(data, fb, fs) 
    n_channels, _ = np.shape(np.array(data))
    pairs = [(r2, r1) for r1 in range(n_channels) for r2 in range(r1)]
    avg = np.zeros((n_channels, n_channels)) 
    for pair in pairs:
        u1, u2 = u_phase[pair,]
        ts_plv = np.exp(1j * (u1-u2))
        #pdb.set_trace()
        #avg_plv = np.abs(np.sum(ts_plv)) / float(label_ts.shape[1])
        r = np.sum(ts_plv) / float(np.array(data).shape[1])
        num = np.power(np.imag(r), 2)
        denom = 1-np.power(np.real(r), 2)
        #pdb.set_trace()
        if denom == 0:
            avg[pair] = 0
        else:
            avg[pair] = num/denom
    return avg


#%%##########################################################################
## Sine waves
#############################################################################
t = np.linspace(0,4*np.pi,200)
A = 1
f = 1
phiA = 0
phiB = 0
phiC = 0

sigA = A*np.sin(f*t + phiA)
sigB = 2*np.sin(f*t + phiB)
sigC = A*np.sin(f*t + phiC) + 1/10*np.sin(4*t + phiC)
#sigD = A*np.sin(f*t*1.5 + phiD)

dat = [sigA,sigB,sigC] #,sigD]

_, PLI = pli(dat, fb=None, fs=None, pairs=None)
_, PLV = plv(dat, fb=None, fs=None, pairs=None)
LPS = lps_plv(dat, fb=None, fs=None)

print(PLI)
print('\n')
print(PLV)
print('\n')
print(LPS)



#%%
fig = plt.figure(figsize=(11,6))
sns.set(font_scale=1.2)

sns.lineplot(t,sigA)
sns.lineplot(t,sigB)
sns.lineplot(t,sigC)

x_tick = np.linspace(0,4*np.pi, 5)
x_ticklabels = [r"0", r"$\pi$", r"2$\pi$", r"$3\pi$", r"$4\pi$"]
#y_tick = np.arange(0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi)

plt.xlabel('Radians')
plt.ylabel('Amplitude')
plt.legend(['Signal A', 'Signal B', 'Signal C'], loc = 'upper right') #, 'signal D'])
plt.xticks(x_tick, x_ticklabels)
#plt.xticklabels(x_ticklabels)

plt.show()
fig.savefig('/share/FannyMaster/PythonNew/Figures/sineWaves.jpg', bbox_inches = 'tight')
sns.reset_orig()

#%%##########################################################################
## Moving phase in A
#############################################################################

t = np.linspace(0,4*np.pi,200)
A = 1
f = 1
phiA = 0
phiB =0 
phiC = 0

sigA = A*np.sin(f*t + phiA)
sigB = 2*np.sin(f*t + phiB)
sigC = A*np.sin(f*t + phiC) + 1/8*np.sin(10*t + phiC)

phiInterval = np.concatenate((np.linspace(0,np.pi,50), np.linspace(np.pi,2*np.pi,50)))
pliAB = []
pliAC = []
plvAB = []
plvAC = []
lpsAB = []
lpsAC = []
corAB = []
corAC = []
for i, phi in enumerate(phiInterval):
    sigA1 = A*np.sin(f*t + phi)
    
    dat = [sigA1,sigB,sigC]

    _, PLI = pli(dat, fb=None, fs=None, pairs=None)
    _, PLV = plv(dat, fb=None, fs=None, pairs=None)
    LPS = lps_plv(dat, fb=None, fs=None)
    cor = corr(dat)
    
    pliAB.append(PLI[0,1])
    pliAC.append(PLI[0,2])
    plvAB.append(PLV[0,1])
    plvAC.append(PLV[0,2])
    lpsAB.append(LPS[0,1])
    lpsAC.append(LPS[0,2])
    corAB.append(np.abs(cor[0,1]))
    corAC.append(np.abs(cor[0,2]))
    



#%%##########################################################################
## Moving phase in C
#############################################################################

pliCB = []
plvCB = []
lpsCB = []
corCB = []
for i, phi in enumerate(phiInterval):
    sigB1 = 2*np.sin(f*t + phi)
    
    dat = [sigC,sigA,sigB1]

    _, PLI = pli(dat, fb=None, fs=None, pairs=None)
    _, PLV = plv(dat, fb=None, fs=None, pairs=None)
    LPS = lps_plv(dat, fb=None, fs=None)
    cor = corr(dat)
    
    pliCB.append(PLI[0,2])
    plvCB.append(PLV[0,2])
    lpsCB.append(LPS[0,2])
    corCB.append(np.abs(cor[0,2]))
 
#%%

sns.set(font_scale=2.3)
fig, ax = plt.subplots(1,3, figsize=(20,5))

############
## Figure 1
sns.lineplot(phiInterval, pliAB, ax = ax[0], color = 'blue')
sns.lineplot(phiInterval, plvAB, ax = ax[0], color = 'green')
sns.lineplot(phiInterval, lpsAB, ax = ax[0], color = 'red')
sns.lineplot(phiInterval, corAB, ax = ax[0], color = 'purple')

ax[0].set(ylim= (-0.05, 1.05))
y_tick = np.array([-0.05 , 0.0, 0.5, 1.0, 1.05 ]) #np.linspace(0.99987,1.00001, 5)
y_ticklabels = ['','0.0', '0.5', '1.0', '']
plt.setp(ax[0], yticks = y_tick, yticklabels = y_ticklabels)


ax[0].set_xlabel(r'$\phi_a$')
ax[0].set_ylabel('Conn. Measure')
ax[0].set_title('Signal A vs. B', fontsize = 30)


############
## Figure 2
sns.lineplot(phiInterval, pliAC, ax = ax[1], color = 'blue')
sns.lineplot(phiInterval, plvAC, ax = ax[1], color = 'green', **{'linewidth': 3})
sns.lineplot(phiInterval, lpsAC, ax = ax[1], color = 'red')
sns.lineplot(phiInterval, corAC, ax = ax[1], color = 'purple')

x_tick = np.linspace(0,2*np.pi, 3)
x_ticklabels = [r"0", r"$\pi$", r"2$\pi$"]

ax[1].set_xlabel(r'$\phi_a$')
ax[1].set_ylabel('Conn. Measure')
ax[1].set_title('Signal A vs. C', fontsize = 30)

ax[1].set(ylim= (-0.05, 1.05))
y_tick = np.array([-0.05 , 0.0, 0.5, 1.0, 1.05 ]) #np.linspace(0.99987,1.00001, 5)
y_ticklabels = ['','0.0', '0.5', '1.0', '']
plt.setp(ax[1], yticks = y_tick, yticklabels = y_ticklabels)

############
## Figure 3
sns.lineplot(phiInterval, pliCB, ax = ax[2], color='blue')
sns.lineplot(phiInterval, plvCB, ax = ax[2], color = 'green', **{'linewidth': 3})
sns.lineplot(phiInterval, lpsCB, ax = ax[2], color = 'red')
sns.lineplot(phiInterval, corCB, ax = ax[2], color = 'purple')

ax[2].set_xlabel(r'$\phi_b$')
ax[2].set_ylabel('Conn. Measure')
ax[2].set_title('Signal B vs. C', fontsize = 30)

ax[2].set(ylim= (-0.05, 1.05))
y_tick = np.array([-0.05 , 0.0, 0.5, 1.0, 1.05 ]) #np.linspace(0.99987,1.00001, 5)
y_ticklabels = ['','0.0', '0.5', '1.0', '']
plt.setp(ax[2], yticks = y_tick, yticklabels = y_ticklabels)

############



plt.subplots_adjust(top = 0.8, bottom = 0.2, hspace = 0.3, wspace = 0.37)
fig.legend(['PLI', 'PLV', 'LPS', 'abs(Correlation)'], bbox_to_anchor = (0.5, 1.1), 
           borderaxespad = 0., loc = 'upper center', ncol = 4)
# plt.tight_layout()
x_tick = np.linspace(0,2*np.pi, 3)
x_ticklabels = [r"0", r"$\pi$", r"2$\pi$"]
plt.setp(ax, xticks = x_tick, xticklabels = x_ticklabels)

#plt.plot(phiInterval, plvAC) #, ax = ax[0])
plt.show()
fig.savefig('/share/FannyMaster/PythonNew/Figures/ConnectivityMeasures.jpg', bbox_inches = 'tight')

sns.reset_orig()

#%%############
## Figure 1 zoom
sns.set(font_scale=2)
fig = plt.figure(figsize=(11,6))

sns.lineplot(phiInterval, pliAB, color = 'blue')
sns.lineplot(phiInterval, plvAB, color = 'green', **{'linewidth': 3})
sns.lineplot(phiInterval, lpsAB, color = 'red')
sns.lineplot(phiInterval, corAB, color = 'purple')

plt.ylim(0.99985, 1.00001)
# y_tick = np.array([0.99987 , 0.99988, 0.99991 , 0.99994, 0.99997, 1.0, 1.00001 ]) #np.linspace(0.99987,1.00001, 5)
# y_ticklabels = ['','0.99988', '0.99991' , '0.99994', '0.99997', '1.0', '']
y_tick = np.array([0.99985 , 0.99986, 0.99993, 1.0, 1.00001 ]) #np.linspace(0.99987,1.00001, 5)
y_ticklabels = ['','0.99982', '0.99993', '1.0', '']
# plt.setp(yticks = y_tick, yticklabels = y_ticklabels)
plt.yticks(y_tick, labels = y_ticklabels)

x_tick = np.linspace(0,2*np.pi, 3)
x_ticklabels = [r"0", r"$\pi$", r"2$\pi$"]
plt.xticks(x_tick, labels = x_ticklabels)

#plt.legend(['PLI', 'PLV', 'LPS', '|Corr|'], loc = 'lower center', ncol = 4) #, bbox_to_anchor = (0.5, 1.1), borderaxespad = 0., loc = 'upper center', ncol = 2)
plt.xlabel(r'$\phi_a$')
plt.ylabel('Conn. Measure')
plt.title('Zoom Signal A vs. B', fontsize = 30)

plt.tight_layout()
plt.show()

fig.savefig('/share/FannyMaster/PythonNew/Figures/zoomAB.jpg', bbox_inches = 'tight')
sns.reset_orig()



