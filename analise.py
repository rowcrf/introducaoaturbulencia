# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 10:13:11 2021

@author: Rogerio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def read_data(file_path,column_name,number_of_files,kind = False):
    '''
    Function to read High and Low Reynolds data also X_5 dataset
    Parameters
    ----------
    file_path : string
        One should enter .dat file path, without the doc number neither .dat 
        extension
        e.g.: 'folder/file' for 'folder/fileNUMBER.dat'
    column_name : string
        column name.
    number_of_files: int
        number of files on folder
    kind : boolean
        False if numbers 1,2,3,4,5...
        True if numbers 01,02,03...

    Returns
    -------
    merged dataframe on 'time' column .

    '''
  
    if kind:
        c1_name = column_name + '01'
        f1_path = file_path + '01.dat'
    else:
        c1_name = column_name + '1'
        f1_path = file_path + '1.dat'        
    df = pd.read_csv(f1_path, header =None, sep= '  ',
                     engine = 'python', names = ['time',c1_name],
                     dtype = np.float64)
    
    for i in range (2,number_of_files+1):
        if kind and i<10:
            f_path = file_path + '0'+ str(i) + '.dat'
            column = column_name + '0'+ str(i)
        else:
            f_path = file_path + str(i) + '.dat'
            column = column_name + str(i)
        new_column = pd.read_csv(f_path,header = None, sep = '  ',
                                 engine = 'python', names = ['time',column],
                                 dtype = np.float64)
        df = pd.merge(df,new_column, on='time')
    df.set_index('time',inplace = True)
    
    
    return df

def read_data2(file_path,column_name,number_of_files,mon):
    '''
    Function to read upstream and downstream data
    Parameters
    ----------
    file_path : string
        One should enter .dat file path, without the doc number neither .dat 
        extension
        e.g.: 'folder/file' for 'folder/fileNUMBER.dat'
    column_name : string
        DESCRIPTION.
    number_of_files : int
        DESCRIPTION.
    mon : boolean
        True if 'montante'
        False if 'jusante'.

    Returns
    -------
    merged dataframe on 'time' column .

    '''
    if mon:
        c1_name = column_name + '01'
        f1_path = file_path + '01'  
    else: 
        c1_name = column_name + '17'
        f1_path = file_path + '17'
        
    df = pd.read_csv(f1_path, header =None, sep= '  ',
                     engine = 'python', names = ['time',c1_name],
                     dtype = np.float64)
    if mon: 
        a = 2
        b = number_of_files+1
    else:
        a = 18
        b = a + number_of_files+1
        
    for i in range (a,b):
        if i<10:
            f_path = file_path + '0'+ str(i)
            column = column_name + '0'+ str(i)
        else:
            f_path = file_path + str(i) 
            column = column_name + str(i)
        new_column = pd.read_csv(f_path,header = None, sep = '  ',
                                 engine = 'python', names = ['time',column],
                                 dtype = np.float64)
        df = pd.merge(df,new_column, on='time')
    df.set_index('time',inplace = True)
    
    
    return df

def my_plot(df,column,save=False,fname=None):
    

    fig, ax = plt.subplots(1, 2,figsize=(10,5),
                           gridspec_kw={'width_ratios': [2, 1]})
    ax[0].plot(df.index,df[column], alpha = 0.85)
    ax[0].set(xlabel = 'time [$s$]',ylabel = '$U$ [$m/s$]')

       #density If True, draw and return a probability density
    ax[1].hist(df[column],orientation = 'horizontal', alpha = 0.85,
               density=True,bins=50,color = 'red',edgecolor='black')
    
    sns.kdeplot(ax=ax[1],data=df,y=column , 
                linewidth=2, color='black', alpha=0.9)
    ax[1].set(xlabel = 'density',ylabel = '$U$ [$m/s$]')
    
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()


    if save:
        plt.savefig(fname,format='pdf')




hre = read_data('hre/hre','x',5,False) #High Reynolds data. Speed on x-axis
lre = read_data('lre/lre','x',5,False) #Low Reynolds data. Speed on x-axis
hre_prob=read_data('hre_prob/hrep','x',25,True) # 25 measurements on x_5 to create an statistic average.

perfil_x1 = read_data2('perfil_mon/PERFILM.W','z',16,True) # x_1 profile
perfil_x3 = read_data2('perfil_jus/PERFILM.W','z',16,False) # x_3 profile

     

#data
H = 5 #cm
H = H*10**-2 #m

z_1 = np.array([34,34.5,35,35.5,36,37,
                38,40,45,50,60,70,80,
                90,100,110])


z_3 = np.array([34,36,38,40,45,50,55,
                60,65,70,75,80,85,90,
                98,100,105,110])




rho_air = 1.204 #kg/m³
mu_air = 1.825*10**-5 #kg/m*s

#Coordinates (x,y,z)

X = np.array([[-1.5*H,0,0.5*H], #x1
             [H,0,0.5*H],       #x2
             [1.5*H,0,0.5*H],   #x3
             [2*H,0,0.5*H],     #x4
             [2.5*H,0,0.5*H]])  #x5


X = X*10**-3 #coordinates [mm]

L = 2.7*H #m Lenght for Reynolds   
positions = ['x1','x2','x3','x4','x5'] #hre and lre

positions_upstream = []
positions_downstream = []
for i in range (1,17):
    if i< 10:
        i = '0'+str(i)
    else:
        i = str(i)
    positions_upstream.append('z'+i)
for i in range (17,35):
    positions_downstream.append('z'+str(i))
    
    
positions_prob = []

for i in range (1,26):
    if i<10:
        p = 'x0'+str(i)
    else:
        p = 'x'+str(i)
    positions_prob.append(p)

#Q1

#Flutuations and mean velocity
for pos in positions:
    hre[pos+'_mean'] = hre[pos].mean()
    hre[pos+'_flut'] = hre[pos] - hre[pos].mean() 
    lre[pos+'_mean'] = hre[pos].mean()
    lre[pos+'_flut'] = lre[pos] - lre[pos].mean()
    
hre.to_csv('hre_tratado.txt')
lre.to_csv('lre_tratado.txt')

#Q2

for pos in positions:
    my_plot(hre,pos,True,'hre_'+pos+'.pdf')
for pos in positions:
    my_plot(lre,pos,True,'lre_'+pos+'.pdf')



#Q3
hRe = [] #high Reynolds dataset Reynolds number
lRe = [] #low Reynolds dataset Reynolds number
for x in hre[positions]:
    hRe.append( rho_air*L*hre[x].mean()/mu_air)
for x in lre[positions]:
    lRe.append(rho_air*L*lre[x].mean()/mu_air)

#Q4
hVar = [] #high Reynolds dataset variance
lVar = [] #low Reynolds dataset variance

for x in hre[positions]:
    hVar.append(hre[x].var(ddof = 1))
for x in lre[positions]:
    lVar.append(lre[x].var(ddof = 1))

#Q5
hStd = [] #high Reynolds dataset Standard deviation
lStd = [] #low Reynolds dataset Standard deviation

for x in hre[positions]:
    hStd.append(hre[x].std(ddof = 1))
for x in lre[positions]:
    lStd.append(lre[x].std(ddof = 1))

#Q6
hI = []
lI = []
for pos in positions:
    hI.append(((hre[pos + '_flut']**2).mean())**(0.5)/hre[pos].mean())
    lI.append(((lre[pos + '_flut']**2).mean())**(0.5)/lre[pos].mean())
    
    
#Q7
hK = []
lK = []
for pos in positions:
    hK.append(0.5*((hre[pos+'_flut']**2).mean()))
    lK.append(0.5*((lre[pos+'_flut']**2).mean()))
    
    
#Q8
hCov = hre[positions].cov() #high Reynolds dataset Covariance
lCov = lre[positions].cov() #low Reynolds dataset Covariance

#Q9
hCorr = hre[positions].corr() #high Reynolds dataset Correlation
lCorr = lre[positions].corr() #low Reynolds dataset Correlation

#Q10

hS = [] #high Reynolds 
lS = []

hT = []
lT = []

for pos in positions:
    hS.append((hre[pos+ '_flut']**3).mean()/(((hre[pos+'_flut'])**2).mean())**1.5)
    hT.append((hre[pos+'_flut']**4).mean()/ ((hre[pos+'_flut']**2).mean())**2)
    
    lS.append((lre[pos+ '_flut']**3).mean()/(((lre[pos+'_flut'])**2).mean())**1.5)
    lT.append((lre[pos+'_flut']**4).mean()/ ((lre[pos+'_flut']**2).mean())**2)
    
    
#Q11

#done with #Q2


#Q12


plt.clf()
plt.plot(perfil_x1.mean(),z_1,label = 'Upstream $X_1$',marker='o')
plt.plot(perfil_x3.mean(),z_3,label= 'Downstream $X_3$',marker='x')
plt.grid()
plt.legend()
plt.xlabel(r'$\bar{U}$ [$m/s$]')
plt.ylabel('z [$mm$]')
plt.tight_layout()
plt.savefig('q12.pdf',format='pdf')


#Q13





for pos in positions_upstream:
    perfil_x1[pos+'_mean'] = perfil_x1[pos].mean()
    perfil_x1[pos+'_flut'] = perfil_x1[pos] - perfil_x1[pos].mean() 
   
for pos in positions_downstream:
    perfil_x3[pos+'_mean'] = perfil_x3[pos].mean()
    perfil_x3[pos+'_flut'] = perfil_x3[pos] - perfil_x3[pos].mean() 
    
    
x1_I = []
x3_I = []

for pos in positions_upstream:
    x1_I.append(((perfil_x1[pos + '_flut']**2).mean())**(0.5)/perfil_x1[pos].mean())

for pos in positions_downstream:
    x3_I.append(((perfil_x3[pos + '_flut']**2).mean())**(0.5)/perfil_x3[pos].mean())


plt.clf()
plt.plot(x1_I,z_1,label = 'Upstream $X_1$',marker='o')
plt.plot(x3_I,z_3,label= 'Downstream $X_3$',marker='x')
plt.grid()
plt.legend()
plt.xlabel('Turbulence Intensity')
plt.ylabel('z [$mm$]')
plt.tight_layout()
plt.savefig('q13.pdf',format='pdf')


    



#Q14


    

for pos in positions_prob:
    hre_prob[pos+'_mean'] = hre_prob[pos].mean()
    hre_prob[pos+'_flut'] = hre_prob[pos] - hre_prob[pos].mean() 
    hre_prob['Re_' + pos] = rho_air*L*hre_prob[pos].mean()/mu_air
    hre_prob[pos + '_var'] = hre_prob[pos].var(ddof = 1)
    hre_prob[pos + '_std'] = hre_prob[pos].std(ddof = 1) 
    hre_prob[pos + '_I'] = ((hre_prob[pos + '_flut']**2).mean())**(0.5)/hre_prob[pos].mean()
    hre_prob[pos + '_K'] = 0.5*((hre_prob[pos+'_flut']**2).mean())
    hre_prob[pos+ '_S'] = (hre_prob[pos+ '_flut']**3).mean()/(((hre_prob[pos+'_flut'])**2).mean())**1.5
    hre_prob[pos+'_T'] = (hre_prob[pos+'_flut']**4).mean()/ ((hre_prob[pos+'_flut']**2).mean())**2
    
hre_prob.to_csv('hre_prob_tratado.txt')    

    
    
    
    
    









        
        