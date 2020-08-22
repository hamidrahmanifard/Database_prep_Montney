# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:39:32 2020

@author: hamid.rahmanifard
"""

#==============================================================================================================
#General Input
#--------------------------------------------------------------------------------------------------------------
# Import all required libraries
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
start_time = time.time()

#==============================================================================================================
# Required functions
#--------------------------------------------------------------------------------------------------------------

#==============================================================================================================
# get or change the wrking directory	
#--------------------------------------------------------------------------------------------------------------
os.chdir('C:/Users/hamid.rahmanifard/OneDrive - University of Calgary/UofC/PhD/Publication/Conferences/URtech-2020/Montney gas prod/2020/Database/Forward/Input Data')

#==============================================================================================================
# Adding Production data, TD, IPyear, TVD, welltype, Fltype, drill_type
#--------------------------------------------------------------------------------------------------------------
# Loading production file
prod_data = pd.read_csv('4.zone_prd.csv', sep=',', skiprows = 1)
prod_data = prod_data.drop(prod_data.columns[np.r_[-5:0]], axis =1)
prod_data.columns =['WAN','Compltn_event_seq','Prod_period','UWI','Area_code',
                    'Formtn_code', 'Pool_seq', 'GAS', 'OIL', 'WAT', 'CON', 'Prod_days']
#--------------------------------------------------------------------------------------------------------------
# making the wide table long
prod_data1 =pd.melt(prod_data, id_vars=['UWI','Prod_period','Prod_days'],value_vars=['GAS', 'OIL', 'WAT', 'CON'])
prod_data1 = prod_data1[prod_data1['value'] > 0]
prod_data1 ['Year'] = prod_data1 ['Prod_period'].astype(str).str[:4]
prod_data1 ['Month'] = prod_data1 ['Prod_period'].astype(str).str[-2:]
#--------------------------------------------------------------------------------------------------------------
# creating a unique identifier for each well
prod_data1 ['Id']= prod_data1 ['UWI'].astype(str)+'-'+ prod_data1 ['Year'].astype(str)+'-'+prod_data1 ['variable']
#--------------------------------------------------------------------------------------------------------------
# creating a dataframe for the pools and formations
form_spec = prod_data[['UWI', 'WAN', 'Area_code', 'Formtn_code','Pool_seq']]
form_spec = pd.DataFrame.drop_duplicates(form_spec)
#--------------------------------------------------------------------------------------------------------------
# creating a dataaframe where for each well all months within a year are in a row
df2 = pd.pivot_table(prod_data1,index= 'Id', columns='Month', values = 'value')
df2 = df2.reset_index()
#--------------------------------------------------------------------------------------------------------------
# ann vol calculations
df2[13] = df2.sum(axis=1)
#--------------------------------------------------------------------------------------------------------------
# adding other columns as necessary to be in the output df
df2 ['UWI'] = df2 ['Id'].astype(str).str[:16]
df2 ['Year'] = df2 ['Id'].astype(str).str[17:21]
df2 ['Fl_type'] = df2 ['Id'].astype(str).str[-3:]
df2.columns = ['Id', 'FlVol_Mnth1', 'FlVol_Mnth2','FlVol_Mnth3', 'FlVol_Mnth4',
               'FlVol_Mnth5','FlVol_Mnth6', 'FlVol_Mnth7', 'FlVol_Mnth8', 
               'FlVol_Mnth9', 'FlVol_Mnth10', 'FlVol_Mnth11', 'FlVol_Mnth12',
               'Ann_Vol','UWI', 'Year', 'Fl_type']
cols = df2.columns.tolist()
cols = cols[-3:] + cols[:-3]
df2 = df2[cols]
del cols
#--------------------------------------------------------------------------------------------------------------
# creating a datgaframe where for each well production days within a year are in a row
df3 = pd.pivot_table(prod_data1,index= 'Id', columns='Month', values = 'Prod_days')
df3 = df3.reset_index()
df3.columns = ['Id', 'MnthProd_d1', 'MnthProd_d2','MnthProd_d3', 'MnthProd_d4',
               'MnthProd_d5','MnthProd_d6', 'MnthProd_d7', 'MnthProd_d8', 
               'MnthProd_d9', 'MnthProd_d10', 'MnthProd_d11', 'MnthProd_d12']
#--------------------------------------------------------------------------------------------------------------
# Attaching the days df to the month df
df2 = pd.merge(df2, df3, how='left',left_on='Id', right_on='Id') 
del df3
#--------------------------------------------------------------------------------------------------------------
# days in a month
mnthdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#--------------------------------------------------------------------------------------------------------------
# Replacing all 0's in Opr_Mnth to mnthdays' values based on two condititions
for ipt in range(12):
    df2.iloc[:,ipt+17] = np.where(df2.iloc[:,ipt+17]==0, np.where(df2.iloc[:,ipt+4]!=0,
            mnthdays[ipt], df2.iloc[:,ipt+17]),df2.iloc[:,ipt+17])
#--------------------------------------------------------------------------------------------------------------
# adding the cols containing monthly prod rate 
for ipt in range(12):
    df2['mnthprodrate_' + str(ipt+1)] = df2.iloc[:,ipt+4]/df2.iloc[:,17+ipt]
#--------------------------------------------------------------------------------------------------------------
# calcualting ann avg prod rategg
df2['ann_prodrate'] =df2['Ann_Vol']/ df2.iloc[:,17:29].sum(axis=1)
#--------------------------------------------------------------------------------------------------------------
# attaching the formation specifications to the main df
df2 = pd.merge(df2, form_spec, how='left',left_on='UWI', right_on='UWI') 
df2 = df2.drop(df2.columns[np.r_[3]], axis =1)
cols = df2.columns.tolist()
cols = cols[:3] + cols[-4:] + cols[3:-4]
df2 = df2[cols]
del cols
#--------------------------------------------------------------------------------------------------------------
#Well type
wtype = pd.read_csv('wtype.csv', sep = ',')
wstatus_df = pd.read_csv('2.w_status.csv', sep = ',',skiprows = 1)
wstatus_df = wstatus_df[wstatus_df['Status Term Date']==' ']
wstatus_df = pd.merge(wstatus_df, wtype, how='left',left_on='Fluid Type', right_on='Fluid Type')
wstatus_df = wstatus_df.drop(wstatus_df.columns[np.r_[1:6]], axis =1)
wstatus_df = wstatus_df.rename(columns={'Wa Num': 'WAN'})
wstatus_df['WAN'] = wstatus_df['WAN'].astype(int)
df2 = pd.merge(df2, wstatus_df, how='left',left_on='WAN', right_on='WAN')
df2['welltype']= df2['welltype'].fillna('UNK')
#--------------------------------------------------------------------------------------------------------------
#TD depth
drill_df = pd.read_csv('2.drill_ev.csv', sep = ',',skiprows = 1)
drill_df = drill_df.rename(columns={'Td Depth (m)': 'TD' })
drill_df = drill_df[['UWI', 'TD']]
df2 = pd.merge(df2, drill_df, how='left',left_on='UWI', right_on='UWI') 
df2['TD']= df2['TD'].fillna(0)
#--------------------------------------------------------------------------------------------------------------
# Well traj and location
well_df = pd.read_csv('2.wells.csv', sep = ',',skiprows = 1)
dtype = pd.read_csv('dtype.csv', sep = ',')
well_df = well_df.rename(columns={'Well Name':'Well_Name','WA Num':'WAN','Directional Flag':'Dir_Flag',
                                  'Surf UTM83 Northng':'Northing','Surf UTM83 Eastng':'Easting'})
well_df = well_df[['WAN', 'Dir_Flag', 'Well_Name','Northing','Easting']]
well_df['WAN'] = well_df['WAN'].astype(int)
df2 = pd.merge(df2, well_df, how='left',left_on='WAN', right_on='WAN')
df2['drltype'] = np.where(df2['Well_Name'].str.contains(' HZ '), 'H', 'V') 
df2['dtype'] = df2['Dir_Flag'] + df2['drltype']
df2 = pd.merge(df2, dtype, how='left',left_on='dtype', right_on='dtype')
df2 = df2.drop(df2.columns[np.r_[-2,-3,-6,-7]], axis =1)
#--------------------------------------------------------------------------------------------------------------
# Attaching the Ipyear col to the dataframe
Ipyear_df = df2.groupby(['UWI']).agg({'Year':'min'})
Ipyear_df.columns = ['Ipyear']
Ipyear_df ['UWI'] =Ipyear_df.index
Ipyear_df = Ipyear_df.reset_index(drop=True)
cols = Ipyear_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
Ipyear_df = Ipyear_df[cols]
del[cols]
df2 = pd.merge(df2, Ipyear_df, how='left',left_on='UWI', right_on='UWI')

#==============================================================================================================
# Adding Inclination,Azimuth, TVD
#--------------------------------------------------------------------------------------------------------------
drill_survey = pd.read_csv('1.dir_survey.csv', sep=',', skiprows = 1)
drill_survey = drill_survey.rename(columns={'WA NUM': 'WAN','Measured Depth (m)': 'MD',
                'Inclination (deg)': 'Inclination','Azimuth (deg)': 'Azimuth','TV Depth (m)':'TVD',
                'North South (m)':'NS', 'East West (m)':'EW','Drilling Event':'drl_event'})
drill_survey['id'] = drill_survey['UWI'].astype('str') + '-' + drill_survey['MD'].astype('str')
drill_survey1 = drill_survey [['id','Inclination','Azimuth','TVD']]
drlsurv_df = drill_survey.groupby(['UWI']).agg({'MD':'max'}) 
drlsurv_df ['UWI'] =drlsurv_df.index
drlsurv_df = drlsurv_df.reset_index(drop=True)
cols = drlsurv_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
drlsurv_df = drlsurv_df[cols]
del [cols]
drlsurv_df ['id']= drlsurv_df['UWI'].astype('str') + '-' + drlsurv_df['MD'].astype('str')
drlsurv_df = pd.merge(drlsurv_df, drill_survey1, how='left',left_on='id', right_on='id')
drlsurv_df = drlsurv_df.drop(drlsurv_df.columns[np.r_[-4,-5]], axis =1)
df2 = pd.merge(df2, drlsurv_df, how='left',left_on='UWI', right_on='UWI')
df2.columns
"""
#==============================================================================================================
# Adding net thickness
#--------------------------------------------------------------------------------------------------------------
"""
"""
In this section I got the perf interval from the 3.perf_net_interval.csv, however as the no. of wells
that we got is 1476 which is lower than number of the wells that we got their HF sepc in the next 
section, I make this part as comment and use the comp interval mentioned in the 3.hydraulic_fracture.csv

perf_interval = pd.read_csv('3.perf_net_interval.csv', sep=',')
perf_interval = perf_interval.rename(columns={'INTERVAL TOP DEPTH (m)': 'd_top','INTERVAL BASE DEPTH (m)': 'd_bot'})
perf_interval = perf_interval [['UWI','d_top','d_bot']]
perf_interval ['net_thick1'] = perf_interval['d_bot'] - perf_interval['d_top']
perf_df = perf_interval.groupby(['UWI']).agg({'net_thick1':'sum'}) 
perf_df ['UWI'] =perf_df.index
perf_df = perf_df.reset_index(drop=True)
cols = perf_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
perf_df = perf_df[cols]
del [cols]
"""
#==============================================================================================================
# Adding HF spec
#--------------------------------------------------------------------------------------------------------------
hyd_frac = pd.read_csv('3.hydraulic_fracture.csv', sep=',')
hyd_frac.columns
hyd_frac = hyd_frac.rename(columns={'FRAC STAGE NUM': 'No_Frac','BREAK DOWN PRESSURE (MPa)': 'Pbd',
                'INST SHUT IN PRESSURE (MPa)': 'ISIP','MAX TREATING PRESSURE (MPa)': 'Ptr_max',
                'AVG TREATING PRESSURE (MPa)':'Ptr_avg', 'AVG RATE (m3/min)':'Q_avg',
                'FRAC GRADIENT (KPa/m)':'Frac_grad', 'TOTAL FLUID PUMPED (m3)':'V_tot',
                'PROPPANT TYPE1 PLACED (t)':'m_prop1', 'PROPPANT TYPE2 PLACED (t)':'m_prop2',
                'PROPPANT TYPE3 PLACED (t)':'m_prop3', 'PROPPANT TYPE4 PLACED (t)':'m_prop4',
                'COMPLTN TOP DEPTH (m)':'comp_top','COMPLTN BASE DEPTH (m)':'comp_bot'})
#--------------------------------------------------------------------------------------------------------------
"""
As there are lots of nan values in each col and we want to get a mean for Pbd, ISIP, Ptr_avg, 
Q_avg, and Frac_grad, I separated the HF dataframe into 6 categories to be able to get mean values
by excluding nan values from each col while for the some variables ('No_Frac','Ptr_max','V_tot',
'm_prop1','m_prop2','m_prop3','m_prop4') I want their max or sum values where I make all nans to 0
"""
HF_df1 = hyd_frac[['UWI','comp_top','comp_bot','No_Frac','Pbd','ISIP','Ptr_avg','Q_avg','Frac_grad',
                   'Ptr_max','V_tot','m_prop1','m_prop2','m_prop3','m_prop4']]
HF_df1['No_Frac'] = HF_df1['No_Frac'].apply(pd.to_numeric, errors='coerce')
HF_df1['comp_int']=HF_df1['comp_bot']-HF_df1['comp_top'] 
# compare the result with the net thickness with perf_df if it is higher then the perf value is considered 
#--------------------------------------------------------------------------------------------------------------
# Completion interval (m)
HF_df2 = HF_df1.groupby(['UWI']).agg({'comp_int':'sum'})
HF_df3 = HF_df1.groupby(['UWI']).agg({'comp_top':'min'})
HF_df4 = HF_df1.groupby(['UWI']).agg({'comp_bot':'max'})
HF_dfh= pd.concat([HF_df2,HF_df3,HF_df4], axis=1, sort=False)
HF_dfh['comp_int1']=HF_dfh['comp_bot']-HF_dfh['comp_top']
HF_dfh = HF_dfh.drop(HF_dfh.columns[np.r_[-2,-3]], axis =1)
HF_dfh['comp_int']=np.where(HF_dfh['comp_int']>HF_dfh['comp_int1'],HF_dfh['comp_int1'],HF_dfh['comp_int'])
HF_dfh = HF_dfh.drop(HF_dfh.columns[np.r_[-1]], axis =1)
#--------------------------------------------------------------------------------------------------------------
# propant placed (t)
HF_df2 = HF_df1[['UWI','m_prop1','m_prop2','m_prop3','m_prop4']]
HF_df2 = HF_df2.fillna(0)
HF_df2 ['m_prop']=HF_df2['m_prop1']+HF_df2['m_prop2']+HF_df2['m_prop3']+HF_df2['m_prop4']
HF_dfprop = HF_df2.drop(HF_df2.columns[np.r_[-2,-3,-4,-5]], axis =1)
HF_dfprop = HF_dfprop.groupby(['UWI']).agg({'m_prop':'sum'})
#--------------------------------------------------------------------------------------------------------------
# number of fracture stages, treatment press (MPa), total pumped vol (m3)
HF_df2 = HF_df1[['UWI','No_Frac','Ptr_max','V_tot']]
HF_df2 = HF_df2.fillna(0)
HF_df2 = HF_df2.groupby(['UWI']).agg({'No_Frac':'max','Ptr_max':'max','V_tot':'sum'})
#--------------------------------------------------------------------------------------------------------------
# number of ISIP (Mpa), avg treatment press (MPa), injection rate (m3/min), Frac_grad (kPa/m), pbd (Mpa)
HF_df3 = HF_df1[['UWI','ISIP']]
HF_df3 = HF_df3[HF_df3.ISIP.notnull()]
HF_df3 = HF_df3.groupby(['UWI']).agg({'ISIP':'mean'}) 

HF_df4 = HF_df1[['UWI','Ptr_avg']]
HF_df4 = HF_df4[HF_df4.Ptr_avg.notnull()]
HF_df4 = HF_df4.groupby(['UWI']).agg({'Ptr_avg':'mean'}) 

HF_df5 = HF_df1[['UWI','Q_avg']]
HF_df5 = HF_df5[HF_df5.Q_avg.notnull()]
HF_df5 = HF_df5.groupby(['UWI']).agg({'Q_avg':'mean'}) 

HF_df6 = HF_df1[['UWI','Frac_grad']]
HF_df6 = HF_df6[HF_df6.Frac_grad.notnull()]
HF_df6 = HF_df6.groupby(['UWI']).agg({'Frac_grad':'mean'}) 

HF_df7 = HF_df1[['UWI','Pbd']]
HF_df7 = HF_df7[HF_df7.Pbd.notnull()]
HF_df7 = HF_df7.groupby(['UWI']).agg({'Pbd':'mean'}) 
#--------------------------------------------------------------------------------------------------------------
HF_df= pd.concat([HF_dfh,HF_dfprop,HF_df2,HF_df3,HF_df4,HF_df5,HF_df6,HF_df7], axis=1, sort=False)
HF_df ['UWI'] =HF_df.index
HF_df = HF_df.reset_index(drop=True)
cols = HF_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
HF_df = HF_df[cols]
del [cols,HF_df1,HF_dfh,HF_dfprop,HF_df2,HF_df3,HF_df4,HF_df5,HF_df6,HF_df7]
df2 = pd.merge(df2, HF_df, how='left',left_on='UWI', right_on='UWI')

#==============================================================================================================
# Final db prep and export
#--------------------------------------------------------------------------------------------------------------
Initial_db = df2 [['UWI','Year','Fl_type','Formtn_code','ann_prodrate', 'welltype', 'TD', 'TVD', 'Northing',
       'Easting', 'drill_type', 'Ipyear', 'Inclination', 'Azimuth','comp_int', 'No_Frac', 'Pbd',
       'ISIP', 'Ptr_max', 'Ptr_avg', 'Q_avg','V_tot', 'm_prop']]
montgas_db1 = Initial_db[(Initial_db.Year == Initial_db.Ipyear)&(Initial_db.Fl_type=='GAS')&\
                         (Initial_db.welltype=='GAS')& (Initial_db.Formtn_code== 5000)
                         & (Initial_db.drill_type =='H')]
#--------------------------------------------------------------------------------------------------------------
# dealing with missing data
# To prepare a db without any nan as 1359 nan value goes for ISIP I start with this one
montgas_db1 = montgas_db1[montgas_db1.comp_int.notnull()]
# To prepare a db without any nan as 156 nan value goes for ISIP I start with this one
montgas_db1 = montgas_db1[montgas_db1.ISIP.notnull()]
# To prepare a db without any nan as 24 nan value goes for Q_avg I start with this one
montgas_db1 = montgas_db1[montgas_db1.Q_avg.notnull()]
# Then it is the Azimuth turn with 2 nan values
montgas_db1 = montgas_db1[montgas_db1.Azimuth.notnull()]
# Finally it is the Azimuth turn with 1 nan value
montgas_db1 = montgas_db1[montgas_db1.Pbd.notnull()]
# Double-check to ensure that there is no null values in the db
montgas_db1.isnull().sum()
montgas_db1.isna().any()
cols = montgas_db1.columns.tolist()
cols = cols[:4] + cols[5:]+ [cols[4]]
montgas_db1 = montgas_db1[cols]
del [cols]
# Export the final format
montgas_db = montgas_db1.drop(['UWI','Year','Fl_type','Formtn_code','welltype','drill_type','Ipyear'], axis=1)
montgas_db.TD = montgas_db.TD.astype('float64')
#montgas_db.dtypes
#A = [montgas_db.TD > montgas_db.TVD]
#--------------------------------------------------------------------------------------------------------------
# dealing with outliers
#montgas_db.plot(kind='box')
montgas_db = montgas_db[montgas_db.TVD <3500] #step1
montgas_db = montgas_db[(montgas_db.Inclination <100)&(montgas_db.Inclination >80)] #step2
montgas_db = montgas_db[(montgas_db.comp_int >1) ] #step3
montgas_db = montgas_db[(montgas_db.No_Frac <80) & (montgas_db.No_Frac > 0) ] #step4
montgas_db = montgas_db[(montgas_db.Pbd > 18) ] #step5
montgas_db = montgas_db[(montgas_db.ISIP <50) & (montgas_db.ISIP > 0) ] #step6
montgas_db = montgas_db[(montgas_db.Ptr_max <200) & (montgas_db.Ptr_max > 40) ] #step7
montgas_db = montgas_db[(montgas_db.Ptr_avg > 20) ] #step8
montgas_db = montgas_db[(montgas_db.Q_avg < 13) & (montgas_db.Q_avg > 2.5)] #step9
montgas_db = montgas_db[(montgas_db.V_tot < 50000)] #step11
montgas_db = montgas_db[(montgas_db.ann_prodrate < 500)] #step12
# Ploting boxplot for each step
for column in montgas_db:
    plt.figure()
    montgas_db.boxplot([column])
#--------------------------------------------------------------------------------------------------------------
# Input file for boxplot in R
montgas_Rplot = montgas_db.copy()
montgas_Rplot['Northing'] =montgas_Rplot.Northing/1000000
montgas_Rplot['Easting'] =montgas_Rplot.Easting/1000000
# The volume is reported in m3 however to reduce the scale it is now expressed in e3m3
montgas_Rplot['V_tot'] =montgas_Rplot['V_tot']/1000
# Adding the unit of the predictors for the boxplot
montgas_Rplot = montgas_Rplot.rename(columns={'ann_prodrate':'Qg_ann'})
#montgas_Rplot = montgas_Rplot.rename(columns={'TD':'TD (m)', 'TVD':'TVD (m)', 'Inclination':'Inclination (deg)', 
#       'Azimuth':'Azimuth (deg)','net_thick':'net_thick (m)',  'Pbd':'Pbd (MPa)','ISIP':'ISIP (MPa)',
#       'Ptr_max (MPa)':'Ptr_max (MPa)','Ptr_avg':'Ptr_avg (MPa)','Q_avg':'Q_avg (m3/min)','Frac_grad':'Frac_grad (KPa/m)',
#       'V_tot':'V_tot (e3m3)','m_prop':'m_prop (t)', 'ann_prodrate':'ann_prodrate (e3m3/day)'})
#--------------------------------------------------------------------------------------------------------------
# Export to the output directory for experimental design
os.chdir('C:/Users/hamid.rahmanifard/OneDrive - University of Calgary/UofC/PhD/Publication/Conferences/URtech-2020/Montney gas prod/2020/Database/Forward/Output')
#montgas_db1.to_csv('montgas_db1.csv', header=True, index=False, sep=',')
montgas_db.to_csv('montgas_db.csv', header=True, index=False, sep=',')
#montgas_Rplot.to_csv('montgas_Rplot.csv', header=True, index=False, sep=',')
#--------------------------------------------------------------------------------------------------------------
# Export to the finall db for ANN after going through minitab
os.chdir('C:/Users/hamid.rahmanifard/OneDrive - University of Calgary/UofC/PhD/Publication/Conferences/URtech-2020/Montney gas prod/2020/Experimental Design/Forward')
final_db = montgas_db [['TD','Azimuth','comp_int', 'No_Frac', 'Pbd','Ptr_avg','V_tot', 'm_prop','ann_prodrate',]]
final_db.to_csv('final_db.csv', header=True, index=False, sep=',')
#--------------------------------------------------------------------------------------------------------------
# Export input and output file for ML
Inputs = final_db[['TD','Azimuth','comp_int', 'No_Frac', 'Pbd','Ptr_avg','V_tot', 'm_prop']]
Targets = final_db[['ann_prodrate']]
os.chdir('C:/Users/hamid.rahmanifard/OneDrive - University of Calgary/UofC/PhD/Publication/Conferences/URtech-2020/Montney gas prod/2020/Model/Forward')
Inputs.to_csv('Inputs.txt', header=True, index=False, sep='\t')
Targets.to_csv('Targets.txt', header=True, index=False, sep='\t')
#==============================================================================================================
runtime = (time.time() - start_time)
# END

