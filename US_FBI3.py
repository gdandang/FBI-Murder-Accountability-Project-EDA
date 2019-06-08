
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import requests
from bs4 import BeautifulSoup


# In[2]:


data = pd.read_csv("database.csv")


# In[2]:


str ="Dataset info :\nThe Murder Accountability Project is the most complete database of homicides in the United States currently available. This dataset includes murders from the FBI's Supplementary Homicide Report from 1976 to the present and Freedom of Information Act data on more than 22,000 homicides that were not reported to the Justice Department. This dataset includes the age, race, sex, ethnicity of victims and perpetrators, in addition to the relationship between the victim and perpetrator and weapon used."
print("\033[1;31;20m \n{}".format(str))
print("\033[0;30;48m   ")


# In[8]:


data.info()


# In[3]:


#rename colums titles to lower with no spaces
data = data.rename(index=int, columns=dict(zip(list(data), [(lambda x: x.lower().replace(' ','_') )(x) for x in list(data)])))


# In[11]:


data.info()


# In[28]:


uni = dict(zip(list(data), [(lambda x: len(data[x].unique()) )(x) for x in list(data)]))
print(uni)


# In[29]:


from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(25, 10))
plt.bar(range(len(uni)), uni.values(), align='center')
plt.xticks(range(len(uni)), list(uni.keys()),rotation=45, ha='right')
plt.show()


# In[4]:


#drooping columns with uniqe values
data = data.drop(['record_id'],axis=1)
data.dropna()


# In[31]:


uni = dict(zip(list(data), [(lambda x: len(data[x].unique()) )(x) for x in list(data)]))
print(uni)


# In[32]:


fig = plt.figure(figsize=(25, 10))
plt.bar(range(len(uni)), uni.values(), align='center')
plt.xticks(range(len(uni)), list(uni.keys()),rotation=45, ha='right')
plt.show()


# In[ ]:


#filtering out column 'agency_code' due to high amount of categories,removing rows with less then 10 apperances for the given agency code

data['count_agency_code'] = data.groupby('agency_code')['agency_code'].transform(pd.Series.value_counts)
data = data[ data['count_agency_code'] > 100  ]
data = data.drop(['count_agency_code'],axis=1)

#  fixing the problematic columns with for loop :

# column_list = ['agency_code','agency_name','city','incident']
# for i in column_list:
#     data['temp'] = data.groupby(i)[i].transform(pd.Series.value_counts)
#     data = data[ data['temp'] > 100  ]
#     data = data.drop(['temp'],axis=1)


# In[315]:


# correlation test is not relevant because most of the columns are catgorical
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[35]:


print("\033[1;31;20m \n       Top 3 frequent categories in dataset for each column by freq percantage:  ")
print("\033[0;30;48m   ")
for i in data.columns:
    print("{}unique values:{}\n".format( round((data.groupby([i]).size()/len(data)*100),2).sort_values(ascending=False).head(3),len(data[i].value_counts())).replace('dtype: float64','').replace("i\n","\n"))


# In[5]:


# Adding 'month_int' column in addition to the existin string column
months = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6, "July":7, "August":8, "September":9, "October":10, "November":11, "December":12 }
data['month_int'] = data['month'].map(months)

#Validating the month_int new column by simple group by
data[ (data['state'] == 'California') & (data['year'] == 1993 )].groupby(['month_int']).size()


# In[6]:


# categorizing numerical columns by groups of 10 years (ages (perpetrator & victim) and year (event year)) 
data['victim_age_group'] =  data['victim_age'].apply( lambda x: "{}-{}".format((int(x/10)*10),(int(x/10))*10+10) if ((str(x) != 0) & (str(x) != '')) else 'unknown' )
data['perpetrator_age_group'] =  data['perpetrator_age'].apply( lambda x: "{}-{}".format((int(x/10)*10),(int(x/10))*10+10) if ('int' in str(type(x))) & (str(x)!='0') else 'unknown')
data['year_group'] = data['year'].apply(lambda x : "{}-{}".format( (int(x) - int(int(x)%10)) , (int(x) + (10-int(int(x)%10))  ) ))


# In[553]:


#Validating the perpetrator_age_group new column by simple group by
data[data['perpetrator_age_group'] == '0-10' ]


# In[549]:


data.head()


# In[8]:


# perpetrator VS victims ages graph

data_age = data[(data['victim_age_group'] != '990-1000') & (data['perpetrator_age_group'] != 'unknown')  & (data['victim_age_group'] != 'unknown') ][['perpetrator_age','victim_age','victim_age_group','perpetrator_age_group']]
# data_age = data[  (data['victim_age_group'] != '990-1000') & (data['perpetrator_age_group'] != 'unknown')  & (data['victim_age_group'] != 'unknown') ]
data_age['temp']=1
print("\033[1;31;20m \n       Perpetrator VS Victims Ages Graph  ")
print("\033[0;30;48m   ")
pd.pivot_table(data_age, index=('victim_age_group'), columns=('perpetrator_age_group'),
               values='temp', aggfunc='count').plot(kind='bar',figsize=(20, 10),width =1.2)  


# In[10]:


# Zooming in ages groups 10-20 & 20-30 which looks suspicious
young_ages_list = [10,11,12,13,14,15,16,17,18,19]
adult_ages_list = [20,21,22,23,24,25,26,27,28,29]
young_ages_data = data[  (data['perpetrator_age'].isin(young_ages_list)) & (data['victim_age'].isin(young_ages_list)) ][['victim_age','perpetrator_age']]
adult_ages_data = data[  (data['perpetrator_age'].isin(adult_ages_list)) & (data['victim_age'].isin(adult_ages_list)) ][['victim_age','perpetrator_age']]


# In[12]:


young_ages_data['victim_age'] = young_ages_data['victim_age'].apply(lambda x: int(x))
young_ages_data['perpetrator_age'] = young_ages_data['perpetrator_age'].apply(lambda x: int(x))
young_ages_data['temp'] = 1
adult_ages_data['victim_age'] = adult_ages_data['victim_age'].apply(lambda x: int(x))
adult_ages_data['perpetrator_age'] = adult_ages_data['perpetrator_age'].apply(lambda x: int(x))
adult_ages_data['temp'] = 1

piv_young_ages_data = pd.pivot_table(young_ages_data, index=('perpetrator_age'), columns=('victim_age'),
               values='temp', aggfunc='count') 
piv_adult_ages_data = pd.pivot_table(adult_ages_data, index=('perpetrator_age'), columns=('victim_age'),
               values='temp', aggfunc='count') 

import ipywidgets as widgets
from IPython import display

def highlight_max(data, color='red'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


print("\033[1;31;20m \n       Pivot table for ages 10-19 (count):  ")
print("\033[0;30;48m   ")
df1 = piv_young_ages_data
df1.style.apply(highlight_max)


# In[13]:


print("\033[1;31;20m \n       Pivot table for ages 20-29 (count):  ")
print("\033[0;30;48m   ")
df2 = piv_adult_ages_data
df2.style.apply(highlight_max)


# In[16]:


print("\033[1;31;20m \n       Weapons Count  ")
print("\033[0;30;48m   ")
my_tab = pd.crosstab(index = data["weapon"],columns="count")      
my_tab.plot.bar()


# In[18]:


print("\033[1;31;20m \n       Weapons Per State Heat Map  ")
df_heatmap = pd.pivot_table(data, index=('state'), columns=('weapon'),values='incident', aggfunc='count')   # index - rows , columns - columns 
sns.heatmap(df_heatmap,annot=False) 
plt.show()


# In[683]:


# https://www.usclimatedata.com/climate/alaska/united-states/3171# 
# climate date

url ="https://www.usclimatedata.com"
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
states_dict = {}
for i in soup.find_all('a', class_='province'):
    states_dict[i.get('title').replace('Climate ','')] = 'https://www.usclimatedata.com'+i.get('href')
    print(i.get('title').replace('Climate ','')+' : '+'https://www.usclimatedata.com'+i.get('href'))

month_list  = []

months = soup.find_all('th', class_='climate_table_data_td')
for p in months:
    month_list.append(p.text)
month_dem = pd.DataFrame(pd.Series(month_list),columns=['month_str'])
month_dem['month_int'] = [1,2,3,4,5,6,7,8,9,10,11,12]

for i in states_dict:
    monthly_avg_temp_f_list = []
    url = states_dict[i]
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    monthly_avg_temp_f = soup.find_all('td', class_='align_right temperature_red')
    for p in monthly_avg_temp_f:
        monthly_avg_temp_f_list.append(round((int(p.text) -32) * (5/9),3) )
    month_dem[i] = pd.Series(monthly_avg_temp_f_list)





# In[684]:


month_dem


# In[296]:


import matplotlib.pyplot as plt
import pandas as pd

# gca stands for 'get current axis'
ax = plt.gca()

for i in month_dem.iloc[:, 2:]:
    month_dem.plot(kind='line',x='month_int',y=i ,ax=ax, figsize = (15,5))

plt.title("Monthly Temperature Per States")
plt.xlabel("Month")
plt.ylabel("Temp (c)");
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=6)
#ax.get_legend().remove()
plt.show()


# In[829]:


states = data[data['state'] != 'Rhodes Island']['state'].drop_duplicates()
# states = data['state'].drop_duplicates()

for i in states:   
    events =  pd.DataFrame()
    st_data = pd.DataFrame()
    st_data[['month_int','avg_temp']] = month_dem[['month_int',i ]]
    events = data[data['state'] == i ][['month_int','month']].groupby('month_int', sort=False)["month"].count().reset_index(name ='events_count')
    result = pd.merge(st_data,events,how='left',on=['month_int'])
    result['avg_temp_manipulated'] = result['avg_temp'] * (result['events_count'].mean() / result['avg_temp'].mean())
    print("\033[1;31;20m \n         {}     ".format(i))
    print("\033[0;30;48m   ")
    print(result)
    
    line_up, = plt.plot(result['month_int'],result['avg_temp_manipulated'], label='Line 2')
    line_down, = plt.plot(result['month_int'],result['events_count'], label='Line 1')
    plt.legend([line_up, line_down], ['avg_temp_manipulated', 'events_count'])

#     print(i)
    plt.show()

