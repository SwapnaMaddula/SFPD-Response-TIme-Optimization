
# coding: utf-8

# # 255- Project 
# # Crime Incident Clustering & Evaluation
# 

# ## By Savitri Swapna Maddula (012551799)

# ### Import necessary Libraries

# In[207]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pandas import ExcelWriter
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from geopy.distance import great_circle
from pandas import ExcelWriter
import os
import warnings
warnings.filterwarnings('ignore')


# ## Step-1: Data Collection

# In[208]:


incident_df = pd.read_csv('data/police-department-incidents.csv')


# In[209]:


incident_df.shape


# In[210]:


incident_df.head()


# In[211]:


incident_df.shape


# In[212]:


incident_df.dtypes


# In[213]:


incident_df['Location'][0]


# In[214]:


data = np.array(incident_df[['X', 'Y']])


# In[215]:


data.shape


# In[216]:


data[0]


# In[217]:


incident_df.sort_values(["Date"], ascending=True, inplace = True)
incident_df.head()


# In[218]:


incident_df.reset_index(inplace=True)
incident_df.head()


# ## Step-2: Preprocessing

# ### a) Feature Transformation

# In[219]:


incident_df['Resolution Status'] = np.where(incident_df["Resolution"]=='NONE', "Unresolved","Resolved")
incident_df.head()


# ### b) Discretization

# In[220]:


bins = [0, 7, 14, 20, 23]
group_names = ['EarlyMorning', 'Morning', 'Evening', 'Night']


incident_df["Period of day"] = pd.cut(incident_df["Time"].str.rstrip(':').str.split(':').str[0].astype(int), 
                                 bins, labels=group_names)
incident_df.head()


# In[221]:


incident_df['Date'] = pd.to_datetime(incident_df['Date'])
incident_df.head()


# ### c) Feature Creation

# In[222]:


incident_df["Year"] = incident_df["Date"].map(lambda x: x.year)
incident_df["Day"] = incident_df["Date"].map(lambda x: x.day)
incident_df["Month"] = incident_df["Date"].map(lambda x: x.month)
incident_df.tail()


# ## Step-3: Data Sampling

# ## 1. 2018 Year Data

# In[223]:


crimes_2018_df = incident_df.loc[incident_df['Year'] == 2018]
crimes_2018_df.head()


# In[224]:


crimes_2018_df.shape


# In[225]:


list(crimes_2018_df.columns)[1:]


# In[227]:


path = os.getcwd()
path = path +'\out'
os.mkdir(path)


# In[296]:


path = os.getcwd()
path = path +'\plots'
os.mkdir(path)


# In[228]:


writer = ExcelWriter('out/crimes_2018_df.xlsx')
crimes_2018_df.to_excel(writer,'Sheet1')
writer.save()


# In[229]:


data = np.array(crimes_2018_df[['X', 'Y']])


# ### 1.1 Clustering

# In[230]:


df = pd.DataFrame(data, columns = ['point_longitude_2018', 'point_latitude_2018'])

writer = ExcelWriter('out/points_2018_plot.xlsx')
df.to_excel(writer,'Sheet1', columns = ['point_longitude_2018', 'point_latitude_2018'])
writer.save()


# In[231]:


kmeans = KMeans(n_clusters=100, random_state=0).fit(data)


# ### 1.2 Predictions

# In[232]:


y_means = kmeans.predict(data)


# In[233]:


centers = kmeans.cluster_centers_


# ### 1.3 Visualization

# In[234]:


plt.title('Locations of police cruisers in SF (2018)', weight ='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(centers[:, 0], centers[:, 1], c= 'black', alpha =0.5)
plt.savefig('plots/centroids_2018.png')  


# ### 1.4 Average Time Calculation

# In[235]:


total_dist = 0
for i in range(len(y_means)):
    coords_1 = (centers[y_means[i]][1] , centers[y_means[i]][0])
    coords_2 = (data[i][1], data[i][0])
    individual_dist = great_circle(coords_1, coords_2).miles
    total_dist = total_dist + individual_dist


# In[236]:


avg_dist = total_dist/len(y_means)
avg_dist


# In[237]:


calculated_time_2018 = avg_dist *3600 / 10
calculated_time_2018


# In[238]:


y_means.shape


# ### 1.5 Saving to File

# In[239]:


df = pd.DataFrame(centers, columns = ['centroid_longitude_2018', 'centroid_latitude_2018'])


# In[240]:


writer = ExcelWriter('out/centroids_2018_check.xlsx')
df.to_excel(writer,'Sheet1', columns = ['centroid_longitude_2018', 'centroid_latitude_2018'])
writer.save()


# ## 2. 2017 Data

# In[241]:


crimes_2017_df = incident_df.loc[incident_df['Year'] == 2017]
crimes_2017_df.head()


# In[242]:


crimes_2017_df.shape


# In[243]:


data = np.array(crimes_2017_df[['X', 'Y']])


# ### 2.1 Clustering

# In[244]:


df = pd.DataFrame(data, columns = ['point_longitude_2017', 'point_latitude_2017'])

writer = ExcelWriter('out/points_2017_plot.xlsx')
df.to_excel(writer,'Sheet1', columns = ['point_longitude_2017', 'point_latitude_2017'])
writer.save()


# In[245]:


kmeans = KMeans(n_clusters=100, random_state=0).fit(data)


# ### 2.2 Predictions

# In[246]:


y_means = kmeans.predict(data)


# In[247]:


centers = kmeans.cluster_centers_


# In[248]:


data.shape, y_means.shape


# ### 2.3 Visualization

# In[249]:


plt.title('Locations of police cruisers in SF (2017)', weight ='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(centers[:, 0], centers[:, 1], c= 'black', alpha =0.5)
plt.savefig('centroids_2017.png')  


# ### 2.4 Average Time Calculation

# In[250]:


total_dist = 0
for i in range(len(y_means)):
    coords_1 = (centers[y_means[i]][1] , centers[y_means[i]][0])
    coords_2 = (data[i][1], data[i][0])
    individual_dist = great_circle(coords_1, coords_2).miles
    total_dist = total_dist + individual_dist


# In[251]:


avg_dist = total_dist/len(y_means)
avg_dist


# In[252]:


calculated_time_2017 = avg_dist *3600 / 10
calculated_time_2017


# In[253]:


y_means.shape


# ### 2.5 Saving to File

# In[254]:


df = pd.DataFrame(centers, columns = ['centroid_longitude_2017', 'centroid_latitude_2017'])


# In[255]:


writer = ExcelWriter('out/centroids_2017_check.xlsx')
df.to_excel(writer,'Sheet1', columns = ['centroid_longitude_2017', 'centroid_latitude_2017'])
writer.save()


# ## 3. 2016 Data

# In[256]:


crimes_2016_df = incident_df.loc[incident_df['Year'] == 2016]
crimes_2016_df.head()


# In[257]:


crimes_2016_df.shape


# In[258]:


data = np.array(crimes_2016_df[['X', 'Y']])


# ### 3.1 Clustering

# In[259]:


df = pd.DataFrame(data, columns = ['point_longitude_2016', 'point_latitude_2016'])

writer = ExcelWriter('out/points_2016_plot.xlsx')
df.to_excel(writer,'Sheet1', columns = ['point_longitude_2016', 'point_latitude_2016'])
writer.save()


# In[260]:


kmeans = KMeans(n_clusters=100, random_state=0).fit(data)


# ### 3.2 Predictions

# In[261]:


y_means = kmeans.predict(data)


# In[262]:


centers = kmeans.cluster_centers_


# In[263]:


data.shape, y_means.shape


# ### 3.3 Visualization

# In[264]:


plt.title('Locations of police cruisers in SF (2016)', weight ='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(centers[:, 0], centers[:, 1], c= 'black', alpha =0.5)
plt.savefig('plots/centroids_2016.png')  


# ### 3.4 Average Time Calculation

# In[265]:


total_dist = 0
for i in range(len(y_means)):
    coords_1 = (centers[y_means[i]][1] , centers[y_means[i]][0])
    coords_2 = (data[i][1], data[i][0])
    individual_dist = great_circle(coords_1, coords_2).miles
    total_dist = total_dist + individual_dist


# In[266]:


avg_dist = total_dist/len(y_means)
avg_dist


# In[267]:


calculated_time_2016 = avg_dist *3600 / 10
calculated_time_2016


# In[268]:


y_means.shape


# ### 3.5 Saving to File

# In[269]:


df = pd.DataFrame(centers, columns = ['centroid_longitude_2016', 'centroid_latitude_2016'])


# In[270]:


writer = ExcelWriter('out/centroids_2016_check.xlsx')
df.to_excel(writer,'Sheet1', columns = ['centroid_longitude_2016', 'centroid_latitude_2016'])
writer.save()


# ## 4. 2015 Data

# In[271]:


crimes_2015_df = incident_df.loc[incident_df['Year'] == 2015]
crimes_2015_df.head()


# In[272]:


crimes_2015_df.shape


# In[273]:


data = np.array(crimes_2015_df[['X', 'Y']])


# ### 4.1 Clustering

# In[274]:


df = pd.DataFrame(data, columns = ['point_longitude_2015', 'point_latitude_2015'])

writer = ExcelWriter('out/points_2015_plot.xlsx')
df.to_excel(writer,'Sheet1', columns = ['point_longitude_2015', 'point_latitude_2015'])
writer.save()


# In[275]:


kmeans = KMeans(n_clusters=100, random_state=0).fit(data)


# ### 4.2 Predictions

# In[276]:


y_means = kmeans.predict(data)


# In[277]:


centers = kmeans.cluster_centers_


# In[278]:


data.shape, y_means.shape


# ### 4.3 Visualization

# In[279]:


plt.title('Locations of police cruisers in SF (2015)', weight ='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(centers[:, 0], centers[:, 1], c= 'black', alpha =0.5)
plt.savefig('plots/centroids_2015.png')  


# ### 4.4 Average Time Calculation

# In[280]:


total_dist = 0
for i in range(len(y_means)):
    coords_1 = (centers[y_means[i]][1] , centers[y_means[i]][0])
    coords_2 = (data[i][1], data[i][0])
    individual_dist = great_circle(coords_1, coords_2).miles
    total_dist = total_dist + individual_dist


# In[281]:


avg_dist = total_dist/len(y_means)
avg_dist


# In[282]:


calculated_time_2015 = avg_dist *3600 / 10
calculated_time_2015


# In[283]:


y_means.shape


# ### 4.5 Saving to File

# In[284]:


df = pd.DataFrame(centers, columns = ['centroid_longitude_2015', 'centroid_latitude_2015'])


# In[285]:


writer = ExcelWriter('out/centroids_2015_check.xlsx')
df.to_excel(writer,'Sheet1', columns = ['centroid_longitude_2015', 'centroid_latitude_2015'])
writer.save()


# ## 5. 2014 Data

# In[286]:


crimes_2014_df = incident_df.loc[incident_df['Year'] == 2014]
crimes_2014_df.head()


# In[287]:


crimes_2014_df.shape


# In[288]:


data = np.array(crimes_2014_df[['X', 'Y']])


# ### 5.1 Clustering

# In[290]:


df = pd.DataFrame(data, columns = ['point_longitude_2016', 'point_latitude_2016'])

writer = ExcelWriter('out/points_2016_plot.xlsx')
df.to_excel(writer,'Sheet1', columns = ['point_longitude_2016', 'point_latitude_2016'])
writer.save()


# In[291]:


kmeans = KMeans(n_clusters=100, random_state=0).fit(data)


# ### 5.2 Predictions

# In[292]:


y_means = kmeans.predict(data)


# In[293]:


centers = kmeans.cluster_centers_


# In[294]:


data.shape, y_means.shape


# ### 5.3 Visualization

# In[297]:


plt.title('Locations of police cruisers in SF (2014)', weight ='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(centers[:, 0], centers[:, 1], c= 'black', alpha =0.5)
plt.savefig('plots/centroids_2014.png')  


# ### 5.4 Average Time Calculation

# In[298]:


total_dist = 0
for i in range(len(y_means)):
    coords_1 = (centers[y_means[i]][1] , centers[y_means[i]][0])
    coords_2 = (data[i][1], data[i][0])
    individual_dist = great_circle(coords_1, coords_2).miles
    total_dist = total_dist + individual_dist


# In[299]:


avg_dist = total_dist/len(y_means)
avg_dist


# In[300]:


calculated_time_2014 = avg_dist *3600 / 10
calculated_time_2014


# In[301]:


y_means.shape


# ### 5.5 Saving to File

# In[302]:


df = pd.DataFrame(centers, columns = ['centroid_longitude_2014', 'centroid_latitude_2014'])


# In[303]:


writer = ExcelWriter('out/centroids_2014_check.xlsx')
df.to_excel(writer,'Sheet1', columns = ['centroid_longitude_2014', 'centroid_latitude_2014'])
writer.save()


# ## 6. 2013 Data

# In[304]:


crimes_2013_df = incident_df.loc[incident_df['Year'] == 2013]
crimes_2013_df.head()


# In[305]:


crimes_2013_df.shape


# In[306]:


data = np.array(crimes_2013_df[['X', 'Y']])


# ### 6.1 Clustering

# In[307]:


df = pd.DataFrame(data, columns = ['point_longitude_2013', 'point_latitude_2013'])

writer = ExcelWriter('out/points_2013_plot.xlsx')
df.to_excel(writer,'Sheet1', columns = ['point_longitude_2013', 'point_latitude_2013'])
writer.save()


# In[308]:


kmeans = KMeans(n_clusters=100, random_state=0).fit(data)


# ### 6.2 Predictions

# In[309]:


y_means = kmeans.predict(data)


# In[310]:


centers = kmeans.cluster_centers_


# In[311]:


data.shape, y_means.shape


# ### 6.3 Visualization

# In[312]:


data.shape, y_means.shape


# In[313]:


plt.title('Locations of police cruisers in SF (2013)', weight ='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(centers[:, 0], centers[:, 1], c= 'black', alpha =0.5)
plt.savefig('plots/centroids_2013.png')  


# ### 6.4 Average Time Calculation

# In[314]:


total_dist = 0
for i in range(len(y_means)):
    coords_1 = (centers[y_means[i]][1] , centers[y_means[i]][0])
    coords_2 = (data[i][1], data[i][0])
    individual_dist = great_circle(coords_1, coords_2).miles
    total_dist = total_dist + individual_dist


# In[315]:


avg_dist = total_dist/len(y_means)
avg_dist


# In[316]:


calculated_time_2013 = avg_dist *3600 / 10
calculated_time_2013


# In[317]:


y_means.shape


# ### 6.5 Saving to File

# In[318]:


df = pd.DataFrame(centers, columns = ['centroid_longitude_2013', 'centroid_latitude_2013'])


# In[319]:


writer = ExcelWriter('out/centroids_2013_check.xlsx')
df.to_excel(writer,'Sheet1', columns = ['centroid_longitude_2013', 'centroid_latitude_2013'])
writer.save()


# ## Step-4: Analytics

# ### 4.1 Heatmap for crimes in 2018

# In[320]:


crimes_2018_cross_tabulate = pd.crosstab(crimes_2018_df.PdDistrict,crimes_2018_df.Category,margins=True)
del crimes_2018_cross_tabulate['All'] 
crimes_2018_cross_tabulate = crimes_2018_cross_tabulate.iloc[:-1] 


# In[321]:


column_labels_2018 = list(crimes_2018_cross_tabulate.columns.values)
row_labels_2018 = crimes_2018_cross_tabulate.index.values.tolist()


# In[322]:


#figure properties
fig,ax = plt.subplots()
heatmap = ax.pcolor(crimes_2018_cross_tabulate,cmap='Blues')
fig.set_size_inches(25,8)

# Ticks 

ax.set_yticks(np.arange(crimes_2018_cross_tabulate.shape[0])+0.5, minor=False)
ax.set_xticks(np.arange(crimes_2018_cross_tabulate.shape[1])+0.5, minor=False)

# Tick Labels
ax.set_xticklabels(column_labels_2018)
ax.set_yticklabels(row_labels_2018)
ax.invert_yaxis()
ax.xaxis.tick_top()
plt.xticks(rotation=90)

# Show & Save Plot
plt.colorbar(heatmap)
plt.savefig ("plots/Heat_map_2018.png")
plt.show()


# ### 4.2 Frequency of Crime percentage by Category

# In[323]:


Category_crimes_2018_df = pd.DataFrame(crimes_2018_df.Category.value_counts())
Category_crimes_2018_df["Percentage"] = (Category_crimes_2018_df["Category"]/Category_crimes_2018_df["Category"].sum())*100
Category_crimes_2018_df


# ### 4.3 Bar Plot by frequency

# In[324]:


fig = Category_crimes_2018_df["Percentage"].plot(kind="bar", figsize = (20,8), rot=90) 

fig.set_title("Frequency of Crime percentage by Category - 2018", fontsize=25, weight = "bold")
fig.set_xlabel("Crime Category", fontsize=18)
fig.set_ylabel("Percentage of Crimes", fontsize=18)
plt.savefig('plots/perc_crime_category_2018.png')
plt.show()


# ## Step-5: Evaluation

# In[325]:


actual_times = pd.read_excel('data/actual_time.xlsx')


# In[326]:


actual_times.head()


# In[327]:


df_2016 = actual_times.loc[actual_times['Year'] == 2016]
df_2016


# In[328]:


df_2017 = actual_times.loc[actual_times['Year'] == 2017]
df_2017


# In[329]:


df_2018 = actual_times.loc[actual_times['Year'] == 2018]
df_2018


# In[330]:


actual_time_2016 = df_2016['Time'].mean()
actual_time_2017 = df_2017['Time'].mean()
actual_time_2018 = df_2018['Time'].mean()


# In[331]:


improvement_2018 = actual_time_2018-calculated_time_2018
print("Improvement(min) in the response time:", improvement_2018/60)


# In[332]:


improvement_2017 = actual_time_2017-calculated_time_2017
print("Improvement(min) in the response time:", improvement_2017/60)


# In[333]:


improvement_2016 = actual_time_2016-calculated_time_2016
print("Improvement(min) in the response time:", improvement_2016/60)


# ### Improvement Visualization

# In[334]:


years = [2016,  2017, 2018]
actual_time = [actual_time_2016, actual_time_2017, actual_time_2018]
calculated_time = [calculated_time_2016, calculated_time_2017, calculated_time_2018]


# In[335]:


fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)

plt.plot(years, actual_time,  c='r', marker="o", label='actual', linestyle='--')
plt.plot(years, calculated_time, c='b', marker="s", label='predicted', linestyle='--')
plt.legend(loc='upper right', fontsize="medium");
plt.xticks(np.arange(min(years), max(years)+1, 1.0), weight = 'bold')
plt.yticks(np.arange(0, 450, 50), weight = 'bold')
plt.title('Year vs Actual & Predicted times', weight = 'bold', fontsize = 15)
plt.xlabel('Year', weight = 'bold', fontsize = 15)
plt.ylabel('Time', weight = 'bold', fontsize = 15)
plt.savefig('plots/Year vs Actual & Predicted times.png')
plt.show()

