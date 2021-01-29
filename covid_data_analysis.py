# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from numpy import nan as NaN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree


# %%
df = pd.read_excel('COVID-19-geographic-disbtribution-worldwide.xlsx')
weather = pd.read_excel('weather_data.xls', sheet_name='Country_temperatureCRU')
pkb = pd.read_excel('PKB.xls', sheet_name='Data')
pop_dens = pd.read_excel('pop_dens.xls', sheet_name='Data')
beds = pd.read_excel('beds.xls', sheet_name='Data')
medics = pd.read_excel('medics.xls', sheet_name='Data')
above65 = pd.read_excel('above65.xls', sheet_name='Data')
accidents = pd.read_excel('accidents.xls', sheet_name='Data')
air_pollution = pd.read_excel('air_pollution.xls', sheet_name='Data')


# %%
df['dateRep'] = pd.to_datetime(df['dateRep'])  
mask = (df['dateRep'] >= '2020-9-1') & (df['dateRep'] <= '2020-11-15')
sep = df.loc[mask]

mask = (df['dateRep'] >= '2020-10-1') & (df['dateRep'] <= '2020-11-15')
octo = df.loc[mask]

countries = octo['countriesAndTerritories'].unique()
geoCode = octo['countryterritoryCode'].unique()


# %%
sum_of_cases_sep = sep.groupby('countriesAndTerritories')['cases'].sum()
sum_of_cases_oct = octo.groupby('countriesAndTerritories')['cases'].sum()
sum_of_deaths_sep = sep.groupby('countriesAndTerritories')['deaths'].sum()
sum_of_deaths_oct = octo.groupby('countriesAndTerritories')['deaths'].sum()


# %%
population = sep.groupby('countriesAndTerritories')['popData2019'].unique()
continent = sep.groupby('countriesAndTerritories')['continentExp'].unique()
pop_dens = pop_dens.rename(columns={'World Development Indicators' : 'geoCode','Unnamed: 62' : 'pop_dens_2018'})


# %%
cases_per_100k = sum_of_cases_sep * 100000
cases_per_100k = cases_per_100k.div(population)
deaths_per_100k = sum_of_deaths_sep * 100000
deaths_per_100k = deaths_per_100k.div(population)


# %%
percentage_cases = sum_of_cases_oct.div(sum_of_cases_sep) * 100
percentage_deaths =  sum_of_deaths_oct.div(sum_of_deaths_sep) * 100


# %%
weather = weather.rename(columns={'ISO_3DIGIT':'geoCode', 'Sept_temp':'weather_sep', 'Oct_temp':'weather_oct'})
pkb = pkb.rename(columns={'World Development Indicators' : 'geoCode','Unnamed: 63' : 'pkb_2020'})
beds = beds.rename(columns={'World Development Indicators' : 'geoCode','Unnamed: 55' : 'num_beds'})
medics = medics.rename(columns={'World Development Indicators' : 'geoCode','Unnamed: 61' : 'num_medics'})
above65 = above65.rename(columns={'World Development Indicators' : 'geoCode','Unnamed: 63' : 'num_above65'})
accidents = accidents.rename(columns={'World Development Indicators' : 'geoCode','Unnamed: 60' : 'num_accidents'})
air_pollution = air_pollution.rename(columns={'World Development Indicators' : 'geoCode','Unnamed: 61' : 'num_air_pollution'})


# %%
column_names = ["geoCode", "weather_sep", "weather_oct"]
weather_data = pd.DataFrame(columns = column_names)
for code in geoCode:
    location = weather.loc[weather['geoCode']==code, ['geoCode','weather_sep','weather_oct']]
    if location.empty == True:
        weather_data = weather_data.append(pd.Series([code,NaN,NaN], index=['geoCode','weather_sep', 'weather_oct']), ignore_index = True)
    else:
        weather_data = weather_data.append(location)
weather_data_sep = weather_data['weather_sep'].to_numpy()
weather_data_oct = weather_data['weather_oct'].to_numpy()

weather_avg = (weather_data_sep + weather_data_oct)/2


# %%
column_names = ["geoCode", "pkb_2020"]
pkb_data = pd.DataFrame(columns = column_names)
for code in geoCode:
    location = pkb.loc[pkb['geoCode']==code, ['geoCode', 'pkb_2020']]
    if location.empty == True:
        pkb_data = pkb_data.append(pd.Series([code,NaN], index=['geoCode','pkb_2020']), ignore_index = True)
    else:
        pkb_data = pkb_data.append(location)
pkb_data_fin = pkb_data['pkb_2020'].to_numpy()


# %%
column_names = ["geoCode", "pop_dens_2018"]
pop_dens_data = pd.DataFrame(columns = column_names)
for code in geoCode:
    location = pop_dens.loc[pop_dens['geoCode']==code, ['geoCode', 'pop_dens_2018']]
    if location.empty == True:
        pop_dens_data = pop_dens_data.append(pd.Series([code,NaN], index=['geoCode','pop_dens_2018']), ignore_index = True)
    else:
        pop_dens_data = pop_dens_data.append(location)
pop_dens_data_fin = pop_dens_data['pop_dens_2018'].to_numpy()


# %%
column_names = ["geoCode", "num_beds"]
beds_data = pd.DataFrame(columns = column_names)
for code in geoCode:
    location = beds.loc[beds['geoCode']==code, ['geoCode', 'num_beds']]
    if location.empty == True:
        beds_data = beds_data.append(pd.Series([code,NaN], index=['geoCode','num_beds']), ignore_index = True)
    else:
        beds_data = beds_data.append(location)
beds_data_fin = beds_data['num_beds'].to_numpy()


# %%
column_names = ["geoCode", "num_medics"]
medics_data = pd.DataFrame(columns = column_names)
for code in geoCode:
    location = medics.loc[medics['geoCode']==code, ['geoCode', 'num_medics']]
    if location.empty == True:
        medics_data = medics_data.append(pd.Series([code,NaN], index=['geoCode','num_medics']), ignore_index = True)
    else:
        medics_data = medics_data.append(location)
medics_data_fin = medics_data['num_medics'].to_numpy()


# %%
column_names = ["geoCode", "num_above65"]
above65_data = pd.DataFrame(columns = column_names)
for code in geoCode:
    location = above65.loc[above65['geoCode']==code, ['geoCode', 'num_above65']]
    if location.empty == True:
        above65_data = above65_data.append(pd.Series([code,NaN], index=['geoCode','num_above65']), ignore_index = True)
    else:
        above65_data = above65_data.append(location)
above65_data_fin = above65_data['num_above65'].to_numpy()


# %%
column_names = ["geoCode", "num_accidents"]
accidents_data = pd.DataFrame(columns = column_names)
for code in geoCode:
    location = accidents.loc[accidents['geoCode']==code, ['geoCode', 'num_accidents']]
    if location.empty == True:
        accidents_data = accidents_data.append(pd.Series([code,NaN], index=['geoCode','num_accidents']), ignore_index = True)
    else:
        accidents_data = accidents_data.append(location)
accidents_data_fin = accidents_data['num_accidents'].to_numpy()


# %%
column_names = ["geoCode", "num_air_pollution"]
air_pollution_data = pd.DataFrame(columns = column_names)
for code in geoCode:
    location = air_pollution.loc[air_pollution['geoCode']==code, ['geoCode', 'num_air_pollution']]
    if location.empty == True:
        air_pollution_data = air_pollution_data.append(pd.Series([code,NaN], index=['geoCode','num_air_pollution']), ignore_index = True)
    else:
        air_pollution_data = air_pollution_data.append(location)
air_pollution_data_fin = air_pollution_data['num_air_pollution'].to_numpy()


# %%
new_df = {
    'cases from september': sum_of_cases_sep,
    'deaths from september' : sum_of_deaths_sep,
    'cases from october' : sum_of_cases_oct,
    'deaths from october' : sum_of_deaths_oct,
    'cases from september per 100k': cases_per_100k,
    'deaths from september per 100k' : deaths_per_100k,
    'percentage of cases (oct/sep)' : percentage_cases,
    'percentage of deaths (oct/sep)' : percentage_deaths,
    'avarage temp':weather_avg,
    'PKB per capita': pkb_data_fin,
    'Population Density':pop_dens_data_fin,
    'number of beds per 1k':beds_data_fin,
    'number of medical staff per 1k':medics_data_fin,
    'number of peaople above 65y': above65_data_fin,
    'number of traffic accidents per 1k': accidents_data_fin,
    'PM2.5 air pollution':air_pollution_data_fin
    }
new_df = pd.DataFrame(data=new_df)


# %%
#new_df = new_df.drop(index='Wallis_and_Futuna')


# %%
new_df = new_df.astype(float)
missing_data = new_df.isnull().sum()
missing_data.name = 'missing_data'

max_data = new_df.max()
max_data.name = 'max'

min_data = new_df.min()
min_data.name = 'min'

mean_data = new_df.mean(skipna=True,numeric_only=True)
mean_data.name = 'mean'

median_data = new_df.median()
median_data.name ='median'

std_deviation_data = new_df.std()
std_deviation_data.name = 'std'


lab2_df = pd.DataFrame()
lab2_df = lab2_df.append([max_data, min_data, mean_data, median_data, std_deviation_data])


# %%
# from sklearn.impute import KNNImputer
knn_df = new_df.drop(columns=['cases from september', 'deaths from september', 'cases from october', 'deaths from october', 'cases from september per 100k', 'deaths from september per 100k', 'percentage of cases (oct/sep)', 'percentage of deaths (oct/sep)'])


# %%
knn_columns = knn_df.columns
knn_index = knn_df.index

imputer = KNNImputer(n_neighbors = 4)
knn_df = imputer.fit_transform(knn_df)
knn_df = pd.DataFrame(knn_df, columns=knn_columns, index= knn_index)
new_df.update(knn_df)


# %%
std_deviation = lab2_df.loc['std']
mean_values = lab2_df.loc['mean']

sig = mean_values + (std_deviation * 3)

mask = (new_df > sig)
outliers = new_df[mask]
outliers = outliers.dropna(how='all')


# %%
#lab3 I
pca = PCA(n_components=2)
lab3_df = new_df
lab3_df = lab3_df.drop(columns=['Population Density'])
lab3_df = lab3_df.fillna(0)
lab3_df.isnull().values.any()
pca.fit(lab3_df)
pca_data = pca.transform(lab3_df)


# %%
#lab3 II
plt.figure()
lab3_2_df = pd.DataFrame(data=pca_data, index = countries, columns=["pca1", "pca2"])
lab3_2_df.plot.scatter(x = 'pca1', y = 'pca2')
plt.axis([-0.2e6 ,1.2e6, -25e4, 1e5])

plt.title('PCA')
'''
for sample in lab3_2_df.index:
    plt.annotate(sample, (lab3_2_df.pca1.loc[sample], lab3_2_df.pca2.loc[sample]))
plt.show()
'''


# %%
#lab3 III
pca_3 = PCA(n_components=2)
lab3_3_df = new_df.filter(['cases from october', 'cases from september', 'deaths from october', 'deaths from september'], axis=1)



pca_3.fit(lab3_3_df)
pca_data_3 = pca_3.transform(lab3_3_df)
lab3_3_df = pd.DataFrame(data=pca_data_3, index = countries, columns=["pca1", "pca2"])
lab3_3_df.plot.scatter(x = 'pca1', y = 'pca2')

plt.title('PCA')
#plt.axis([-0.2e6 ,1.2e6, -25e4, 1e5])

'''
for sample in lab3_3_df.index:
    plt.annotate(sample, (lab3_3_df.pca1.loc[sample], lab3_3_df.pca2.loc[sample]))
plt.show()
'''


# %%
#lab3 IV
pca_4 = PCA(n_components=15)
lab3_4_df = lab3_df
lab3_4_df = lab3_4_df.reset_index(drop=True)
pca_4.fit(lab3_4_df)
pca_data_4 = pca_4.transform(lab3_4_df)
print(pca_4.explained_variance_ratio_)


# %%
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# %%
lab4_df_all = new_df.fillna(0)
lab4_df_part = new_df.filter(['cases from october', 'cases from september', 'deaths from october', 'deaths from september'], axis=1)
clustering_all = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(lab4_df_all)
clustering_part = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(lab4_df_part)

clustering_all.labels_


# %%
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(clustering_all, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# %%
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(clustering_part, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


# %%
lab4_df_all['cluster'] = clustering_all.labels_
lab4_df_part['cluster'] = clustering_all.labels_

lab4_df_all_0 = lab4_df_all.loc[lab4_df_all['cluster'] == 0]
lab4_df_all_1 = lab4_df_all.loc[lab4_df_all['cluster'] == 1]
lab4_df_all_2 = lab4_df_all.loc[lab4_df_all['cluster'] == 2]
lab4_df_all_3 = lab4_df_all.loc[lab4_df_all['cluster'] == 3]
lab4_df_all_4 = lab4_df_all.loc[lab4_df_all['cluster'] == 4]

lab4_df_part_0 = lab4_df_part.loc[lab4_df_part['cluster'] == 0]
lab4_df_part_1 = lab4_df_part.loc[lab4_df_part['cluster'] == 1]
lab4_df_part_2 = lab4_df_part.loc[lab4_df_part['cluster'] == 2]
lab4_df_part_3 = lab4_df_part.loc[lab4_df_part['cluster'] == 3]
lab4_df_part_4 = lab4_df_part.loc[lab4_df_part['cluster'] == 4]



lab4_df_all_mean = lab4_df_all.mean(skipna=True,numeric_only=True)
lab4_df_all_0_mean = lab4_df_all_0.mean(skipna=True,numeric_only=True)
lab4_df_all_1_mean = lab4_df_all_1.mean(skipna=True,numeric_only=True)
lab4_df_all_2_mean = lab4_df_all_2.mean(skipna=True,numeric_only=True)
lab4_df_all_3_mean = lab4_df_all_3.mean(skipna=True,numeric_only=True)
lab4_df_all_4_mean = lab4_df_all_4.mean(skipna=True,numeric_only=True)
lab4_df_all_mean.name = 'all'
lab4_df_all_0_mean.name = 'cluseter0'
lab4_df_all_1_mean.name = 'cluseter1'
lab4_df_all_2_mean.name = 'cluseter2'
lab4_df_all_3_mean.name = 'cluseter3'
lab4_df_all_4_mean.name = 'cluseter4'

lab4_df_all_mean_val = pd.DataFrame()
lab4_df_all_mean_val = lab4_df_all_mean_val.append([lab4_df_all_mean, lab4_df_all_0_mean, lab4_df_all_1_mean, lab4_df_all_2_mean, lab4_df_all_3_mean, lab4_df_all_4_mean]).drop(columns = ['cluster'])


lab4_df_part_mean = lab4_df_part.mean(skipna=True,numeric_only=True)
lab4_df_part_0_mean = lab4_df_part_0.mean(skipna=True,numeric_only=True)
lab4_df_part_1_mean = lab4_df_part_1.mean(skipna=True,numeric_only=True)
lab4_df_part_2_mean = lab4_df_part_2.mean(skipna=True,numeric_only=True)
lab4_df_part_3_mean = lab4_df_part_3.mean(skipna=True,numeric_only=True)
lab4_df_part_4_mean = lab4_df_part_4.mean(skipna=True,numeric_only=True)
lab4_df_part_mean.name = 'all'
lab4_df_part_0_mean.name = 'cluseter0'
lab4_df_part_1_mean.name = 'cluseter1'
lab4_df_part_2_mean.name = 'cluseter2'
lab4_df_part_3_mean.name = 'cluseter3'
lab4_df_part_4_mean.name = 'cluseter4'

lab4_df_part_mean_val = pd.DataFrame()
lab4_df_part_mean_val = lab4_df_part_mean_val.append([lab4_df_part_mean, lab4_df_part_0_mean, lab4_df_part_1_mean, lab4_df_part_2_mean, lab4_df_part_3_mean, lab4_df_part_4_mean]).drop(columns = ['cluster'])


# %%
lab4_3_df_all = new_df.fillna(0)
lab4_3_df_part = new_df.filter(['cases from october', 'cases from september', 'deaths from october', 'deaths from september'], axis=1)


# %%
colormap = np.array(['r', 'g', 'b', 'c', 'y', 'k'])

lab3_2_df = pd.DataFrame(data=pca_data, index = countries, columns=["pca1", "pca2"])
lab3_2_df.plot.scatter(x = 'pca1', y = 'pca2', c=colormap[clustering_all.labels_])


plt.title('PCA')


# %%
colormap = np.array(['r', 'g', 'b', 'c', 'y', 'k'])
pca_3 = PCA(n_components=2)
lab3_3_df = new_df.filter(['cases from october', 'cases from september', 'deaths from october', 'deaths from september'], axis=1)



pca_3.fit(lab3_3_df)
pca_data_3 = pca_3.transform(lab3_3_df)
lab3_3_df = pd.DataFrame(data=pca_data_3, index = countries, columns=["pca1", "pca2"])
lab3_3_df.plot.scatter(x = 'pca1', y = 'pca2', c=colormap[clustering_part.labels_])

plt.title('PCA')
#plt.axis([-0.2e6 ,1.2e6, -25e4, 1e5])

'''
for sample in lab3_3_df.index:
    plt.annotate(sample, (lab3_3_df.pca1.loc[sample], lab3_3_df.pca2.loc[sample]))
plt.show()
'''


# %%
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=5)
# kmeans.fit(lab4_df_all)
# y_kmeans = kmeans.predict(lab4_df_all)

# colormap = np.array(['r', 'g', 'b', 'c', 'y', 'k'])

# lab3_2_df = pd.DataFrame(data=pca_data, index = countries, columns=["pca1", "pca2"])
# lab3_2_df.plot.scatter(x = 'pca1', y = 'pca2', c=y_kmeans, cmap='viridis')


# %%
# kmeans = KMeans(n_clusters=5)
# kmeans.fit(lab4_df_part)
# y_kmeans = kmeans.predict(lab4_df_part)

# lab3_2_df = pd.DataFrame(data=pca_data, index = countries, columns=["pca1", "pca2"])
# lab3_2_df.plot.scatter(x = 'pca1', y = 'pca2', c=y_kmeans, cmap='viridis')


# %%
lab5pre = pd.read_excel('opady.xls', sheet_name='Country_precipitationCRU')
lab5pre = lab5pre.rename(columns={'ISO_3DIGIT' : 'geoCode'})
lab5pre.set_index('geoCode',inplace=True)
lab5_pre = lab5pre[['Sept_precip','Oct_precip']]
lab5_pre['avg'] = lab5_pre.mean(axis=1)

# weather_avg = (weather_data_sep + weather_data_oct)/2

# lab5_sept_pre.sum(lab5_oct_pre)


# %%
lab5df = new_df
lab5df['geoCode'] = geoCode
lab5df.set_index('geoCode',inplace=True)
lab5df['mean_precipitation'] = lab5_pre['avg']


lab5_2d = lab5df[['mean_precipitation', 'avarage temp']]
lab5_2d = lab5_2d.dropna();

lab5_train = lab5_2d.sample(frac=0.5,random_state=200)
lab5_test = lab5_2d.drop(lab5_train.index)

lab5_train = lab5_train.dropna();
lab5_test = lab5_test.dropna();


# %%
lab5_2d.plot.scatter(x = 'mean_precipitation', y = 'avarage temp')
# plt.axis([-0.2e6 ,1.2e6, -25e4, 1e5])


# %%
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(lab5_train)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# %%
kmeans = KMeans(n_clusters=4, init='random', max_iter=300, n_init=10, random_state=42).fit(lab5_2d)

pred_y = kmeans.predict(lab5_2d)
plt.scatter(lab5_2d['mean_precipitation'], lab5_2d['avarage temp'], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()


# %%
# colormap = np.array(['r', 'g', 'b', 'c', 'y', 'k'])
# clustering_all = AgglomerativeClustering(n_clusters=4).fit(lab5_train)

# lab5_train.plot.scatter(x = 'mean_precipitation', y = 'avarage temp', c=colormap[clustering_all.labels_])


# plt.title('PCA')


# %%
X_train, X_test, y_train, y_test = train_test_split(lab5_2d, kmeans.labels_, test_size=0.50)
classifier = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

equality = np.equal(y_pred, y_test)


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)


# %%
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf)


# %%
y_pred_tree = clf.predict(X_test)
equality_tree = np.equal(y_pred_tree, y_test)


# %%



