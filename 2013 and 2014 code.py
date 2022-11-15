# Done in Pycharm, using regions for organization
#Analysis of NBA players using season long stats in 2014 and 2015

import numpy as np
import sklearn
from sklearn.decomposition import PCA
import csv
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


# region Data Legend
# 0 Player
# 1 Pos
# 2 Age
# 3 Tm
# 4 G
# 5 GS
# 6 MP
# 7 PER
# 8 TS %
# 9 3PAr
# 10 FTr
# 11 ORB %
# 12 DRB %
# 13 TRB %
# 14 AST %
# 15 STL %
# 16 BLK %
# 17 TOV %
# 18 USG %
# 19 OWS
# 20 DWS
# 21 WS
# 22 WS per  48
# 23 OBPM
# 24 DBPM
# 25 BPM
# 26 VORP
# 27 FG
# 28 FGA
# 29 FG %
# 30 3P
# 31 3PA
# 32 3P%
# 33 2P
# 34 2PA
# 35 2P %
# 36 eFG %
# 37 FT
# 38 FTA
# 39 FT %
# 40 ORB
# 41 DRB
# 42 TRB
# 43 AST
# 44 STL
# 45 BLK
# 46 TOV
# 47 PF
# 48 PTS

#

# region Read 2013-2014 stats csv 
print('Reading NBA season data ...')
seasonData = []
with open('2013_2014_cumulative.csv', newline='') as f:
    read_f = csv.reader(f)
    for row in read_f:
        if seasonData == []:
            seasonData.append(row)
        elif int(row[4]) > 20 and row[0] != seasonData[-1][0] and row[7]>'0':
            seasonData.append(row)


seasonArray = np.array(seasonData)
# print(np.shape(seasonArray))
seasonArray_numericOnly = np.c_[seasonArray[:,4], seasonArray[:,6], seasonArray[0:, 27:]] # Only using basic counting stats
# print(seasonArray_numericOnly)

seasonArray_float = seasonArray_numericOnly.astype(float)

# print(seasonArray_float)

# endregion

# region Run PCA
print('Running PCA ...')
pca = PCA(n_components=2)
seasonArray_float = sklearn.preprocessing.normalize(seasonArray_float, axis=0)

seasonArray_PCA = pca.fit_transform(seasonArray_float[:, :])

# print(seasonArray[:, 0])
# endregion

# region Plot PCA
print('Plotting Results of PCA ...')
fig, ax = plt.subplots(1,1, figsize=(6,6))
# ax = fig.gca(projection='3d')

X = seasonArray_PCA[:,0]
Y = seasonArray_PCA[:,1]
# Z = seasonArray_PCA[:,2]
PER = seasonArray[:, 7].astype(float)
PER = 100*PER/np.max(PER)
label = seasonArray[:, 0]
# print(PER)

#  Plot data
for x,y, lab in zip(X, Y, label):
        ax.scatter(x,y,label=lab)

# Make colormap and apply to data after plotting
colormap = cm.gist_ncar #nipy_spectral, Set1,Paired
colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax.collections))]
for t,j1 in enumerate(ax.collections):
    j1.set_color(colorst[t])

for i, txt in enumerate(label):
    ax.annotate(txt, (X[i],Y[i]), xytext = (X[i]+.005,Y[i]+.0010))

plt.title('Reduced Dimensionality Data of 2013-2014 Season, Labeled with Player Names', fontsize=18)
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
# ax.legend(fontsize='small')
# endregion

# region KMeans
est = KMeans(n_clusters = 15, n_init=50)

# Calculate KMeans
est.fit(seasonArray_PCA)
labels = est.labels_

fig1, ax1 = plt.subplots(1,1)
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

ax1.scatter( X,Y,
               c=labels.astype(np.float), cmap = colormap, edgecolor='k')
plt.title('KMeans, 15 Clusters, 50 Initializations', fontsize = 18)
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
# endregion

#region Visualize KMeans decision boundary


# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = seasonArray_PCA[:, 0].min(), seasonArray_PCA[:, 0].max()
y_min, y_max = seasonArray_PCA[:, 1].min(), seasonArray_PCA[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))

# Obtain labels for each point in mesh. Use last trained model.
Z = est.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig3, ax3 = plt.subplots(1,1, figsize=(6,6))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=colormap,
           aspect='auto', origin='lower')
X = seasonArray_PCA[:, 0]
Y = seasonArray_PCA[:, 1]
plt.plot(X, Y, 'k.', markersize=2)
# Plot the centroids as a white X
centroids = est.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
# for i, txt in enumerate(label):
#     ax3.annotate(txt, (X[i],Y[i]), xytext = (X[i]+.005,Y[i]+.0010))

plt.title('K-means clustering on the PCA-reduced Statistics from the 2013-2014 season \n'
          'Centroids are marked with white cross', fontsize = 18)
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.xlim(1.1*x_min, 1.1*x_max)
plt.ylim(1.1*y_min, 1.1*y_max)


# endregionY

# region Make 2014-2015 player: Team dict
print('Assigning PCA Stats to 2014-2015 Teams ...')
seasonData2 = []
with open('2014_2015_cumulative.csv', newline='') as f:
    read_f = csv.reader(f)
    for row in read_f:
        if seasonData2 == []:
            seasonData2.append([row[0], row[3]])
        elif row[4] >= '20' and row[0] != seasonData2[-1][0]:
            seasonData2.append([row[0], row[3]])


seasonArray2 = np.array(seasonData2)
# print(np.shape(seasonArray2))
# print(seasonArray2[0,1])

Team_stat = []
# np.shape(seasonArray2)[0]
for TeamName in seasonArray2[:,1]:
    if TeamName in Team_stat:
        pass
        # print('repeat!')
    else:
        Team_stat.append(TeamName)

Team_stat = {el:[0,0] for el in Team_stat} #dict(zip(Team_stat, [[0, 0]]*31))

pca_team = list(zip( seasonArray2[:, 1], seasonArray_PCA[:, 0], seasonArray_PCA[:, 1]))
player_pca_dict = dict(zip(seasonArray[:, 0], pca_team))
# print(player_pca_dict)
for k in player_pca_dict:
    teamStr = player_pca_dict[k][0]
    Team_stat[teamStr][0] += player_pca_dict[k][1]
    Team_stat[teamStr][1] +=  player_pca_dict[k][2]

# cnt = 1
# for k in Team_stat:
#     Team_stat[k].append(cnt)
#     cnt+=1
# endregion

# region Import 2014-2015 Schedule and Organize
print('Organizing 2014-2015 Teams ...')
WL = []
with open('2014_2015.csv', newline='') as f:
    read_f = csv.reader(f)
    for row in read_f:
        WL.append(row)


WL = [[Team_stat.get(item, item) for item in it] for it in WL]

for index, row in enumerate(WL):
    if row[1] > row[3]:
        row.append([1])
    else:
        row.append([0])
    del row[1]
    del row[2]
    # row = [sum(row[0]), sum(row[1]), row[-1][0]]
    # WL[index] = row
    WL[index] = sum(row,[])

# endregion
