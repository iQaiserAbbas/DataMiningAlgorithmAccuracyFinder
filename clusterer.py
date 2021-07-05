import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering , KMeans
import scipy.cluster.hierarchy as shc 
from sklearn.utils import resample
from subprocess import call
from random import random


def kmeans_cluster( X_principal , n_cluster):
    X = X_principal
    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=25, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)
    
    plt.figure(figsize =(6, 6)) 
    plt.scatter(X_principal['P1'], X_principal['P2'])
    plt.title('Number of Clusters = ' + str(n_cluster))
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    name = 'kmean_clust' + str(int(random()*15)) + '.png'
    plt.savefig('static/cluster/'+name)
    return name


def kmeans(X_principal):
    wcss = []
    for i in range(1, 25):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=15, n_init=10, random_state=0)
        kmeans.fit(X_principal)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize =(6, 6))  
    plt.plot(range(1, 25), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    name = 'kmeans' + str(int(random()*15)) + '.png'
    plt.savefig('static/cluster/'+name)
    return name

def aglo(X_principal):
    #Dendogram
    plt.figure(figsize =(8, 8)) 
    plt.title('Visualising the data') 
    Dendrogram = shc.dendrogram((shc.linkage( resample(X_principal, n_samples=350, random_state=0) , method ='ward')))

    name1 = 'dendo' + str(int(random()*15)) + '.png'
    plt.savefig('static/cluster/'+name1)

    ac2 = AgglomerativeClustering(n_clusters = 3)
    
    # Visualizing the clustering 
    plt.figure(figsize =(6, 6))
    plt.title('Number of Clusters = 3')  
    plt.scatter(X_principal['P1'], X_principal['P2'],  
            c = ac2.fit_predict(X_principal), cmap ='rainbow') 

    name3 = 'aglo_3' + str(int(random()*15)) + '.png'
    plt.savefig('static/cluster/'+name3)

    ac2 = AgglomerativeClustering(n_clusters = 4)

    # Visualizing the clustering 
    plt.figure(figsize =(6, 6))
    plt.title('Number of Clusters = 4') 
    plt.scatter(X_principal['P1'], X_principal['P2'],  
            c = ac2.fit_predict(X_principal), cmap ='rainbow') 
    # plt.show()
    name4 = 'aglo_4' + str(int(random()*15)) + '.png'
    plt.savefig('static/cluster/'+name4)
    return name1 , name3 , name4

def clustering(X_principal):
    call('rm -r static/cluster/*.png',shell=True)
    dendo , algo_3 , algo_4 = aglo(X_principal)
    kmean = kmeans(X_principal)
    return { "dendo" : "cluster/"+dendo , "algo_3" : "cluster/"+algo_3 ,
             "algo_4" : "cluster/"+algo_4 , "kmean" : "cluster/"+kmean }