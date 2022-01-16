import os
import numpy as np
import pandas as pd

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import get_test_data
    

def run(args):
    df = pd.read_csv(os.path.join(args.result,'sentiment.csv'), index_col=0)
    data = df.to_numpy()
    data = np.array(data[:,[2,6,3,4,5]], dtype=np.int32)
    
    labels = np.array(['stars', 'sentiment', 'useful', 'funny', 'cool'])

    n_clusters = 4
    selection = [1,2]
    kmeans(data[:,selection], n_clusters, labels[selection], args)
    selection = [1,2,3]
    kmeans(data[:,selection], n_clusters, labels[selection], args)
    selection = [1,2,3,4]
    kmeans(data[:,selection], n_clusters, labels[selection], args)

# extended from a k-means example
# https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
def kmeans(data, n_clusters, labels, args):
    # get data
    assert len(data.shape) == 2
    dim = data.shape[1]
    if dim >= 2:
        X = data[:,0]
        Y = data[:,1]
    if dim >= 3:
        Z = data[:,2]
    if dim >= 4:
        C = data[:,3]

    # plot data
    if dim == 2:
        plt.scatter(X, Y)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(X, Y, Z)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    elif dim == 4:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(X, Y, Z, c=C, cmap='viridis')
        cbar = fig.colorbar(img)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        cbar.set_label(labels[3])
    plt.savefig(os.path.join(args.result, f'data_{dim}d.png'))
    #plt.show()
    plt.close()

    # elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig(os.path.join(args.result, f'elbow_{dim}d.png'))
    #plt.show()
    plt.close()

    # kmeans
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(data)

    # plot data with kmeans centers
    if dim == 2:
        plt.scatter(data[:,0], data[:,1])
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(X, Y, Z)
        X = kmeans.cluster_centers_[:, 0]
        Y = kmeans.cluster_centers_[:, 1]
        Z = kmeans.cluster_centers_[:, 2]
        img = ax.scatter(X, Y, Z, s=300, marker='x', cmap='viridis')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    elif dim == 4:        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')    
        img = ax.scatter(X, Y, Z, c=C, cmap=plt.hot())
        X = kmeans.cluster_centers_[:, 0]
        Y = kmeans.cluster_centers_[:, 1]
        Z = kmeans.cluster_centers_[:, 2]
        C = kmeans.cluster_centers_[:, 3]
        img = ax.scatter(X, Y, Z, c=C, s=300, marker='x', cmap='viridis')
        cbar = fig.colorbar(img)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        cbar.set_label(labels[3])
    plt.savefig(os.path.join(args.result, f'kmeans_{dim}d.png'))
    #plt.show()
    plt.close()
    #print(metrics.calinski_harabaz_score(X, pred_y))
