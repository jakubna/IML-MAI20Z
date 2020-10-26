import numpy as np


def SSE(cluster):
    centroid = np.mean(cluster, 0)
    errors = np.linalg.norm(cluster-centroid, ord=2, axis=1)
    return np.sum(errors)

def bisecting_kmeans(db, k=2):
    clusters = [db]
    while len(clusters) < k:
        #calculate the clusyer with bigger error
        max_sse_i = np.argmax([SSE(c) for c in clusters])
        #select the cluster and take it off from the clusters list
        cluster = clusters.pop(max_sse_i)
        #split in 2 clusters using k_means
        kmeans = k_means(k=2)
        kmeans.fit(cluster)
        two_labels = kmeans.predict(cluster)
        #use the labels to split the data according to clusters
        two_clusters=[]
        for act in range(0, 2):
            # select the index of all points in the same cluster
            indX = np.where(two_labels == act)[0]
            cluster_x = cluster[indX, :]
            two_clusters.append(cluster_x)
        #append the clusters list
        clusters.extend(two_clusters)
    #trying to set the labels to the original dataset
    inter=[]
    for x in db:
        cat=0
        a=0
        for c in clusters:
            if cat ==0:
                for i in c:
                    if np.all(x == i):
                        cat = a
                        break                    
            a+=1
        inter.append(cat)
    labels=np.array(inter)
    return clusters, labels

