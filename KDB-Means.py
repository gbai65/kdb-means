import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_mutual_info_score, adjusted_rand_score
import time
import os, random
from sklearn.neighbors import NearestNeighbors

#note: delete the data directory each time you run this

starttime = time.time()
plotNum= 0; silPlotNum = 0; iter=0
PLOT = True; DIR=''; DATASETTYPE = 'real'

def KDBMeans(inputs, k):
    r0 = calculateR0(inputs)
    localDensities = np.maximum(calculateLocalDensity(inputs, r0), 1e-10)
    distMatrix = cdist(inputs, inputs)
    selectedCentroidsIxs = selectCentroids(localDensities, distMatrix, k)
    selectedCentroidsPos = [list(inputs[ix]) for ix in selectedCentroidsIxs]
    selectedCentroidsPos = np.array(selectedCentroidsPos, dtype=np.float64)
    if PLOT: plot(inputs, selectedCentroidsPos, np.array([0 for i in range(len(inputs))], dtype=int),1)

    kMeans = KMeans(n_clusters = k, init=selectedCentroidsPos, n_init=1)
    kMeans.fit(inputs)

    centroids= kMeans.cluster_centers_
    labels = kMeans.labels_
    if PLOT: plot(inputs, centroids, labels, k)
    return kMeans
    
def selectCentroids(localDensities, distMatrix, k): #density-based k-means++
    centroidIxs = []
    firstIx = max(range(len(localDensities)), key = localDensities.__getitem__)
    centroidIxs.append(firstIx)
    for iter in range(k-1):
        minDistCentroid = np.min(distMatrix[:, centroidIxs], axis=1)
        minDistCentroid[centroidIxs] = 0 #do NOT use anything except 0!!! since it's finding max, any other value breaks this
        centroidIxs.append(np.argmax(localDensities*minDistCentroid**2))
    return centroidIxs

def calculateLocalDensity(inputs, r0):
    nn = NearestNeighbors(radius=r0)
    nn.fit(inputs)
    neighbors = nn.radius_neighbors(inputs, return_distance=False) #fix this ?
    return np.array([len(n) for n in neighbors])

def calculateR0(inputs): #based on data spread
    numSamples = len(inputs)
    sampleIdx = np.random.choice(numSamples, min(1000, numSamples), replace=False) #take a sample if dataset too big
    sampleDist = pdist(inputs[sampleIdx], metric= "euclidean")
    avgDist = np.mean(sampleDist) #calculate r0 based off average distances between nodes
    alpha = 0.2 #we should test what alpha val to use with trial-and-error
    return alpha * avgDist

def normalizeFeatures(inputs): #scikit-learn z-score normalization; (x - mean)/stdDev
    sl = StandardScaler()
    return sl.fit_transform(inputs), sl

def normalKMeans(inputs, k):
    initialCntrds = inputs[np.random.choice(len(inputs), k, replace=False)]
    if PLOT: plot(inputs, initialCntrds, np.array([0 for i in range(len(inputs))], dtype=int), 1)
    kMeans = KMeans(n_clusters = k, init= initialCntrds, n_init=1)
    kMeans.fit(inputs)
    clusters = kMeans.cluster_centers_; labels = kMeans.labels_
    if PLOT: plot(inputs, clusters, labels, k)
    return kMeans

def kMeansPP(inputs, k):
    initialKMeansPP = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=0) #set a random seed so the initial centroids are actually the initial centroids
    initialKMeansPP.fit(inputs)
    initialKMPP = initialKMeansPP.cluster_centers_
    if PLOT: plot(inputs, initialKMPP, np.array([0 for i in range(len(inputs))], dtype=int), 1)
    fullKMeansPP = KMeans(n_clusters=k, init='k-means++', random_state=0, n_init='auto') #same random seed
    fullKMeansPP.fit(inputs)
    clusters = fullKMeansPP.cluster_centers_; labels = fullKMeansPP.labels_
    if PLOT: plot(inputs, clusters, labels, k)
    return fullKMeansPP

def plot(inputs, centroids, labels, k): #make a pretty looking plot
    global plotNum
    plt.figure(figsize=(10, 6))
    for i in range(k):
        cluster_points = inputs[labels == i] #if it's an initial plot, keep all labels 0 and k=1 for nice visualization because no real clustering has happened yet
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", alpha=0.7, s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="X", s=200, label="Centroid", linewidths=2)
    plt.xlabel("Attribute 1")
    plt.ylabel("Attribute 2")
    plt.title(f"Clustering Results: {['KDB-Means Initial Centroids', 'KDB-Means Final Clusters', 'K-Means Initial Centroids', 'K-Means Final Centroids', 'K-Means++ Initial Centroids', 'K-Means++ Final Centroids'][plotNum]}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{DIR}plots/z{['KDB-Means', 'KMeans', 'KMeans++'][silPlotNum]}{plotNum}.png")
    plt.close()
    plotNum+=1

def silAnalysis(inputs, model):
    global silPlotNum
    centroids = model.cluster_centers_; labels = model.labels_
    silSamps = silhouette_samples(inputs, labels)
    avgSil = silhouette_score(inputs, labels)
    bottomLine = 10
    if PLOT:
        plt.figure(figsize=(8, 6))
        for thisLabel in range(len(np.unique(labels))):
            theseVals = [i for i in range(len(inputs)) if labels[i] == thisLabel]
            theseSamps = sorted(silSamps[theseVals])
            if not theseSamps: continue
            plt.barh(range(bottomLine, bottomLine+len(theseSamps)), theseSamps)
            bottomLine+= len(theseSamps) +10
        plt.axvline(avgSil, linestyle="--", color="red")
        plt.xlabel("Silhouette Score")
        plt.ylabel("Clusters")
        plt.title(f"Silhouette Graph: {['KDB-Means', 'KMeans', 'KMeans++'][silPlotNum]}")
        plt.tight_layout()
        plt.savefig(f"{DIR}plots/silPlot{silPlotNum}.png")
        plt.close()

    silPlotNum+=1

def finAnalysis(inputs, targets, KDBmodel, KMeansmodel, KMeansPPmodel):
    for metric in range(4):
        for ix, model in enumerate([KDBmodel, KMeansmodel, KMeansPPmodel]):
            theseLabels = model.labels_
            if not metric: thisMetric = (adjusted_rand_score(targets, theseLabels))
            if metric==1: thisMetric = (adjusted_mutual_info_score(targets, theseLabels))
            if metric==2: thisMetric = (silhouette_score(inputs, theseLabels))
            if metric==3: thisMetric = (model.n_iter_)
            with open(f"{DIR}stats/{['rand', 'MI', 'sil', 'steps'][metric]}/{['KDB', 'KM', 'KMPP'][ix]}", "a") as file: #keep appending to file instead of overwriting to collect data across runs
                file.write(str(thisMetric) +"\n")

def makeBlobs(numSamples, k):
    X, y = make_blobs(n_samples=numSamples, centers=k, n_features=2)
    df = pd.DataFrame(dict(feature1=X[:,0], feature2=X[:,1], label=y))
    df.to_csv(f'{DIR}skewedData.csv', index=False)
    if PLOT:
        #varied num samples and centers in this file to make ~25 toy datasets for data collection
        plt.figure(figsize=(10, 6))
        plt.scatter(df['feature1'], df['feature2'], c=df['label'], s=50, alpha=0.6, linewidths = 2)
        plt.xlabel("Attribute 1")
        plt.ylabel("Attribute 2")
        plt.grid(True, alpha=0.3)
        plt.title('Original Plot')
        plt.tight_layout()
        plt.savefig(f'{DIR}plots/originalPlot.png')
        plt.close()

def runModels(k):
    if DATASETTYPE!="toy": 
        from ucimlrepo import fetch_ucirepo 
        dataset = fetch_ucirepo(id=257) 
        
        trainingData = dataset.data.features.values
        targets = np.array([{"Very Low": 0, "very_low": 0, "Low": 1, "Middle": 2, "High": 3}[key] for key in dataset.data.targets.values.ravel()])
    else: 
        trainingData = pd.read_csv(f"{DIR}skewedData.csv")
        targets = trainingData["label"]
        trainingData = trainingData.drop(columns=["label"])
        trainingData = trainingData.values
    trainingData = normalizeFeatures(trainingData)[0]
    print(f"KDBMeans commencing; {round(time.time()-starttime, 3)}s elapsed")
    KDBmodel  = KDBMeans(trainingData, k)
    print(f"KDBMeans finished; {round(time.time()-starttime, 3)}s elapsed")
    silAnalysis(trainingData, KDBmodel)
    print(f"KMeans commencing; {round(time.time()-starttime, 3)}s elapsed")
    KMeansmodel  = normalKMeans(trainingData, k)
    print(f"KMeans finished; {round(time.time()-starttime, 3)}s elapsed")
    silAnalysis(trainingData, KMeansmodel)
    print(f"KMeans++ commencing; {round(time.time()-starttime, 3)}s elapsed")
    KMeansPPmodel = kMeansPP(trainingData, k)
    print(f"KMeans++ finished; {round(time.time()-starttime, 3)}s elapsed")
    silAnalysis(trainingData, KMeansPPmodel)
    finAnalysis(trainingData, targets, KDBmodel, KMeansmodel, KMeansPPmodel)

def main():
    global PLOT, DIR, iter, plotNum, silPlotNum, DATASETTYPE
    if DATASETTYPE == "toy":
        for iter in range(25):
            plotNum, silPlotNum = 0, 0
            DIR = f"data/trial{iter:02d}/"
            try:
                for ea in ["plots", "stats", "stats/rand", "stats/MI", "stats/sil", "stats/steps"]:
                    os.makedirs(DIR+ea, exist_ok=True)
            except OSError as e: print(f"Error: {e}")

            numSamples = random.randint(100, 500)
            k = random.randint(3, 10)
            if iter>10: PLOT=False
            with open(f"{DIR}info", "a") as file:
                file.write(f"N: {numSamples}, k: {k}")
            makeBlobs(numSamples, k)
            runModels( k)
    else: 
        DIR = "knowledge/"
        try:
            for ea in ["plots", "stats", "stats/rand", "stats/MI", "stats/sil", "stats/steps"]:
                os.makedirs(DIR+ea, exist_ok=True)
        except OSError as e: print(f"Error: {e}")
        runModels(4) #we're assuming we already know k from the dataset itself for the purposes of this algorithm
main()