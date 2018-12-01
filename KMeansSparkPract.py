#importing necessary Libraries
from pyspark.mllib.clustering import KMeans
#MLlib often requires numpy arrays
from numpy import array, random
from math import sqrt
#regular important libraries
from pyspark import SparkConf,SparkContext
from sklearn.preprocessing import scale

#defining number of Cmusters
K = 5

#setting up local Spark Configuration
conf = SparkConf().setMaster("local").setAppName("PracticeSparkKMeans")
#creating SparkContext object which creates RDD's
sc = SparkContext(conf =conf)

#let's create fake income/age Clusters for N people in K- clusters
def createClusteredData(N,K):
    random.seed(10)
    pointsPerCluster = float(N)/K
    X = []
    for i in range(K):
        incomecentroid = random.uniform(20000.0, 2000000.0)
        agecentroid = random.uniform(20, 80)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomecentroid, 10000.0), random.normal(agecentroid, 5.0)])
            X = array(X)
            return X
            
        

#Load our data; Note: I'm normalizing data with scale() - very important!
data = sc.parallelize(scale(createClusteredData(100,K))) #here our data is RDD, scale - it's make income and age more compromising 

#build our model (cluster the data) by calling KMeans
clusters = KMeans.train(data, K, maxIterations=10, runs=10, initializationMode="random")#to pick ur initial centroid clusters randomly before we start iterating

#print out the cluster assignments for each of our points
resultRDD = data.map(lambda point: clusters.predict(point)).cache()
#this lambda function transofroms each point into the cluster number that is predicted from our model
#cache() is used to not to recompute again when you called 2 or more actions in a function

#print out how many points we have for each cluster id
print ("Count By Value")
counts = resultRDD.countByValue() #used to count for hoamany values we see for each given cluster ID
print(counts)

#to look at raw results
#let's see how theose points have distributed and display those raw results
print("Cluster Assignments")
results = resultRDD.collect()
print(results)

#Evaluation clustering-to check how good is our clusters are
#so we can evaluate that by computing within Set Sum of Squared Errors (WSSSE)
  
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum(x**2 for x in [point - center]))

#WSSSE = distance from each point to it's centroid in each cluster, take that squared error and sum it up for entire dataset
#WSSSE - it's just a measure of how far each point from cenroid is
WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y:x+y)
print("Within Set Sum of Sqaured Error :" + str(WSSSE))
    