from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
#for every spark program we should import these two libraries
from pyspark import SparkConf, SparkContext
from numpy import array

#setting up our SparkContext and local Spark configuration
#SparkConf = Spark configuration
#setting up master node to local as i'm going to run it on my local pc, if you run on cluster, have to set master node as cluster
conf = SparkConf().setMaster("local").setAppName("PracticeDecisionTree")
#creating Spark object(i called it here as sc) for creating RDD's
sc = SparkContext(conf = conf)

#Load up our CSV file, filter out the header with column names
#way of loading files in Spark
rawData = sc.textFile("D:/DataScience/DataScience-Python3/PastHires.csv")
#exstracting first row from that file to get only headers using first function
header = rawData.first()
#getting other information except the header row using filter function (here rawData contains only data without header line)
rawData = rawData.filter(lambda x:x != header)

#split each line into a list based on the comma delimeters using map function
csvData = rawData.map(lambda x: x.split(",")) #splits everyline into fields based on commas, here is our new RDD called csvData

#convert our csv input data into numeric
#features for each job candidate
def binary(YN):
    if(YN == 'Y'):
        return 1
    else:
        return 0

def mapEducation(degree):
    if(degree == "BS"):
        return 1
    elif(degree == 'MS'):
        return 2
    elif(degree == 'PhD'):
        return 3
    else:
        return 0
        

#Converting a list of raw fields from our csv file into a LabeledPoint that Mllib can use.All data must be numercial
def createLabeledPoints(feilds):
    yearsExperience = int(feilds[0])
    employed = binary(feilds[1])
    previousEmployers = binary(feilds[2])
    educationLevel = mapEducation(feilds[3])
    topCollege = binary(feilds[4])
    Interned = binary(feilds[5])
    hired = binary(feilds[6])
    
    return LabeledPoint(hired, array([yearsExperience, employed, previousEmployers, educationLevel, topCollege, Interned]))
    

#inorder to use Decision tress, certain things need to be done
#input should be in form of labeled data types and all has to be numeric 
#so that mllib can consume this data and can perform operations
#so we should convert these lists into LabeledPoints
trainingData = csvData.map(createLabeledPoints) #this is the one Mllib want to contruct a Decision Tree
    
    
#Create a Test Candidate
testCandidates = [array([5,1,2,2,0,1])]
#convert that array into lists
testData = sc.parallelize(testCandidates)

#Trin our Decision Tree Classifier using our data set
model = DecisionTree.trainClassifier(trainingData, numClasses = 2, #as we are making Yes/no PREDICTIONS
                                     categoricalFeaturesInfo={1:2, 3:4, 4:2, 5:2}, #feilds to number of categories
                                     impurity = "gini", #measures entropy goes down in decision tree
                                     maxDepth=5, maxBins=32)
                                    
#now get predictions for unknown candidates
#(note: we could seperate our source data into training set and a test set while tuning parameters and measure accuracy)

#let's make predictions on our test data      
Predictions = model.predict(testData)
                            
print("Hire Predictions :")
#action goes here
results = Predictions.collect()#collect function will return python object
for result in results:
    print (result) #print results of prediction
    
#we can also print out the decision tree itself
print("Learned Classification Tree Model :")
print(model.toDebugString())

#to run this, tools->canopy command prompt (when it opens, need to change to working directory) then spark-submit AppName
    
    
    

