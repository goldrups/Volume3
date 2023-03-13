# solutions.py

import pyspark
from pyspark.sql import SparkSession
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as MCE



# --------------------- Resilient Distributed Datasets --------------------- #

### Problem 1
def word_count(filename='huck_finn.txt'):
    """
    A function that counts the number of occurrences unique occurrences of each
    word. Sorts the words by count in descending order.
    Parameters:
        filename (str): filename or path to a text file
    Returns:
        word_counts (list): list of (word, count) pairs for the 20 most used words
    """ 
    #open spark sesh
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()

    hf = spark.sparkContext.textFile(filename)
    hf = hf.flatMap(lambda row: row.split())
    hf = hf.map(lambda row: (row,1))
    hf = hf.reduceByKey(lambda x,y: x + y)
    hf = hf.sortBy(lambda row: row[1])
    return_var = hf.sortBy(lambda row: row[1], ascending=False).collect()[:20] #first 20 most common words
    spark.stop() #closed!

    return return_var   
    
### Problem 2
def monte_carlo(n=10**5, parts=6):
    """
    Runs a Monte Carlo simulation to estimate the value of pi.
    Parameters:
        n (int): number of sample points per partition
        parts (int): number of partitions
    Returns:
        pi_est (float): estimated value of pi
    """
    #open the spark session
    spark = SparkSession\
            .builder\
            .appName("app_name")\
            .getOrCreate()

    #partition into parts partitions
    pts = spark.sparkContext.parallelize(np.random.uniform([-1,-1],[1,1],size=(n*parts,2)),parts)
    in_circle = pts.map(lambda row: 1 if np.linalg.norm(row,ord=2) <= 1 else 0) #norm <= 1
    total_in_circle = in_circle.reduce(lambda x,y: x + y) #add up no. of pts in circle
    spark.stop()

    pi_est = 4*(total_in_circle / (n*parts)) #square has area 2*2=4

    return pi_est


# ------------------------------- DataFrames ------------------------------- #

### Problem 3
def titanic_df(filename='titanic.csv'):
    """
    Calculates some statistics from the titanic data.
    
    Returns: the number of women on-board, the number of men on-board,
             the survival rate of women, 
             and the survival rate of men in that order.
    """
    spark = SparkSession\
          .builder\
          .appName("app_name")\
          .getOrCreate()

    schema = ('survived INT, pclass INT, name STRING, sex STRING, age FLOAT, sibsp INT, parch INT, fare FLOAT')
    titanic = spark.read.csv("titanic.csv",schema=schema)
    survival = titanic.groupBy('sex','survived').count()

    data = survival.collect()

    spark.stop()

    num_women, num_men = 0.,0. #start out with floats
    num_women += sum([row['count'] for row in data if row['sex'] == 'female']) #filter by male, female
    num_men += sum([row['count'] for row in data if row['sex'] == 'male'])
    
    #filter by male,female survivors
    women_survive = sum([row['count'] for row in data if row['sex'] == 'female' and row['survived'] == 1])
    women_survival_rate = women_survive / num_women
    men_survive = sum([row['count'] for row in data if row['sex'] == 'male' and row['survived'] == 1])
    men_survival_rate = men_survive / num_men

    answer = (num_women, num_men, women_survival_rate, men_survival_rate)
    return answer


### Problem 4
def crime_and_income(crimefile='london_crime_by_lsoa.csv',
                     incomefile='london_income_by_borough.csv', major_cat='Robbery'):
    """
    Explores crime by borough and income for the specified major_cat
    Parameters:
        crimefile (str): path to csv file containing crime dataset
        incomefile (str): path to csv file containing income dataset
        major_cat (str): major or general crime category to analyze
    returns:
        numpy array: borough names sorted by percent months with crime, descending
    """
    spark = SparkSession\
          .builder\
          .appName("app_name")\
          .getOrCreate()

    #read in the framez
    schema_income = ('borough STRING, mean FLOAT, median FLOAT')
    income = spark.read.csv(incomefile,header = True, inferSchema = True)

    schema_crime = ('lsoa_code STRING, borough STRING, major_category STRING, minor_category STRING, value INT, year INT, month INT')
    crime = spark.read.csv(crimefile,header = True, inferSchema = True)

    crime = crime.filter(crime.major_category == major_cat) #filter by major_cat
    crime = crime.groupBy("borough").sum("value") #groupBy borough, adding total values

    income_crime = crime.join(income, on='borough') #merge on borough name
    income_crime = income_crime.select("borough", "sum(value)", "median-08-16") #keep these columns
    income_crime = income_crime.sort("sum(value)", ascending=False) #sort by value size, descending order
    income_crime = income_crime.withColumnRenamed("sum(value)", f"{major_cat}"+"_total_crime") #rename column

    ic_arr = np.array(income_crime.collect()) #the final data array

    spark.stop()

    #visualization
    plt.scatter(ic_arr[:,2].astype(float),ic_arr[:,1].astype(float))
    plt.xlabel("Median Income (2008-2016)")
    plt.ylabel("Total Crimes (2008-2016)")
    plt.title(f"{major_cat}")
    plt.show()

    return ic_arr


### Problem 5
def titanic_classifier(filename='titanic.csv'):
    """
    Implements a classifier model to predict who survived the Titanic.
    Parameters:
        filename (str): path to the dataset
    Returns:
        metrics (tuple): a tuple of metrics gauging the performance of the model
            ('accuracy', 'weightedRecall', 'weightedPrecision')
    """
    spark = SparkSession\
          .builder\
          .appName("app_name")\
          .getOrCreate()

    #set up the data problem
    schema = ('survived INT, pclass INT, name STRING, sex STRING, ''age FLOAT, sibsp INT, parch INT, fare FLOAT')
    titanic = spark.read.csv('titanic.csv', schema=schema)

    sex_binary = StringIndexer(inputCol='sex', outputCol='sex_binary')
    onehot = OneHotEncoder(inputCols=['pclass'], outputCols=['pclass_onehot'])

    features = ['sex_binary', 'pclass_onehot', 'age', 'sibsp', 'parch', 'fare']
    features_col = VectorAssembler(inputCols=features, outputCol='features')

    pipeline = Pipeline(stages=[sex_binary, onehot, features_col])
    titanic = pipeline.fit(titanic).transform(titanic)
    titanic = titanic.drop('pclass', 'name', 'sex')
    train, test = titanic.randomSplit([0.75, 0.25], seed=11)

    #logistic regression classifier
    lr = LogisticRegression(labelCol='survived', featuresCol='features')
    paramGrid = ParamGridBuilder()\
                    .addGrid(lr.elasticNetParam, [0, 0.5, 1]).build()

    tvs = TrainValidationSplit(estimator=lr,estimatorParamMaps=paramGrid,evaluator=MCE(labelCol='survived'),trainRatio=0.75,seed=11)
    clf = tvs.fit(train)
    results = clf.bestModel.evaluate(test)
    #results.predictions.select(['survived', 'prediction']).show(5)

    #now we do with random forest classifier
    rf = RandomForestClassifier(labelCol = 'survived', featuresCol = 'features')
    paramGrid = ParamGridBuilder()\
                .addGrid(rf.maxBins, [6, 3, 12]).build()

    rf_tvs = TrainValidationSplit(estimator=rf, estimatorParamMaps=paramGrid, evaluator = MCE(labelCol='survived'),trainRatio=0.75,seed=11)


    rf_clf = rf_tvs.fit(train)
    results = rf_clf.bestModel.evaluate(test)
    accuracy = results.accuracy
    weightedRecall = results.weightedRecall
    weightedPrecision = results.weightedPrecision

    spark.stop()

    return (accuracy, weightedRecall, weightedPrecision)
