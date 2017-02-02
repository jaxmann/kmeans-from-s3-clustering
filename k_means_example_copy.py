#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

# $example on$
from numpy import array
from math import sqrt
# $example off$

from pyspark import SparkContext
# $example on$
from pyspark.mllib.clustering import KMeans, KMeansModel
# $example off$
from collections import defaultdict
import csv
import boto3

if __name__ == "__main__":
    sc = SparkContext(appName="KMeansExample")  # SparkContext

    # $example on$
    # Load and parse the data
    data = sc.textFile("<file on distributed file system>")
    parsedData = data.map(lambda line: array([str(x) for x in line.split(',')]))

    date_only = parsedData.map(lambda x: str(x[1])).collect() #date columns
    customer_id_only = parsedData.map(lambda x: int(x[10])).collect() #customer_id columns
    parsedData_only = parsedData.map(lambda x: array([int(x[6]) - int(x[3])])) #operation to do training on... in this case subtraction
    #for x in parsedData_only.collect():
     #   print(x)

    # Build the model (cluster the data)
    clusters = KMeans.train(parsedData_only, 2, maxIterations=10, initializationMode="random")

    result = parsedData_only.map(lambda point: clusters.predict(point)).collect()
    #print(result)
    #print(customer_id_only)

    g = open("<local file to write to>", "w")
    w = csv.writer(g, lineterminator='\n')


    cust_array = defaultdict(dict)
    j=0 # j used in line 62 for the key's specific index to be correct (the array within the key)
    for i in xrange(len(customer_id_only)):
        try:
            cust_array[str(customer_id_only[i])].append(cust_array[str(customer_id_only[i])][j] + result[i])
            w.writerow((str(customer_id_only[i]), cust_array[str(customer_id_only[i])][j] + result[i], str(date_only[i])))
            j += 1
        except (KeyError, AttributeError, IndexError) as e:
            j=0
            print("error caught")
            w.writerow((str(customer_id_only[i]), result[i], str(date_only[i])))
            cust_array[str(customer_id_only[i])] = []
            cust_array[str(customer_id_only[i])].append(result[i])

    data = open('<local file above that was written to>').read()
    s3 = boto3.resource('s3')
    s3.Bucket('<bucket-name>').put_object(Key='<s3 path>', Body=data)
    #print(cust_array)

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    WSSSE = parsedData_only.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    #print("Within Set Sum of Squared Error = " + str(WSSSE))

    # Save and load model
    #clusters.save(sc, "file:///mnt/scripts/mllib/jonathan-test/KMeansModel")
    #sameModel = KMeansModel.load(sc, "file:///mnt/scripts/mllib/jonathan-test/KMeansModel")
    # $example off$

    sc.stop()