import csv
import numpy as np
import math

# Function to read and process csv file
def Data(path):
    f = open(path, "r")
    data = csv.reader(f)
    data = np.array(list(data))
    f.close()
    
    # Removes header and id from read data
    data = np.delete(data, 0, 0)
    data = np.delete(data, 0, 1)
    
    # Shuffles and extracts trainset and testset
    np.random.shuffle(data)
    trainSet = data[:100]
    testSet = data[100:]
    return trainSet, testSet

# Function to calculate the distances
def calcDists(x1, x2, dimensions):
    distance = 0
    for i in range(1, dimensions):
        distance += (float(x1[i]) - float(x2[i])) ** 2
    return math.sqrt(distance)

# Function to find k nearest neighbors
def KNN(trainSet, point, k):
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[0],
            "value": calcDists(item, point, 31)
        })
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]
    return labels[:k]

# Function to find the most common label
def mostCommon(arr):
    labels = set(arr)
    ans = ""
    most_common = 0
    for label in labels:
        num = arr.count(label)
        if num > most_common:
            most_common = num
            ans = label
    return ans

if __name__ == "__main__":
    import os
    # Get the directory where knn.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the full path to the CSV file
    csv_path = os.path.join(current_dir, "Breast_cancer.csv")
    trainSet, testSet = Data(csv_path)
    numOfRightAnwser = 0
    for item in testSet:
        knn = KNN(trainSet, item, 3)
        answer = mostCommon(knn)
        numOfRightAnwser += item[0] == answer
        print("Diagnosis: {} -> Predicted: {}".format(item[0], answer))
        
    print("Accuracy", numOfRightAnwser / len(testSet))