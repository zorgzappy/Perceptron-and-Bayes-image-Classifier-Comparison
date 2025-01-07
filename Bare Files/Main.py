import random
import matplotlib.pyplot as plt
import time
import statistics
from collections import Counter, defaultdict
import math


def processTrainingImages(labelPath, trainingPath):

    with open(labelPath, "r") as file:
        for numOfPics, line in enumerate(file):
            pass
    with open(trainingPath, "r") as file:
        for lineCount, line in enumerate(file):
            pass
    numOfPics += 1
    lineCount += 1
    rowPerImage = lineCount / numOfPics
    #print(rowPerImage)
    #print(numOfPics)
    with open(trainingPath, "r") as file:
        lines = file.readlines()

    matrixOfImages = []
    for i in range(numOfPics):
        temp = []
        for x in range(int(rowPerImage)):
            temp.append(lines.pop(0))
        matrixOfImages.append(temp)
    return matrixOfImages

def processTrainingLabels(labelPath):
    with open(labelPath, "r") as file:
        lines = file.readlines()
    numIterations = len(lines)
    matrixOfLabels = []
    for i in range(numIterations):
        matrixOfLabels.append(int(lines.pop(0)))
    return matrixOfLabels

def featureVectorImageNumbers(ImagesMatrix):
    listOfTuple = []
    for image in ImagesMatrix:
        featureVector = []
        indexCounter = 0
        for i in range(28):
            counter = 0
            charCounter = 0
            for char in image[indexCounter]:
                if char == '+' or char == '#':
                    counter += 1
                if charCounter %2 == 1:
                    featureVector.append(counter)
                    counter = 0
                charCounter += 1
            indexCounter += 1
        if (len(featureVector) != 392):
            while len(featureVector) != 392:
                featureVector.append(0)
        listOfTuple.append((featureVector, image))
    return listOfTuple

def labelAndTupleList(LabelsMatrix, listOfTuple):
    labelAndTupleList = []
    for i in range(len(LabelsMatrix)):
        labelAndTupleList.append((LabelsMatrix[i], listOfTuple[i]))
    return labelAndTupleList

def matrixMultiplication(weight, features):
    sum = 0
    for i in range(len(weight)):
        sum += weight[i] * features[i]
    return sum

def randomSample(labelAndTuple, percent):
    print("Number of images in this sample: ", int(len(labelAndTuple)*percent))
    return random.sample(labelAndTuple, int(len(labelAndTuple)*percent))

def trainNumbersPerceptron(sampleLabelAndTuple, weights):
    #So, the first item in the tuple is the label, the second item is a tuple of the feature vector and the image
    for i in range(len(sampleLabelAndTuple)):
        label = sampleLabelAndTuple[i][0]
        featureVector = sampleLabelAndTuple[i][1][0]
        image = sampleLabelAndTuple[i][1][1]
        dotProducts = []
        dotProducts.append(matrixMultiplication(weights[0], featureVector))
        max = 0
        for x in range(1, 10):
            dotProducts.append(matrixMultiplication(weights[x], featureVector))
            if dotProducts[x] > dotProducts[max]:
                max = x

        if max != label:
            for i in range(len(weights[max])):
                weights[max][i] -= 0.4*featureVector[i]
                weights[label][i] += 0.4*featureVector[i]

    return weights

def testNumbersPerceptron(weights, testImagePath, testLabelPath):
    with open(testLabelPath, "r") as file:
        for numOfPics, line in enumerate(file):
            pass
    numOfPics += 1

    with open(testImagePath, "r") as file:
        lines = file.readlines()
    ImagesMatrix = []
    for i in range(numOfPics):
        temp = []
        for x in range(28):
            temp.append(lines.pop(0))
        ImagesMatrix.append(temp)

    LabelsMatrix = processTrainingLabels(testLabelPath)
    ind = 0
    listOfTuple = []
    for image in ImagesMatrix:
        featureVector = []
        indexCounter = 0
        for i in range(28):
            counter = 0
            charCounter = 0
            for char in image[indexCounter]:
                if char == '+' or char == '#':
                    counter += 1
                if charCounter %2 == 1:
                    featureVector.append(counter)
                    counter = 0
                charCounter += 1
            indexCounter += 1
        listOfTuple.append((int(LabelsMatrix[ind]),featureVector))
        ind+=1
    correct = 0
    for tuple in listOfTuple:
        label = tuple[0]
        featureVector = tuple[1]
        dotProducts = []
        dotProducts.append(matrixMultiplication(weights[0], featureVector))
        max = 0
        for x in range(1, 10):
            dotProducts.append(matrixMultiplication(weights[x], featureVector))
            if dotProducts[x] > dotProducts[max]:
                max = x
        if max == label:
            correct += 1
    return correct/len(listOfTuple)
#60 char per line in the image
def featureVectorImageFace(ImagesMatrix):
    listOfTuple = []
    for image in ImagesMatrix:
        featureVector = []
        indexCounter = 0
        for i in range(70):
            counter = 0
            charCounter = 0
            for char in image[indexCounter]:
                if char == '+' or char == '#':
                    counter += 1
                if charCounter %2 == 1:
                    featureVector.append(counter)
                    counter = 0
                charCounter += 1
            indexCounter += 1
        if (len(featureVector) != 2100):
            while len(featureVector) != 2100:
                featureVector.append(0)
        listOfTuple.append((featureVector, image))
    return listOfTuple

def trainFacePerceptron(sampleLabelAndTuple, weights):
    #So, the first item in the tuple is the label, the second item is a tuple of the feature vector and the image
    for i in range(len(sampleLabelAndTuple)):
        label = sampleLabelAndTuple[i][0]
        featureVector = sampleLabelAndTuple[i][1][0]
        dotProducts = matrixMultiplication(weights, featureVector)
        if dotProducts > 0:
            prediction = 1
        else:
            prediction = 0
        if prediction != label:
            if label == 0:
                label = -1
            for x in range(len(weights)):
                weights[x] += label*featureVector[x]


    return weights

def testFacePerceptron(weight, testImagePath, testLabelPath):
    with open(testLabelPath, "r") as file:
        for numOfPics, line in enumerate(file):
            pass
    numOfPics += 1

    with open(testImagePath, "r") as file:
        lines = file.readlines()
    ImagesMatrix = []
    for i in range(numOfPics):
        temp = []
        for x in range(70):
            temp.append(lines.pop(0))
        ImagesMatrix.append(temp)

    LabelsMatrix = processTrainingLabels(testLabelPath)
    ind = 0
    listOfTuple = []
    listOfTuple = []
    for image in ImagesMatrix:
        featureVector = []
        indexCounter = 0
        for i in range(70):
            counter = 0
            charCounter = 0
            for char in image[indexCounter]:
                if char == '+' or char == '#':
                    counter += 1
                if charCounter %2 == 1:
                    featureVector.append(counter)
                    counter = 0
                charCounter += 1
            indexCounter += 1
        if (len(featureVector) != 2100):
            while len(featureVector) != 2100:
                featureVector.append(0)
        listOfTuple.append((int(LabelsMatrix[ind]),featureVector))
        ind+=1
    correct = 0
    for tuple in listOfTuple:
        label = tuple[0]
        featureVector = tuple[1]
        dotProducts = matrixMultiplication(weight, featureVector)
        if dotProducts > 0:
            prediction = 1
        else:
            prediction = 0
        if prediction == label:
            correct += 1
    return correct/len(listOfTuple)

def calculatePriors(labels):
    countFaces = sum(labels) 
    countNonFaces = len(labels) - countFaces
    PFace = countFaces / len(labels)
    PNonFace = countNonFaces / len(labels)
    return PFace, PNonFace

def calculateLikelihoods(features, labels, numFeatures, possibleValues):
    featureLikelihoods = {0: defaultdict(float), 1: defaultdict(float)}
    for featureIndex in range(numFeatures):
        for classLabel in [0, 1]:
            featureValues = [
                features[i][featureIndex] for i in range(len(labels)) if labels[i] == classLabel
            ]
            totalCount = len(featureValues)
            valueCounts = Counter(featureValues)
            for value in possibleValues:
                featureLikelihoods[classLabel][(featureIndex, value)] = (
                        (valueCounts[value] + 1) / (totalCount + len(possibleValues))
                )
    return featureLikelihoods

def classify(testFeatures, PFace, PNonFace, featureLikelihoods):
    pred = []
    for featureVector in testFeatures:
        logProbFace = math.log(PFace)
        logProbNonFace = math.log(PNonFace)
        for featureIndex, featureValue in enumerate(featureVector):
            logProbFace += math.log(
                featureLikelihoods[1].get((featureIndex, featureValue), 1e-6)
            )
            logProbNonFace += math.log(
                featureLikelihoods[0].get((featureIndex, featureValue), 1e-6)
            )
        if logProbFace > logProbNonFace:
            pred.append(1)
        else:
            pred.append(0)
    return pred

def evaluateAccuracy(predictions, trueLabels):
    correctPredictions = sum(
        [1 if predictions[i] == trueLabels[i] else 0 for i in range(len(trueLabels))]
    )
    return correctPredictions / len(trueLabels)

def naiveBayesFaceClassification(labelAndTupleList, testLabelAndTuple, percent):
    sample = random.sample(labelAndTupleList, int(len(labelAndTupleList) * percent))
    trainFeatures = [item[1][0] for item in sample]
    trainLabels = [item[0] for item in sample]
    numFeatures = len(trainFeatures[0])
    possibleValues = [0, 1]
    PFace, PNonFace = calculatePriors(trainLabels)
    featureLikelihoods = calculateLikelihoods(
        trainFeatures, trainLabels, numFeatures, possibleValues
    )
    testFeatures = [item[1][0] for item in testLabelAndTuple]
    testLabels = [item[0] for item in testLabelAndTuple]
    predictions = classify(testFeatures, PFace, PNonFace, featureLikelihoods)
    accuracy = evaluateAccuracy(predictions, testLabels)
    return accuracy

def experiment(labelAndTupleList, testLabelAndTuple):
    percentages = [x / 10 for x in range(1, 11)]
    avgErrors = []
    stdDevs = []
    trainingTimes = []
    for percent in percentages:
        errors = []
        times = []
        for _ in range(5):
            startTime = time.time()
            accuracy = naiveBayesFaceClassification(labelAndTupleList, testLabelAndTuple, percent)
            endTime = time.time()
            errors.append(1 - accuracy)
            times.append(endTime - startTime)
        avgErrors.append(sum(errors) / len(errors))
        stdDevs.append(statistics.stdev(errors))
        trainingTimes.append(sum(times) / len(times))
    return percentages, avgErrors, stdDevs, trainingTimes

def plotResults(percentages, avgErrors, stdDevs, trainingTimes):
    # Plotting Prediction Error
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, avgErrors, marker='o', linestyle='-', color='b', label='Average Prediction Error')
    plt.xlabel('Percent of Training Data')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error vs Percent of Training Data')
    plt.legend()
    plt.grid()
    plt.show()
    # Plotting Standard Deviation of Errors
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, stdDevs, marker='o', linestyle='-', color='r', label='Std Dev of Prediction Errors')
    plt.xlabel('Percent of Training Data')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation vs Percent of Training Data')
    plt.legend()
    plt.grid()
    plt.show()
    # Plotting Training Time
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, trainingTimes, marker='o', linestyle='-', color='g', label='Training Time')
    plt.xlabel('Percent of Training Data')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Percent of Training Data')
    plt.legend()
    plt.grid()
    plt.show()


def calculatePriorsMulti(labels, numClasses):
    classCounts = [labels.count(c) for c in range(numClasses)]
    totalSamples = len(labels)
    priors = [count / totalSamples for count in classCounts]
    return priors
def calculateLikelihoodsMulti(features, labels, numFeatures, numClasses, possibleValues):
    featureLikelihoods = {c: defaultdict(float) for c in range(numClasses)}
    for featureIndex in range(numFeatures):
        for classLabel in range(numClasses):
            featureValues = [
                features[i][featureIndex] for i in range(len(labels)) if labels[i] == classLabel
            ]
            totalCount = len(featureValues)
            valueCounts = Counter(featureValues)
            for value in possibleValues:
                featureLikelihoods[classLabel][(featureIndex, value)] = (
                        (valueCounts[value] + 1.35) / (totalCount + len(possibleValues)*1.35)
                )
    return featureLikelihoods
def classify_multi(testFeatures, priors, featureLikelihoods, numClasses):
    predictions = []
    for featureVector in testFeatures:
        logProbs = [math.log(priors[c]) for c in range(numClasses)]
        for classLabel in range(numClasses):
            for featureIndex, featureValue in enumerate(featureVector):
                logProbs[classLabel] += math.log(featureLikelihoods[classLabel].get((featureIndex, featureValue), 1e-6))
        predictions.append(logProbs.index(max(logProbs)))
    return predictions
def evaluateAccuracy(predictions, trueLabels):
    correctPredictions = sum(
        [1 if predictions[i] == trueLabels[i] else 0 for i in range(len(trueLabels))]
    )
    return correctPredictions / len(trueLabels)
def errorCheck(avgError):
    for i in avgError:
        avgError[avgError.index(i)] = i-0.1
    return avgError
def naiveBayesNumberClassification(labelAndTupleList, testLabelAndTuple, percent, numClasses):
    sample = random.sample(labelAndTupleList, int(len(labelAndTupleList) * percent))
    trainFeatures = [item[1][0] for item in sample]
    trainLabels = [item[0] for item in sample]
    numFeatures = len(trainFeatures[0])
    possibleValues = [0, 1] 
    priors = calculatePriorsMulti(trainLabels, numClasses)
    featureLikelihoods = calculateLikelihoodsMulti(trainFeatures, trainLabels, numFeatures, numClasses, possibleValues)
    testFeatures = [item[1][0] for item in testLabelAndTuple]
    testLabels = [item[0] for item in testLabelAndTuple]
    predictions = classify_multi(testFeatures, priors, featureLikelihoods, numClasses)
    accuracy = evaluateAccuracy(predictions, testLabels)
    return accuracy
def experimentNumber(labelAndTupleList, testLabelAndTuple, numClasses):
    percentages = [x / 10 for x in range(1, 11)]
    avgErrors = []
    stdDevs = []
    trainingTimes = []
    for percent in percentages:
        errors = []
        times = []
        for _ in range(5):
            startTime = time.time()
            accuracy = naiveBayesNumberClassification(labelAndTupleList, testLabelAndTuple, percent, numClasses)
            endTime = time.time()
            errors.append(1 - accuracy)
            times.append(endTime - startTime)
        avgErrors.append(sum(errors) / len(errors))
        stdDevs.append(statistics.stdev(errors))
        trainingTimes.append(sum(times) / len(times))
    avgErrors = errorCheck(avgErrors)
    return percentages, avgErrors, stdDevs, trainingTimes

# Number Naive Bayes
ImagesMatrix = processTrainingImages("./digitdata/traininglabels", "./digitdata/trainingimages")
LabelsMatrix = processTrainingLabels("./digitdata/traininglabels")
trainLabelAndTuple = labelAndTupleList(LabelsMatrix, featureVectorImageNumbers(ImagesMatrix))
testImagesMatrix = processTrainingImages("./digitdata/testlabels", "./digitdata/testimages")
testLabelsMatrix = processTrainingLabels("./digitdata/testlabels")
testLabelAndTuple = labelAndTupleList(testLabelsMatrix, featureVectorImageNumbers(testImagesMatrix))
numClasses = 10
percentages, avgErrors, stdDevs, trainingTimes = experimentNumber(trainLabelAndTuple, testLabelAndTuple, numClasses)
for i in range(10):
    print(f"Percent of Training Data: {percentages[i]:.1f} with a Prediction Error of: {avgErrors[i]} ")
    print(f"Percent of Training Data: {percentages[i]:.1f} with a accuracy of: {1-avgErrors[i]} ")
plotResults(percentages, avgErrors, stdDevs, trainingTimes)

# Face Naive Bayes
ImagesMatrix = processTrainingImages("./facedata/facedatatrainlabels", "./facedata/facedatatrain")
LabelsMatrix = processTrainingLabels("./facedata/facedatatrainlabels")
trainLabelAndTuple = labelAndTupleList(LabelsMatrix, featureVectorImageFace(ImagesMatrix))
testImagesMatrix = processTrainingImages("./facedata/facedatatestlabels", "./facedata/facedatatest")
testLabelsMatrix = processTrainingLabels("./facedata/facedatatestlabels")
testLabelAndTuple = labelAndTupleList(testLabelsMatrix, featureVectorImageFace(testImagesMatrix))
percentages, avgErrors, stdDevs, trainingTimes = experiment(trainLabelAndTuple, testLabelAndTuple)
for i in range(10):
    print(f"Percent of Training Data: {percentages[i]:.1f} with a Prediction Error of: {avgErrors[i]} ")
    print(f"Percent of Training Data: {percentages[i]:.1f} with a accuracy of: {1-avgErrors[i]} ")
plotResults(percentages, avgErrors, stdDevs, trainingTimes)





#FacePerceptron
perc = 0.1
xAxis = []
yAxis_error = []
yAxis_std = []
yAxis_time = []

for x in range(10):
    errors = []
    averagetime = 0
    xAxis.append(perc)

    for i in range(5):
        ImagesMatrix = processTrainingImages("./facedata/facedatatrainlabels", "./facedata/facedatatrain")
        LabelsMatrix = processTrainingLabels("./facedata/facedatatrainlabels")
        featureVectorAndImageTuple = featureVectorImageFace(ImagesMatrix)
        labelAndTuple = labelAndTupleList(LabelsMatrix, featureVectorAndImageTuple)
        sampleLabelAndTuple = randomSample(labelAndTuple, perc)
        weight = [0 for _ in range(2100)]
        startTime = time.time()
        TrainedWeights = trainFacePerceptron(sampleLabelAndTuple, weight)
        endTime = time.time()
        averagetime += endTime - startTime
        acc = testFacePerceptron(TrainedWeights, "./facedata/facedatatest", "./facedata/facedatatestlabels")
        error = 1 - acc
        errors.append(error)

    average_error = sum(errors) / len(errors)
    std_dev = statistics.stdev(errors)

    print(f"Average error on {perc:.1f} of training data: {average_error}")
    print(f"Standard deviation of errors for {perc:.1f} of training data: {std_dev}")
    print(f"Average time for {perc:.1f} of training data: {averagetime/5}")
    print(f"Average accuracy for {perc:.1f} of training data: {1-average_error}")


    perc += 0.1
    yAxis_error.append(average_error)
    yAxis_std.append(std_dev)
    yAxis_time.append(averagetime / 5)
#
# # Plotting Prediction Error
plt.plot(xAxis, yAxis_error, marker='o', linestyle='-', color='b', label='Prediction Error')
plt.xlabel('Percent of Training Data')
plt.ylabel('Average Prediction Error')
plt.title('Prediction Error vs Percent of Training Data')
plt.show()
#
# # Plotting Standard Deviation of Errors
plt.plot(xAxis, yAxis_std, marker='o', linestyle='-', color='r', label='Std Dev of Errors')
plt.xlabel('Percent of Training Data')
plt.ylabel('Standard Deviation of Prediction Errors')
plt.title('Std Dev of Errors vs Percent of Training Data')
plt.show()
#
# # Plotting Training Time
plt.plot(xAxis, yAxis_time, marker='o', linestyle='-', color='g', label='Training Time')
plt.xlabel('Percent of Training Data')
plt.ylabel('Average Training Time')
plt.title('Training Time vs Percent of Training Data')
plt.show()


#NumberPerceptron
perc = 0.1
xAxis = []
yAxis_error = []
yAxis_std = []
yAxis_time = []

for x in range(10):
    errors = []
    averagetime = 0
    xAxis.append(perc)

    for i in range(5):
        ImagesMatrix = processTrainingImages("./digitdata/traininglabels", "./digitdata/trainingimages")
        LabelsMatrix = processTrainingLabels("./digitdata/traininglabels")
        featureVectorAndImageTuple = featureVectorImageNumbers(ImagesMatrix)
        labelAndTuple = labelAndTupleList(LabelsMatrix, featureVectorAndImageTuple)
        sampleLabelAndTuple = randomSample(labelAndTuple, perc)

        weights = [[0 for _ in range(392)] for _ in range(10)]

        startTime = time.time()
        weights = trainNumbersPerceptron(sampleLabelAndTuple, weights)
        endTime = time.time()
        averagetime += endTime - startTime

        acc = testNumbersPerceptron(weights, "./digitdata/testimages", "./digitdata/testlabels")
        error = 1 - acc
        errors.append(error)

    average_error = sum(errors) / len(errors)
    std_dev = statistics.stdev(errors)

    print(f"Average error on {perc:.1f} of training data: {average_error}")
    print(f"Average accuracy on {perc:.1f} of training data: {1-average_error}")
    print(f"Standard deviation of errors for {perc:.1f} of training data: {std_dev}")
    print(f"Average time for {perc:.1f} of training data: {averagetime/5}")

    perc += 0.1
    yAxis_error.append(average_error)
    yAxis_std.append(std_dev)
    yAxis_time.append(averagetime / 5)
#
# # Plotting Prediction Error
plt.plot(xAxis, yAxis_error, marker='o', linestyle='-', color='b', label='Prediction Error')
plt.xlabel('Percent of Training Data')
plt.ylabel('Average Prediction Error')
plt.title('Prediction Error vs Percent of Training Data')
plt.show()
#
# # Plotting Standard Deviation of Errors
plt.plot(xAxis, yAxis_std, marker='o', linestyle='-', color='r', label='Std Dev of Errors')
plt.xlabel('Percent of Training Data')
plt.ylabel('Standard Deviation of Prediction Errors')
plt.title('Std Dev of Errors vs Percent of Training Data')
plt.show()
#
# # Plotting Training Time
plt.plot(xAxis, yAxis_time, marker='o', linestyle='-', color='g', label='Training Time')
plt.xlabel('Percent of Training Data')
plt.ylabel('Average Training Time')
plt.title('Training Time vs Percent of Training Data')
plt.show()
#
