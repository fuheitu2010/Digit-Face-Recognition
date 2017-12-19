# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **

    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
    
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    
    self.condProb = util.Counter()
    self.prior = util.Counter()
    self.priorCount = util.Counter()

        
    numOfTraining = len(trainingLabels)
    #get the prior probability
    for label in trainingLabels:
      self.priorCount[label] += 1
    for label in self.priorCount:
      self.prior[label] = 1.0 * self.priorCount[label] / numOfTraining
    
    condProbCount = util.Counter()
    for idx in range(numOfTraining):  
          for feature in self.features:
            if trainingData[idx][feature] == 0:
              condProbCount[(trainingLabels[idx], feature, 0)] += 1
            else:
              condProbCount[(trainingLabels[idx], feature, 1)] += 1
    
    maxCount = 0
    for k in kgrid: 
        condProb = util.Counter()
        for label in self.legalLabels:
            for feature in self.features:
                summation=condProbCount[(label, feature,0)]+condProbCount[(label, feature,1)]
                condProb[(label, feature,0)] =(condProbCount[(label, feature,0)]+ k) / (summation + 2.0*k)
                condProb[(label, feature,1)] =(condProbCount[(label, feature,1)] + k) / (summation +2.0*k)
        self.condProb = condProb
        predictGuesses = self.classify(validationData)
        correctCount =  [predictGuesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
        
        if correctCount > maxCount:
            maxCount = correctCount
            bestParams = (condProb, k)
    self.condProb, self.k = bestParams
    #print self.condProb
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint Counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    "*** YOUR CODE HERE ***"
    for label in self.legalLabels:
      logJoint[label] = math.log(self.prior[label])
     
      for feature in datum:
        if datum[feature] == 0:

          logJoint[label] +=  math.log(self.condProb[(label, feature,0)])
        else:
          logJoint[label] +=  math.log(self.condProb[(label, feature,1)])
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
