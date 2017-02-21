import numpy as np
from math import log,exp
from collections import Counter
import re
import time

start_time = time.time()
#define NaiveBayes function
class NaiveBayesText:
       def init(self):
              self.MoiveReview = {}
              for label in self.labels:
                     self.MoiveReview[label] = []

       def Train(self,TrainDataSet,Label_Set):
              self.labels = np.unique(Label_Set)
              self.classprob = {}
              for label in self.labels:
                     self.classprob[label] = log(Counter(Label_Set)[label] / float(len(Label_Set)))
              self.init()
              for i in range(0,len(Label_Set)):        
                     label = Label_Set[i]
                     self.MoiveReview[label] += TrainDataSet[i]
              for label in self.labels:
                     self.MoiveReview[label] = Counter(self.MoiveReview[label])
              for label in self.labels:
                     for k in self.MoiveReview[label].keys():
                            if self.MoiveReview[0][k] <= 1 and self.MoiveReview[1][k] <= 1:
                                   del self.MoiveReview[label][k]

       def MutualInformation(self,word):
              total_neg = float(sum(self.MoiveReview[0].values()))
              total_pos = float(sum(self.MoiveReview[1].values()))
              word_neg = float(self.MoiveReview[0][word])
              word_pos = float(self.MoiveReview[1][word])
              T = total_neg + total_pos
              W = word_neg + word_pos
              MI = 0
              if W == 0:
                     return 0
              if word_neg > 0:
                     MI += (total_neg - word_neg) * log((total_neg - word_neg) * T / ((T - W) * total_neg)) / T
                     MI += word_neg * log(word_neg * T / (W * total_neg)) / T
              if word_pos > 0:
                     MI += (total_pos - word_pos) * log((total_pos - word_pos) * T / ((T - W) * total_pos)) / T
                     MI += word_pos * log(word_pos * T / (W * total_pos)) / T
              return MI
              
       def Feature(self):
              print len(self.MoiveReview[0].keys()) + len(self.MoiveReview[1].keys())
              self.words = list(set(self.MoiveReview[0].keys() + self.MoiveReview[1].keys()))
              print "Total number of features:",len(self.words)
              self.words.sort(key=lambda w: -self.MutualInformation(w)) #sort decending by MI
              print self.words[0:30]
              self.feature = set()
              start = 10000
              step = 1000
#              for w in words[:start]:
#                     self.feature.add(w)
              for k in xrange(start,39000,step):
                     for w in self.words[k:k+step]:
                            self.feature.add(w)
       
       def SingleClassify(self,SingleReview):
              prob = {}
              for label in self.labels:
                     prob[label] = self.classprob[label]
                     length = sum(self.MoiveReview[label].values())
                     review_dict = self.MoiveReview[label]
                     reviewlength = len(SingleReview)
                     for word in SingleReview:
#                            if self.MoiveReview[0][word]!=0 or self.MoiveReview[1][word]!=0:
                            word_frequency = log((review_dict[word] + 1) / (float(length+reviewlength)))
                            prob[label] += word_frequency
              values = prob.values()
              keys = prob.keys()
              return keys[values.index(max(values))]            
                            
       def NaiveBayesClassify(self, TestDataSet):
              self.prediction = []
              for i in range(0,len(TestDataSet)):
                     Review = TestDataSet[i]
                     predict_label = self.SingleClassify(Review)
                     self.prediction.append(predict_label)
              return self.prediction


#Handle negation, remove ".,:;", remain "?!"
#Remove label from dataset, remove stopword
def negated(DataSet,stopword):
       """
       Remove non alphanumeric
       Handle nagation by trasfering n't and not to not_. Double Negation also resolved
       """
       negated = False
       result = []
       for Data in DataSet:
              temp = []
              for word in Data[:-1]:
                    word = word.lower()
                    if word in [",",".",":",";"]:
                           negated = False
                           continue
                    if word in ["br"] or word in stopword:
                           continue
                    if word in ["?","!"]:
                           negated = False       
                    if negated == True:
                           word = "not_" + word
                    if any(neg in word for neg in ["not","n't","no"]):
                           negated = not negated
                           continue
                    temp.append(word)
              result.append(temp)
       return result


def NaiveBayesClassifier():
       #get train_data, test_data, train_label, test_label
       stopword = open('stopword.txt','r').read()
       stopword = stopword.split()
       training_file = open('training1.txt','r').readlines()
       train_data = [re.findall(r"[\w']+|[!?,;.:]",x) for x in training_file]
       train_label = [int(x[-1]) for x in train_data]

       testing_file = open('training2.txt','r').readlines()
       test_data = [re.findall(r"[\w']+|[!?,;.:]",x) for x in testing_file]
       test_label = [int(x[-1]) for x in test_data]

       train_data = negated(train_data,stopword)
       test_data = negated(test_data,stopword)
       print("--- %s seconds ---" % (time.time() - start_time))

#start NaiveBayesTraining
       NBC = NaiveBayesText()
       NBC.Train(train_data,train_label)
       NBC.Feature()
       test_output = NBC.NaiveBayesClassify(test_data)
       print("--- %s seconds ---" % (time.time() - start_time))
       accuracy = 0.0
       for i in range(0,len(test_output)):
              accuracy += test_output[i]==test_label[i]
       accuracy = accuracy / len(test_output)

       print accuracy
       #print test_output
       #print test_label
       print("--- %s seconds ---" % (time.time() - start_time))



