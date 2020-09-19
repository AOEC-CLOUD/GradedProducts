#Customer HP (Honest Product) Requirement Learning Team-Suite (Proof of concept, work in progress version)
import os
import pandas as pd
import numpy as np
from numpy import array
from IPython.display import display
#For natural language processing ability
import nltk.data
from nltk.corpus import stopwords

#gensim libraries
import gensim

from gensim.models import word2vec
from gensim.test.utils import get_tmpfile
from gensim.models.keyedvectors import KeyedVectors


#to visualize the clusters
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#for clustering
from sklearn.cluster import DBSCAN
import sklearn.metrics as metrics

#to compute service request description similarity
from sklearn.metrics.pairwise import cosine_similarity

#Load Google’s pre-trained Word2Vec model known to contain 300 dimensioned vectors for # 3 million words and phrases
#model_google3M = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
path = get_tmpfile('./GoogleNews-vectors-negative300.bin')
#an empty model, no training yet
model = gensim.models.Word2Vec(iter=1)  
model.wv.save(path)

model_google3M = gensim.models.KeyedVectors.load(path,mmap='r')

#Create training data by averaging vectors for words in the short_description column
def createFeatureVec(words, model, num_features):

    #convert Index2word list to a set for speedy execution
    index2word_set = set (model.wv.index2word)

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="int")

    #
    nwords = 0.

    #loop over each word in the short_description 
    #if it is in the model’s vocabulary, add its feature vector to the total
    for word in words:
         if word in index2word_set:
            nwords  = nwords + 1.
            featureVec = np.add (featureVec, model[word])
            
    
    #divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(vShortDescription_s, model, num_features):
    #for the given set of vShortDescription calculate the average feature vector for each list of    #words and return a 2D numpy array 
    counter = 0

    #preallocate a 2D numpy array for speed in execution
    vShortDescriptionVecs = np.zeros((len(vShortDescription_s), num_features),dtype='int')

    for vShortDescription in vShortDescription_s:
        vShortDescriptionVecs[int(counter)] = createFeatureVec(vShortDescription_s, model,num_features)
        counter = counter + 1.
            
    return vShortDescriptionVecs 


def newSRClusterer(newSRText):  
    #vectorize SR text
    newSRVector = getAvgFeatureVecs (newSRText, model_google3M, 300)

    #Build the data frame with Service Request meta data and similarity scores
    f_6_1 =  'old_customer_HP_requirement_x.xlsx'
    data_6_1 = pd.read_excel (f_6_1, sheet_name='SRN')
    data_6_1.to_csv (r'old_customer_HP_requirement_x.csv', index = None, header=True)

    f_6_1 =  'old_customer_HP_requirement_x.csv'
    data_6_1 = pd.read_csv (f_6_1)
    
    data_8 = pd.concat ([pd.DataFrame(labels), data_6_1[['SR number','short_description','assignment_group']]], axis=1)
    
    data_8.rename (columns = {0:'Cluster'},inplace = True)
    data_8['similarityScore'] = cosine_similarity(data_6_1.iloc[:,26:326], newSRVector)

    # Find the cluster that the Service Request is assigned to where this is done based on the
    # maximum similarity score and averaged across all service requests in the cluster
    similarityScoreMean = data_8.groupby ('Cluster')['similarityScore'].mean().max()
    newSRCluster = data_8.groupby ('Cluster')['similarityScore'].mean().idxmax()
    
    if similarityScoreMean >= 0.7:
        #this threshold needs to be tuned to ensure noise element is not incorrectly assigned a clustered bucket
        print ('The Customer Honest Product requirement is assigned to the cluster', newSRCluster)
        print ('The Customer Honest Product requirement similarity to the assigned cluster:', round(similarityScoreMean,2))
    else:
        print ('This Customer Honest Product requirement is unlike any detail in the training repository and is not assigned to any cluster')
        
    return similarityScoreMean, newSRCluster


#Service Request processing, this array is still work in progress
assignment_group_subset =  {
'GDSN_FMCG Hub',
'Grade 1 Asset_with_score',
'Grade 1.1 Asset_with_score_and_P2PC',
'Grade 1.2 Asset_with_score_and_CVODC',
'Grade 2 Green_or_organic',
'Grade 3 Fresh_or_natural',
'Grade 4 Frozen_or_preserved',
'Grade 5 Others',
'Grade NA'
}

#data file that contains old service requests
f_1 =  'old_customer_HP_requirement_x.xlsx'
#data_1 = pd.read_excel (f_1, sheet_name='SRN', converters={'short_description':str})

data_1 = pd.read_excel (f_1, sheet_name='SRN')
data_1.to_csv (r'old_customer_HP_requirement_x.csv', index = None, header=True)

f_1 =  'old_customer_HP_requirement_x.csv'
data_1 = pd.read_csv (f_1)
    
data_1 = data_1[data_1.assignment_group.isin(assignment_group_subset)]

#clustering using DBSCAN
clustering_vec = getAvgFeatureVecs(data_1['short_description'], model_google3M, 300)

db = DBSCAN(eps=0.3, min_samples = 10).fit(clustering_vec)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

#plot result
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip (unique_labels, colors):
  if k == -1:
  #use black for noise aspect
    col = [0,0,0,1]

class_member_mask = (labels == k)
xy = pd.DataFrame( clustering_vec[class_member_mask & core_samples_mask])
#plt.plot(xy.iloc[:,0],xy.iloc[:,1],'o',markerfacecolor = tuple(col),markeredgecolor='k',makersize = 14)

xy = pd.DataFrame( clustering_vec[class_member_mask & core_samples_mask])
#plt.plot(xy.iloc[:,0],xy.iloc[:,1],'o',markerfacecolor = tuple(col),markeredgecolor='k',makersize = 1)

plt.title('Graded Product Estimated number of clusters')
plt.show()

newSRClusterer('GDSN')
