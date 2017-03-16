# -*- coding: utf-8 -*-
from gensim.models import doc2vec 

class DataPrepare:

#[["Referandumda",""],["Cinemaximum"]
    
    def __init__ (self,s):
        self.model = None 
        self.split_s = []
        self.s = s

    def convertSentence(self):
        
        for i in range(0,len(self.s)):
            self.split_s.append(doc2vec.LabeledSentence(words=self.s[i].split(),tags=["Tweet" + str(i)]))
   
    def createModel(self):
        print (self.split_s)

        self.model = doc2vec.Doc2Vec(self.split_s,size=100,window=3,min_count=1,workers=5)

        #self.model.train(self.split_s)
 
        print (self.model.docvecs[0].shape)

        #print (self.model["Referandumda"])
    def convertToVector(self,sentence):
        deneme = doc2vec.LabeledSentence(words=sentence.split(),tags="Testeet")
        return self.model.infer_vector(["sinema","bileti"])
        

d = DataPrepare(["Referandumda ne olacak?","Bedava sinema bileti"])
d.convertSentence()
d.createModel()
v1 = d.convertToVector("bileti sinema")
v2 = d.convertToVector("Referandumda olacak")

