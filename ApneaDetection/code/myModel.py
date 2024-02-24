import os
import mne
import numpy as np
import librosa
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional

from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import pickle

from sklearn.ensemble import RandomForestClassifier
import joblib

from sklearn.metrics import balanced_accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#######################################3
from scipy.io.wavfile import write
import random
import requests
###########################################################
#channel
#Mic
#'Effort ABD'
##################################################
##double confirm adopted
#strong#'Effort ABD': np.mean(fft_spectrum_abs[:20])  =>most <10 is 1,  >10 is 0
##########################################################
##not quite sure
#medium#'Flow Patient-0':np.mean(fft_spectrum_abs[:20])/10000 =>don't use mean=>use ratio=(after-before)/before *100 =>0:50%neg 50%pos, 1:80%neg
#medium#'Tracheal':np.mean(fft_spectrum_abs[600:1400]) =>don't use mean=>use decrease ratio, 1 has more decrease ratio =>0:50%neg 50%pos, 1:80%neg
#############################################
##############################################
##feature transformation
#mfcc=>take the suggestion to 13x4 for Mic
#fft_spectrum_abs=>for 'Effort ABD'
#np.bartlett(16)
######################################
##feature selection
#mask
#pca
#######################################
##model
#LSTM
#-original52
#-mask
#-pca

#random forest
#-original52
#-pca
##############################################

class myLSTM:
    def __init__(self):
        self.featureselect='pca'
        mode='test'
        ####################################################
        #if mode=='test':
        #    neg=10
        #    pos=1
        #else:
        #    neg=2
        #    pos=1
        neg=10
        pos=1


        total=neg+pos
        weight_for_0=(1/neg)*(total/2.0)
        weight_for_1=(1/pos)*(total/2.0)
        self.class_weight = {0:weight_for_0,1:weight_for_1}
        #######################################################
        self.model=Sequential()
        self.pca = PCA(0.95)#0.95
        self.mask=[0,2,13,15,16,18,22]
        ######################################
        #############################
        
        if self.featureselect=='pca':
            self.inputdim=4
            self.weightname='./saveweight/LSTMweightPCA.h5'
        elif self.featureselect=='mask':
            self.inputdim=len(self.mask)
            self.weightname='./saveweight/LSTMweightMASK.h5'
        elif self.featureselect=='lda':
            self.inputdim=1
            self.weightname='./saveweight/LSTMweightLDA.h5'
        else:
            self.inputdim=13*4
            self.weightname='./saveweight/LSTMweightNormal.h5'

        tf.random.set_seed(7)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy',verbose=1,patience=10,mode='max',restore_best_weights=True)
        self.model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2,return_sequences = True), input_shape = (1,self.inputdim)))

        self.model.add(LSTM(units = 50, return_sequences = True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units = 50, return_sequences = True))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.load_weights(self.weightname)
        
        #############################################################

    def fit(self,trainx,trainy):
        x=np.array(trainx).reshape(-1,1,self.inputdim)
        y=np.array(trainy).reshape(-1,1)
        
        self.model.fit(x,y,epochs=200, batch_size=64,class_weight=self.class_weight,callbacks=[self.early_stopping])#
        self.model.save_weights(self.weightname)

    def predict(self,testx):
        x=np.array(testx).reshape(-1,1,self.inputdim)
        y=self.model.predict(x)

        return y
        
##################################################
class myForest:
    def __init__(self):
        self.featureselect='lda'#pca
        mode='test'
        #####################################
        #if mode=='test':
        #    neg=10
        #    pos=1
        #else:
        #    neg=2
        #    pos=
        neg=1
        pos=1


        total=neg+pos
        weight_for_0=(1/neg)*(total/2.0)
        weight_for_1=(1/pos)*(total/2.0)
        self.class_weight = {0:weight_for_0,1:weight_for_1}
        ##################################
        self.clf = RandomForestClassifier(max_depth=2,random_state=0,class_weight=self.class_weight)
        
        ###########################
        if self.featureselect=='pca':
            self.weightname='./saveweight/random_forestPCA.joblib'
        elif self.featureselect=='lda':
            self.weightname='./saveweight/random_forestLDA.joblib'
        else:
            self.weightname='./saveweight/random_forestNormal.joblib'

        self.clf = joblib.load(self.weightname)


    def fit(self,trainx,trainy):

        x=np.array(trainx)
        y=np.array(trainy).reshape(len(trainy),)

        self.clf.fit(x,y)
        joblib.dump(self.clf, self.weightname)

    def predict(self,testx):

        y=self.clf.predict(np.array(testx))

        return y





class project:
    def __init__(self):
        self.gpath = "./trainY"
        self.gdirs = os.listdir(self.gpath)

        self.spath = "./trainX"
        self.sdirs = os.listdir(self.spath)
        #################################################
        self.gpath2 = "./testY"
        self.gdirs2 = os.listdir(self.gpath2)

        self.spath2 = "./testX"
        self.sdirs2 = os.listdir(self.spath2)

        self.windowsize=10 #seconds
        self.stepsize=5 #seconds
        self.ywindowpercent=0.5

        self.channel='Mic'
        self.modeltype='lstm'
        self.featureselect='pca'#pca
        self.model=''
        self.n_mfcc=13
        self.pca = PCA(0.95)
        self.lda=LDA(n_components=1, solver='svd', tol=0.0001)

        #################################################
        if self.featureselect=='pca':
            self.pca= pickle.load(open("./saveweight/pca.pkl",'rb'))
        elif self.featureselect=='mask':
            self.mask = np.array([True, False, True, False, False, False, False, False, False, False, False, False, False, True, False, True, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], dtype=np.bool)
        elif self.featureselect=='lda':
            self.lda= pickle.load(open("./saveweight/lda.pkl",'rb'))


        ###################################
        if self.modeltype=='lstm':
            self.model=myLSTM()
        else:
            self.model=myForest()

        #####################################        
        self.edfst=4 #file index for [005]

    ##################################################

    def gety(self,rmlPosition,totalsec):
        start=[]
        duration=[]
        #################################################################
        with open(rmlPosition,'r',encoding="utf-8") as fp:
            lines = fp.readlines()
	
        for line in lines:
            if ("Event Family=\"Respiratory\"" in line) and (("Hypopnea" in line) or ("ObstructiveApnea" in line)):
                see=("{}".format(line.strip())).split()
                ###################################################
                num=see[3].partition("Start=")[2]
                num=float(num[1:len(num)-1])
                start.append(num)
                #print(num)
                ##########################################################
                num2=see[4].partition("Duration=")[2]
                num2=float(num2[1:len(num2)-2])
                duration.append(num2)
                #print(num2)
        #################################################
        #print(start)
        #print(duration)
        ###################################################
        #[0,0,1,0] =>0:0~1 is 0, 1:1~2 is 0, 2:2~3 is 1
        #start=3600 => 3600th~3601
        #start=1 => 1th~2th
        #start=0 => 0th~1th

        #start=10,period=5 => 10~11,11~12,12~13,13~14,14~15 =>y idx=10,11,12,13,14 is 1
        #start=10.5 ,period=5.5=> 10~11,11~12,12~13,13~14,14~15,15~16=>y idx=10,11,12,13,14,15 is 1
        #start=10.5 ,period=5=> 10~11,11~12,12~13,13~14,14~15,15~16=>y idx=10,11,12,13,14,15 is 1
        #start=10 ,period=5.5=> 10~11,11~12,12~13,13~14,14~15,15~16=>y idx=10,11,12,13,14,15 is 1
        ##################################################3
        lala=[]
        ############################################
        y=np.zeros((totalsec,), dtype=int)
        for i in range(len(start)):
            st=start[i]
            if st>=3600*self.edfst:
                st=st-3600*self.edfst
                pp=duration[i]
                lala.append([st,pp])

                end=math.ceil(st+pp)#-1
                stt=math.floor(st)
                for j in range(stt,end):
                    y[j]=1
        #########################################################
        print(lala)
        finaly=[]
        for secstart in range(0,totalsec,self.stepsize):
            ywindow=y[secstart:secstart+self.windowsize]

            if sum(ywindow)/self.windowsize>self.ywindowpercent:
                finaly.append(1)
            else:
                finaly.append(0)


        return finaly

    ###########################################################
    def getfea(self,edfposition,xbuff):
        ################################################
        data=mne.io.read_raw_edf(edfposition)

        info=data.info
        sfreq=info['sfreq']
        #print(sfreq)

        channels=data.ch_names
        print(channels)
        dd=np.array(data[self.channel],dtype=object)
        first=(dd[0])[0] #dim:not necessarily 1x172800000
        
        #amplitude = np.iinfo(np.int16).max
        #ddd = amplitude *first

        ####################################################
        #print(ddd)
        #write("example.wav", int(sfreq), ddd.astype(np.int16))
        #second=dd[1]
        ##########################################################
        seconds=int(len(first)/sfreq) #not necessarily 3600

        for startsec in range(0,seconds,self.stepsize):
            stframe=int(startsec*sfreq)
            endframe=int((startsec+self.windowsize)*sfreq)

            dataPerwindow=first[stframe:endframe] #48000*windowsize frames
            xbuff=self.featureMethod(dataPerwindow,xbuff,sfreq)

            ###############################################

        return xbuff,seconds

    ################################################################

    def test(self):
        
        ###############################################
        for patientname in self.sdirs2:
            #################################
            ######################################
            totalseconds=0
            testx=[]
            patientdir=self.spath2+"/"+patientname
            patientfiles = os.listdir(patientdir)

            yhat=[]
            for edffile in patientfiles:
                edfposition=patientdir+"/"+edffile
                testx,seconds=self.getfea(edfposition,testx)
                
                #(148,4)
                
                y=self.model.predict(testx)
                #y=self.lda.predict_proba(testx)

                if self.modeltype=='lstm':
                    #print(y.shape) (148, 1, 1)
                    for p in range(y.shape[0]):
                        

                        if y[p,0,0]>0.981:#self.ywindowpercent
                            yhat.append(1)
                        else:
                            yhat.append(0)
                else:
                    yhat.extend(y)

                totalseconds+=seconds
            
            ###########################
            #print(testy)
            rmlposition=self.gpath2+"/"+patientname+".rml"
            ytruth=self.gety(rmlposition,totalseconds)
            ###################################
            #print(balanced_accuracy_score(ytruth, yhat))
            #print(len(ytruth))
            #print(len(yhat))
            #print(yhat)
            #print(ytruth)
            
            
            
            

    #######################################################################
    def featureMethod(self,data,buff,sfreq):
        ############################################3
        #tempsec=int(len(data)/sfreq)

        #weight=signal.windows.hamming(tempsec)

        #tempdata=[]

        #for k in range(len(weight)):
        #    w=weight[k]
        #    stt=int(k*sfreq)
        #    if k!=len(weight)-1:
        #        edd=int((k+1)*sfreq)
        #        mydd=data[stt:edd]
        #    else:
        #        mydd=data[stt:]

        #    ddd=np.mean(mydd)*w
        #    tempdata.append(ddd)

        ###########################################
        #mean=np.mean(tempdata) #13x1
        #sd=np.std(tempdata) #13x1
        #diffmean = (np.diff(tempdata)).mean()
        #diffsd = np.std(np.diff(tempdata))
        #########################################3
        mfccs = librosa.feature.mfcc(y=data, sr=sfreq,n_mfcc=self.n_mfcc,dct_type=2, norm='ortho',lifter=0)#(13, 94)
        #############################
        #print(mfccs.shape)
        weight=signal.windows.hamming(mfccs.shape[1])
        mean=np.average(mfccs,axis=1,weights=weight)
        #mean=mfccs.mean(axis=1) #13x1
        sd=np.std(mfccs, axis = 1) #13x1
        diffmean = (np.diff(mfccs, axis=1)).mean(axis=1)
        diffsd = np.std(np.diff(mfccs,axis=1),axis=1)
        ################################
        feat=[]
        feat.append(mean)
        feat.append(sd)
        feat.append(diffmean)
        feat.append(diffsd)
        myfeat=np.array(feat).flatten() #13x4=52
        #########################################################


        if self.featureselect=='pca':
            myfeat=(self.pca.transform(np.array(myfeat).reshape(1, -1)).tolist())[0]
            #print(myfeat)
        elif self.featureselect=='lda':
            myfeat=(self.lda.transform(np.array(myfeat).reshape(1, -1)).tolist())[0]
        elif self.featureselect=='mask':
            myfeat=myfeat[self.mask]
        
        buff.append(myfeat)

        return buff
    
    def train(self):
        #########################################
        ##only take the data before ,in, and after the event
        ###############################################
        for patientname in self.sdirs:
            #################################
            ######################################
            totalseconds=0
            patientdir=self.spath+"/"+patientname
            patientfiles = os.listdir(patientdir)
            rmlposition=self.gpath+"/"+patientname+".rml"
            ###############################
            ######################################3
            start=[]
            duration=[]
            #################################################################
            with open(rmlposition,'r',encoding="utf-8") as fp:
                lines = fp.readlines()
    
            for line in lines:
                if ("Event Family=\"Respiratory\"" in line) and (("Hypopnea" in line) or ("ObstructiveApnea" in line)):
                    see=("{}".format(line.strip())).split()
                    ###################################################
                    num=see[3].partition("Start=")[2]
                    num=float(num[1:len(num)-1])
                    start.append(num)
                    #print(num)
                    ##########################################################
                    num2=see[4].partition("Duration=")[2]
                    num2=float(num2[1:len(num2)-2])
                    duration.append(num2)
                   
            #################################################
            stc=0
            pc=0
            st=start[stc]
            du=duration[pc]
            trainx=[]
            trainy=[]
            ####################################################

            edfid=0

            for edffile in patientfiles:
                edfposition=patientdir+"/"+edffile
                data=mne.io.read_raw_edf(edfposition)

                info=data.info
                sfreq=info['sfreq']

                #channels=data.ch_names
                dd=np.array(data[self.channel],dtype=object)
                first=(dd[0])[0] #dim:not necessarily 1x172800000
                #second=dd[1]
                #########################################################
                seconds=int(len(first)/sfreq) #not necessarily 3600
                totalseconds+=seconds

                ############################
                edfstart=3600*edfid
                edfend=3600*(edfid+1)

                if ((st<edfend) and (st>edfstart)):
                    flag=True
                else:
                    flag=False

                while flag:                    
                    ##################################################3
                    st=st-3600*edfid
                    st=st-du
                    ################################################
                    tempst=st
                    tempend=tempst+du
                    if (tempst>0) and (tempst<3600) and (tempend<3600):
                        stframe=int(tempst*sfreq)
                        edframe=int(tempend*sfreq)

                        myd=first[stframe:edframe]
                        trainx=self.featureMethod(myd,trainx,sfreq)
                        trainy.append(0)
                    ##############################################
                    tempst=st+du
                    tempend=tempst+du
                    stframe=int(tempst*sfreq)
                    edframe=int(tempend*sfreq)

                    myd=first[stframe:edframe]
                    trainx=self.featureMethod(myd,trainx,sfreq)
                    trainy.append(1)
                    #################################################
                    tempst=st+2*du
                    tempend=tempst+du
                    if (tempst>0) and (tempst<3600) and (tempend<3600):
                        stframe=int(tempst*sfreq)
                        edframe=int(tempend*sfreq)

                        myd=first[stframe:edframe]
                        trainx=self.featureMethod(myd,trainx,sfreq)
                        trainy.append(0)
                    #############################################



                    ###############################################
                    #hop=int(2*du)
                    ##############################################
                    #for h in range(0,(hop+1),int(du)):
                    #    tempst=st+h
                    #    tempend=tempst+du

                    #    if (tempst>0) and (tempst<3600) and (tempend<3600):
                    #        stframe=int(tempst*sfreq)
                    #        edframe=int(tempend*sfreq)

                    #        myd=first[stframe:edframe]
                    #        trainx=self.featureMethod(myd,trainx,sfreq)
                            
                    #        if h==int(du):
                    #            trainy.append(1)
                    #        else:
                    #            trainy.append(0)
                            #if (h>int(3*du/4)) and (h<int(5*du/4)):
                            #    trainy.append(1)
                            #else:
                            #    trainy.append(0)
                    #######################################################
                    stc+=1
                    pc+=1
                    
                    if stc<len(start):
                        st=start[stc]
                        du=duration[pc]
                        if not ((st<edfend) and (st>edfstart)):
                            flag=False
                    else:
                        flag=False

                ###############################
                edfid+=1
            
            ##############################################
            #self.model.fit(trainx,trainy)
            #self.lda.fit(trainx,trainy)
            #with open('lda.pkl', 'wb') as pickle_file:
            #    pickle.dump(self.lda, pickle_file)
        #############
        #print(self.pca.n_components_)

    def LineNotify(self,timetag):
        token='xxxPut Your Line Token'
        message = 'You have Hypopnea at {t} seconds from the start.'.format(t=timetag)

        headers = { "Authorization": "Bearer " + token }
        data={'message': message }

        requests.post("https://notify-api.line.me/api/notify",headers = headers, data = data)
    
    def show(self,data,timetag):
        print('predicting')
        mfccs = librosa.feature.mfcc(y=np.array(data,dtype=float), sr=240,n_mfcc=self.n_mfcc,dct_type=2, norm='ortho',lifter=0,n_fft=10)#(13, 94)
        
        #weight=signal.windows.hamming(mfccs.shape[1])
        #mean=np.average(mfccs,axis=1,weights=weight)
        mean=mfccs.mean(axis=1) #13x1
        sd=np.std(mfccs, axis = 1) #13x1
        diffmean = (np.diff(mfccs, axis=1)).mean(axis=1)
        diffsd = np.std(np.diff(mfccs,axis=1),axis=1)
        ################################
        feat=[]
        feat.append(mean)
        feat.append(sd)
        feat.append(diffmean)
        feat.append(diffsd)
        myfeat=np.array(feat).flatten() #13x4=52
        #########################################################
        #print(np.array(myfeat))
        myfeat=(self.pca.transform(np.array(myfeat).reshape(1, -1)).tolist())[0]
        y=self.model.predict(myfeat)
        
        if y[0,0,0]>0.5:
            pre=1
            self.LineNotify(timetag)
        else:
            pre=0
        #return pre
        
#########################################################
