#preprocessing methods
import pandas as pd
import numpy as np
import sys

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from itertools import chain
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn import metrics

from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Dense, Embedding, Input,Flatten,multiply,Lambda,add,Reshape,CuDNNLSTM,CuDNNGRU,GRU,Conv1D,BatchNormalization
from keras.layers import LSTM, Bidirectional, Dropout,GlobalMaxPool1D,GlobalAveragePooling1D,concatenate,Activation,SpatialDropout1D
print('Preprocessing_module_import_finish')

#seed everything
import os
os.environ['PYTHONHASHSEED'] = '10000'
import random
np.random.seed(10001)
random.seed(10002)
import tensorflow as tf
tf.set_random_seed(10003)

def runtraining(model_type = 31,save_name = '',lr = 0.1,batch_size = 1000,nb_epoch = 5,is_preprocessing = True,best_threshold = None,is_GPU = False，finetune=True):
    #load
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
    def clean_contractions(text, mapping):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
        return text
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
    def clean_special_chars(text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p])

        for p in punct:
            text = text.replace(p, ' '+p+' ')

        specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
        for s in specials:
            text = text.replace(s, specials[s])

        return text
    mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}
    def correct_spelling(x, dic):
        for word in dic.keys():
            x = x.replace(word, dic[word])
        return x
    if is_preprocessing == True:
        print('doing pre-pre-processing')
        # Lowering
        train_df['treated_question'] = train_df['question_text'].apply(lambda x: x.lower())
        # Contractions
        train_df['treated_question'] = train_df['treated_question'].apply(lambda x: clean_contractions(x, contraction_mapping))
        # Special characters
        train_df['treated_question'] = train_df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
        # Spelling mistakes
        train_df['treated_question'] = train_df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))
        print('train_prepre_done')
        test_df['treated_question'] = test_df['question_text'].apply(lambda x: x.lower())
        # Contractions
        test_df['treated_question'] = test_df['treated_question'].apply(lambda x: clean_contractions(x, contraction_mapping))
        # Special characters
        test_df['treated_question'] = test_df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
        # Spelling mistakes
        test_df['treated_question'] = test_df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))
        print('test_prepre_done')

        
        ## fill up the missing values
        train_X = train_df['treated_question'].fillna("_na_").values
        test_X = test_df['treated_question'].fillna("_na_").values
        train_y = train_df['target'].values
    else:
        ## fill up the missing values
        train_X = train_df["question_text"].fillna("_na_").values
        test_X = test_df["question_text"].fillna("_na_").values
        train_y = train_df['target'].values

    #parameter
    #GLOBAL VARIABLES
    path = '../input/'
    embedding_path = '../input/embeddings/'
    maxlen = 50
    max_features = 220000
    embed_size = 300
    #num_embed_length = num_train_X.shape[1]
    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(np.concatenate((train_X,test_X))))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    word_index = tokenizer.word_index
    print(len(word_index))
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)
    print('padding finish')
    
    def load_glove(word_index):
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

        EMBEDDING_FILE = embedding_path+'glove.840B.300d/glove.840B.300d.txt'
        #embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE)  if o.split(" ")[0] in word_index )
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE)  )

        all_embs = np.stack(embeddings_index.values())
        emb_mean1,emb_std1 = all_embs.mean(), all_embs.std()

        EMBEDDING_FILE = embedding_path+'paragram_300_sl999/paragram_300_sl999.txt'
        #embeddings_index2  = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8', errors='ignore') if o.split(" ")[0] in word_index )
        embeddings_index2  = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8', errors='ignore')  )

        all_embs = np.stack(embeddings_index2.values())
        emb_mean2,emb_std2 = all_embs.mean(), all_embs.std()

        #embedding_matrix = np.zeros( (len(word_index)+2,  300 ) , dtype=np.float32)
        embedding_matrix = np.zeros( (max_features+1,  300 ) , dtype=np.float32)
        countingiter = 0
        for word, i in word_index.items():
            if countingiter >= max_features:
                return embedding_matrix
            embedding_vector = None
            if (word in embeddings_index) & (word in embeddings_index2) :
                embedding_vector = (embeddings_index.get(word) + (embeddings_index2.get(word)-emb_mean2+emb_mean1))/2.# can change
            else:            
                if (word in embeddings_index):
                    embedding_vector = embeddings_index.get(word)
                else:
                    if (word in embeddings_index2):
                        embedding_vector = embeddings_index2.get(word)-emb_mean2+emb_mean1
            #        else:
            #            embedding_vector = embeddings_index3.get(word)       
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector.astype(np.float32)
            countingiter +=1
        #return embedding_matrix
    
    class Attention(Layer):
            def __init__(self, step_dim,
                         W_regularizer=None, b_regularizer=None,
                         W_constraint=None, b_constraint=None,
                         bias=True, **kwargs):
                self.supports_masking = True
                self.init = initializers.get('glorot_uniform')

                self.W_regularizer = regularizers.get(W_regularizer)
                self.b_regularizer = regularizers.get(b_regularizer)

                self.W_constraint = constraints.get(W_constraint)
                self.b_constraint = constraints.get(b_constraint)

                self.bias = bias
                self.step_dim = step_dim
                self.features_dim = 0
                super(Attention, self).__init__(**kwargs)

            def build(self, input_shape):
                assert len(input_shape) == 3

                self.W = self.add_weight((input_shape[-1],),
                                         initializer=self.init,
                                         name='{}_W'.format(self.name),
                                         regularizer=self.W_regularizer,
                                         constraint=self.W_constraint)
                self.features_dim = input_shape[-1]

                if self.bias:
                    self.b = self.add_weight((input_shape[1],),
                                             initializer='zero',
                                             name='{}_b'.format(self.name),
                                             regularizer=self.b_regularizer,
                                             constraint=self.b_constraint)
                else:
                    self.b = None

                self.built = True

            def compute_mask(self, input, input_mask=None):
                return None

            def call(self, x, mask=None):
                features_dim = self.features_dim
                step_dim = self.step_dim

                eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                                K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

                if self.bias:
                    eij += self.b

                eij = K.tanh(eij)

                a = K.exp(eij)

                if mask is not None:
                    a *= K.cast(mask, K.floatx())

                a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

                a = K.expand_dims(a)
                weighted_input = x * a
                return K.sum(weighted_input, axis=1)

            def compute_output_shape(self, input_shape):
                return input_shape[0],  self.features_dim
            
    def BidLstm(maxlen, max_features, embed_size,embedding_matrix):
        inp = Input(shape=(maxlen, ),name="text_input")
        x = Embedding(max_features+1, embed_size,input_length = maxlen, weights=[embedding_matrix],trainable=False,name='emb')(inp)
        if model_type == 31:
            if is_GPU == False:
                x1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1,
                                       recurrent_dropout=0.1))(x)
                x2 = Bidirectional(GRU_layer(128, return_sequences=True, dropout=0.1,
                                       recurrent_dropout=0.1))(x1)
            else:
                x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
                x1 = SpatialDropout1D(0.1)(x1)
                x2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x1)
                x2 = SpatialDropout1D(0.1)(x2)
            x1_att = Attention(maxlen)(x1)
            x2_att = Attention(maxlen)(x2)
            x2_avg = GlobalAveragePooling1D()(x2)
            x2_max = GlobalMaxPool1D()(x2)
            x = concatenate([x1_att,x2_att,x2_avg,x2_max])
            x = Dense(16, activation="relu")(x)
            x = Dense(1, activation="sigmoid")(x)
            #model = Model(inputs=[inp,inp_num], outputs=x)
            model = Model(inputs=inp, outputs=x)
        if model_type == 32:
            avg_emb = Lambda(lambda x : K.sum(x, axis=1))(x)
            avg_emb = BatchNormalization()(avg_emb)
            if is_GPU == False:
                lstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1,
                                       recurrent_dropout=0.1))(x)
            else:
                lstm = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
                lstm = SpatialDropout1D(0.1)(lstm)


            max_lstm = GlobalMaxPool1D( )(lstm)
            avg_lstm = Lambda(lambda x : K.sum(x, axis=1))(lstm)
            avg_lstm = BatchNormalization()(avg_lstm)

            # main layers
            main_l = concatenate([max_lstm, avg_lstm, avg_emb])#, max_lstm2, avg_lstm2
            main_l = Dense(196)(main_l)
            main_l = Activation('relu')(main_l)
            main_l = Dense(64)(main_l)
            main_l = Activation('relu')(main_l)
            main_l = Dense(1, activation="sigmoid")(main_l)
            model = Model(inputs=inp, outputs=main_l)
        return model
    
    #makeglove
    embedding_matrix = load_glove(word_index)
    #modeling
    print('startmodeling')
    #model.fit(train_X,train_y , batch_size=batch_size, epochs=nb_epoch, verbose=1)
    if best_threshold == None:
        
        kfold = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
        f1_score_matrix = []
        for train_index, valid_index in kfold.split(train_X, train_y):
            X_train, X_val, Y_train,Y_val = train_X[train_index], train_X[valid_index], train_y[train_index], train_y[valid_index]
            #X_train_num, X_val_num = num_train_X[train_index], num_train_X[valid_index]
            model = BidLstm(maxlen, max_features, embed_size,embedding_matrix)
            model.compile(loss='binary_crossentropy',
                          optimizer = Adam(lr = lr),
                          metrics=['accuracy'])
            #model.summary()
            #model.fit([X_train,X_train_num],Y_train , batch_size=batch_size, epochs=nb_epoch, verbose=1)
            model.fit(X_train,Y_train , batch_size=batch_size, epochs=nb_epoch, verbose=1)
            pred_y_val = model.predict([X_val,X_val_num], batch_size=1024, verbose=2)
            f1_score_vector = []
            for thresh in np.arange(0.1, 0.981, 0.01):
                pred_y_val2 = (pred_y_val>thresh).astype(int)
                f1_score_temp = metrics.f1_score(Y_val,pred_y_val2)
                f1_score_vector.append(f1_score_temp)
            f1_score_matrix.append(f1_score_vector)
            del model
        np.save(save_name,np.array(f1_score_matrix))
    else:
        if best_threshold>0 and best_threshold <1:
            all_preds_sub = []
            lr1 = scale*2e-3
            lr2 = scale*1e-3
            print("Fitting RNN model ...")
            for bag in range(7):
                np.random.seed(bag)
                model = BidLstm(maxlen, max_features, embed_size,embedding_matrix)
                optimizer = Adam(lr=lr1)
                model.compile(loss="binary_crossentropy", optimizer=optimizer)
                model.fit(train_X,train_y , batch_size=batch_size, epochs=nb_epoch, verbose=1)
                if finetune: #finetune helps a lot
                    model.get_layer('emb').trainable=True
                    optimizer = Adam(lr=lr2)
                    model.compile(loss="binary_crossentropy", optimizer=optimizer)
                    model.fit(train_X,train_y , batch_size=batch_size, epochs=1, verbose=1)

                preds_sub = model.predict( test_X , batch_size=1024).squeeze()
                all_preds_sub.append(preds_sub)
                del model
                print('*******************************************************')
            subThreshold = best_threshold
            mean_preds = np.array(all_preds_sub).transpose()    
            mean_preds = mean_preds.mean(axis=1)

            sub_df = pd.DataFrame()
            sub_df['qid'] = test_df.qid.values
            sub_df['prediction'] = (mean_preds>subThreshold).astype(int)
            sub_df.to_csv('submission.csv', index=False)
        else:
            print('threshold_wrong')

scale = 1.
BATCH_SIZE = int(scale*1024)
batch_size = BATCH_SIZE
epochs = 5
runtraining(model_type = 31,save_name = 'first part',batch_size = batch_size,nb_epoch = epochs,is_preprocessing = True,best_threshold = 0.33,is_GPU = True)
