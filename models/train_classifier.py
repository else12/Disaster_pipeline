import sys
import re
import nltk
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, fbeta_score, precision_recall_fscore_support
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    load the data from the database in database-filepath
    
    Outputs: 
        X: features , dataframe
        Y: target dataframe
        category_nmes: target names
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df =  pd.read_sql_table(database_filepath, engine)
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names=list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    '''
    tokenize and clean the text 
    Input: 
        text: text to be tokenized and cleaned
    output: 
        clean_tokens: cleaned  and tokenized text
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)  
    tokens  = [w for w in tokens if   w not in stopwords.words("english")]
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()  
    #iterate through each token
    clean_tokens = []
    for tok in tokens:      
        #lemmatize, normalize case, and remove leading/trailing white space
         clean_tok =  lemmatizer.lemmatize(tok).lower().strip( )
         clean_tokens.append(clean_tok)       
    return clean_tokens

def build_model():
    ''' building the model 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier(random_state=100)))
         ])
    # hyperparameters
    parameters = {'clf__estimator__n_estimators': [50, 100] ,
                  'clf__estimator__min_samples_split': [2, 4]
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters, return_train_score=True, verbose=2) # n_jobs=1, cv=1)
    return cv

def calc_scores(y_test, y_pred, category_names):
    '''
    Inputs:
        y_test: testing labels
        y_pred: predicted labels
        category_names: names of the labels
    '''
    res =[]
    precision =0
    recall=0
    f1score =0
    for i in range(len(category_names)):
         res = (precision_recall_fscore_support(y_test.iloc[:,i].values, y_pred[:,i], average='weighted'))
         precision += res[0]
         recall += res[1]
         f1score += res[2]
     
    precision = precision/len(category_names)
    recall = recall/len(category_names)
    f1score = f1score/len(category_names)
    
    tot_accuracy = (y_pred == y_test).mean().mean()
    
    print('Average  Weighted Prediction Scores:')
    print ("Precision: {:2f} Recall: {:2f}  F1-Score: {:2f}".format(precision*100, recall*100, f1score*100))
    print('total Accuracy: %2.2f'% (tot_accuracy*100))
    
def evaluate_model(model, X_test, y_test, category_names):
   '''  
    inputs: 
        model: trained model
        X_test: testing data
        y_test: data labels
        category_names: names of the labels
    '''
    y_pred = model.predict(X_test)
    calc_scores(y_test, y_pred, category_names)
    

def save_model(model, model_filepath):
    '''
    save the trained moel in python pickle file
    model: trained model
    model_filepath: where the model is located
    '''
    pickle.dump(model, open(model_filepath, "wb" ) )


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()