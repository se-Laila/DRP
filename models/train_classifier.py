import sys
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    INPUT 
    database_filepath - the path where the database is saved
    
    OUTPUT
    X - the messages
    Y - the rest of db
    Y.columns -  message categories
    
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response',engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    
    return X, Y, Y.columns


def tokenize(text):
    '''
    INPUT 
    text - text to be tokenized, lemmatized, normalized, striped, and doesn't have stop words
    
    OUTPUT
    clean_tokens
    '''
    # initialization 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if tok not in stopWords:
            clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    INPUT 
    
    OUTPUT
    CV - The model resulted from the ML pipeline and grid search
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
 
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200]
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'tfidf__use_idf': (True, False)
        #'clf__min_samples_split': [2, 3]
        #'vect__max_df': (0.75, 1.0),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)  
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - the model to be evaluated
    X_test - test feature set
    Y_test - test target set
    category_names - target category names
    '''
    # Use model to predict
    Y_pred = model.predict(X_test)
    
    # Turn prediction into DataFrame
    Y_pred = pd.DataFrame(Y_pred,columns=category_names)
    # For each category column, print performance
    for col in category_names:
        print(f'Column Name:{col}\n')
        print(classification_report(Y_test[col],Y_pred[col]))
        
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == Y_pred)))


def save_model(model, model_filepath):
    '''
    INPUT 
    model - model to be saved
    model_filepath - File path where the model will be saved
    
    OUTPUT
    a pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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