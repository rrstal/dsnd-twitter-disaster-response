import sys

import re
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from sqlalchemy import create_engine

import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath):
    """Loads the input data
    
    Args:
        database_filepath (str): path to the database file
    Returns:
        X (pandas.Dataframe): features dataframe
        Y (pandas.Dataframe): targets dataframe
        label_categories (pandas.Series): list of label names, for visualization in the app
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('twitter-data', con=engine)
    
    X = df['message']
    y = df[df.columns[4:]]
    
    # map the extraordinary value 2 to 1, so that values can take one of two values: 0 or 1.
    y['related']=y['related'].map(lambda x: 1 if x == 2 else x)
    label_categories = y.columns
    
    return X, y, label_categories


def tokenize(text):
    """Tokenizes the input
    
    Args:
        text: contains a list of messages (in english)
    Returns:
        clean_tokens: text, tokenized
    """
    # 1 normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # 2 tokenize
    tokens = word_tokenize(text)

    # 3 remove stop words
    words = [w for w in tokens if w not in stopwords.words("english")]
    
    # 4 init Lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for word in words:
        clean_tok = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Initializes machine learning model
    
    Returns:
        model: model to perform grid search cross validation on
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=32)))
            ])
    
    params = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [3, 5]
    }

    cv_model = GridSearchCV(estimator = pipeline, param_grid = params, cv = 2, verbose = 3)
    return cv_model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model and generates report
    
    Args:
        model: ml model trained
        X_test: features for test
        y_test: labels for test
        category_names: names of the labels
    """
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_pred, Y_test.values, target_names = category_names))
    


def save_model(model, model_filepath):
    """Saves model to pickle
    
    Args:
        model: ml model trained
        model_filepath: output path
    """
    with open(model_filepath, 'wb') as output_file:
        pickle.dump(model, output_file)
    

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
        print('Please provide the filepath of the s messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()