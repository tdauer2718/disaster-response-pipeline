# Imports
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
from time import time, perf_counter
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, make_scorer
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

# Download the 'punkt' package for sentence tokenization
nltk.download(['punkt', 'wordnet', 'stopwords'])

# Import a tokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM LabeledMessages", engine)
    # Create the series of messages
    X = df['message']
    # Create the dataframe of targets (each column is a target/category)
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    # Figure out which columns have more or fewer than 2 labels
    cols = Y.columns[np.where(Y.nunique() != 2)[0]]
    # Make a list to store the number of unique values in 
    # columns with != 2 unique values
    n_vals_list = []
    for i, c in enumerate(cols):
        n_vals = Y.nunique()[c]
        n_vals_list.append(n_vals)
    for i, c in enumerate(cols):
        n_vals = n_vals_list[i]
        if n_vals > 2:
            to_drop = Y[(Y[c] != 0) & (Y[c] != 1)].index
            # Drop all rows that have label not in {0, 1}
            X.drop(to_drop, inplace=True)
            Y.drop(to_drop, inplace=True)
        if n_vals < 2:
            # Drop columns from Y that don't have more than one label
            Y.drop(columns=c, inplace=True)

    # This is a bit annoying (not as efficient as I'd like), 
    # but I need to get rid of rows my Count Vectorizer can't work with
    # (these rows have messages that are stripped all the way down to
    # whitespace by my vectorizer, meaning they are just made up of
    # special characters/whitespace, and so they were likely left in by
    # mistake and wouldn't be useful for the disaster response team anyway)
    # Create a count vectorizer using my tokenizer function
    my_stopwords = list(set([tokenize(w)[0] for w in stopwords.words('english')]))
    vect = CountVectorizer(tokenizer=tokenize, stop_words=my_stopwords)
    # Make a list of row indices for which the vectorizer fails
    failed_is = []
    for i in X.index:
        try:
            vect.fit([X.loc[i]])
        except:
            failed_is.append(i)
    # Drop the appropriate rows
    X.drop(failed_is, inplace=True)
    Y.drop(failed_is, inplace=True)

    return X, Y


def tokenize(text):
    """Takes some text as input, then tokenizes it, lemmatizes it, strips it
        of characters that are not letters or numbers, and normalizes its case 
        (to lowercase only).
        
    Args:
    text -- a string of text.
    
    Returns:
    tokens_cleaned -- the final list of cleaned tokens.
    """
    text = text.lower()
    # Remove characters that aren't letters or numbers
    text = re.sub('[^a-z0-9]', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Instantiate a lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Lemmatize and strip the tokens
    tokens_cleaned = [lemmatizer.lemmatize(t).strip() for t in tokens]
    return tokens_cleaned


def build_model():
    # Create stopwords that are consistent with my tokenizer
    my_stopwords = list(set([tokenize(w)[0] for w in stopwords.words('english')]))
    # Create a Pipeline that uses a CountVectorizer with my above tokenizer
    # function, then uses a TF.IDF transformer, then uses an independent
    # random forest classifier for each output (utilizing MultiOutputClassifier())
    pipe = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, stop_words=my_stopwords)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42))),
        ])
    best_params_found = {'clf__estimator__class_weight': 'balanced',
                        'clf__estimator__min_samples_split': 3,
                        'clf__estimator__n_estimators': 100,
                        'tfidf__norm': 'l2',
                        'tfidf__use_idf': True,
                        'vect__max_features': None}
    pipe.set_params(**best_params_found)
    return pipe


def evaluate_model(model, X_test, Y_test):
    # Make predictions on the test set using the trained model
    Y_pred = model.predict(X_test)
    # Show the recall and precision for test set predictions for each target
    # Also show the class imbalance, i.e. the ratio of instances in class 1 to class 0
    print('Here are the results fr each category (for Class 1):')
    for i, col in enumerate(Y_test.columns):
        print(f'\033[1m{col}:\033[0m')
        report = classification_report(Y_test[col], Y_pred[:, i], 
                                    zero_division=0, output_dict=True)
        recall = report['1']['recall']
        precision = report['1']['precision']
        frac = report['1']['support']/report['0']['support']
        print(f'recall = {recall:.3g}; precision = {precision:.3g}; class ratio = {frac:.3g}')
        print('-'*50)


def save_model(model, model_filepath):
    # Pickle the model and save to it to model_filepath
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data from database {database_filepath} and preparing it...')
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Training model on full dataset (post testing):')
        model.fit(X, Y)

        print(f'Saving model to {model_filepath}')
        save_model(model, model_filepath)
        print('Full trained model saved')

    else:
        print_str = 'Provide the filepath of the prepared disaster messages database '
        print_str += 'as the first argument and the filepath where you want '
        print_str += 'to save the pickle file for the trained model as the '
        print_str += 'second argument.'
        print_str += '\n\nExample: python train_model.py ../data/data.db final_model.pkl'
        print(print_str)

if __name__ == '__main__':
    main()