# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """ Loads data from sql data base

    loads cleaned data to X and Y, as well as taking the Y labels.

    Args:
        path to database (str)

    Returns:
        tuple (DataFrames and list): containing message data and labels.
    """
    # database_filepath "../data/DisasterResponse.db"
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # read in data to data frame
    df = pd.read_sql_table('coded_responses', engine)
    # separate into X and Y variables
    X = df["message"]
    Y = df.drop(columns=["message","id","genre","original"])
    # store the column headings of Y in categories.
    categories = list(Y.columns)
    return X, Y, categories

def tokenize(text):
    """ Creates tokens of a message

    Used to tokenize input message from webapp.

    Args:
        text (string): message to be tokenized
    Returns:
        list (string): list of tokenens
    """
    # create word tokens
    tokens = word_tokenize(text)
    # initialize lemmetizer
    lemmatizer = WordNetLemmatizer()
    # initialize tokens list
    clean_tokens = []
    for tok in tokens:
        # lemmetize, normalize and strip words for space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # add to tokens list
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ builds the model to be fitted on the data.

    Model used to train on the labeled disaster messages. Uses GridSearchCV
    to optimize the model in terms of parameters. In order to include more
    parameters, uncomment the lines in "parameters" below.

    Args:
        None

    Returns:
        model (GridSearchCV): optimized model to be fitted on the dataset

    """
    # build pipeline using tokenize function as defined above
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=1)))
    ])
    # define parameters to be used for optimizing model using GridSearchCV
    # Uncomment lines below to include more parameters
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 7500),
        'clf__estimator__n_estimators': [200, 1100, 2000],
        #'clf__estimator__max_features': ['auto', 'sqrt']
        #'clf__estimator__max_depth': max_depth,
        #'clf__estimator__min_samples_split': min_samples_split,
        #'clf__estimator__min_samples_leaf':min_samples_leaf,
        #'clf__estimator__bootstrap': bootstrap
    }
    # Use GridSearchCV to find optimized model when fitted to data.
    model = GridSearchCV(model, param_grid=parameters,cv = 5,n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate fitted model printing out scores per category

    This function will print out F1 score, Accuracy, Recall and Precision
    for all categories.

    Args:
        model (GridSearchCV): model to be fitted to dataset
        X_test (DataFrame): Test data messages
        Y_test (DataFrame): Test data labels
        category_names (list(string)): labels of categories

    Returns:
        None
    """
    # use model to predict on test messages
    y_pred = model.predict(X_test)
    # create data frame of predicte values
    y_pred=pd.DataFrame(data=y_pred,columns=category_names)
    # loop through each category labels and print score
    for column in Y_test:
        # print category name
        print(column)
        # print F1 score, Accuracy, Recall and Precision
        print(classification_report(
            Y_test[column].values, y_pred[column].values))


def save_model(model, model_filepath):
    """ saves the trained model as a pickle file.

    Args:
        model (GridSearchCV): trained model
        model_filepath (string) : filepath for the pickle file

    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ main function that is called when running script from command line

    Args:
        filename (string): name of this file (train_classifier.py)
        database_filepath (string): filepath to database
        model_filepath (string): filepath to store model as pickle file

    Returns:
        None
    """
    # Check for command line arguments
    if len(sys.argv) == 3:
        # get arguments from command line
        database_filepath, model_filepath = sys.argv[1:]
        # load data and print to user
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # split data into train and test set
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # build model and print to user
        print('Building model...')
        model = build_model()

        # train model and print to user
        print('Training model...')
        model.fit(X_train, Y_train)

        # evaluate model and print to user
        print('Evaluating model...')
        print('May take a long time...')
        evaluate_model(model, X_test, Y_test, category_names)

        # Save model into pickle file and print to user
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        # print to user when done
        print('Trained model saved!')
    # Prompt user to provide command line arguments
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
