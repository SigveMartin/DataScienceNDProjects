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

    # database_filepath "../data/DisasterResponse.db"
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('coded_responses', engine)
    X = df["message"]
    Y = df.drop(columns=["message","id","genre","original"])
    categories = list(Y.columns)
    return X, Y, categories

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=1)))
    ])
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

    model = GridSearchCV(model, param_grid=parameters,n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred=pd.DataFrame(data=y_pred,columns=category_names)
    for column in Y_test:
        print(column)
        print(classification_report(
            Y_test[column].values, y_pred[column].values))


def save_model(model, model_filepath):
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
