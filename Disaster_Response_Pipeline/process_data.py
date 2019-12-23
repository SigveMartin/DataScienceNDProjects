import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def process_data():

    # load messages dataset
    messages = pd.read_csv("messages.csv")
    # load categories dataset
    categories = pd.read_csv("categories.csv")
    # merge datasets
    df = pd.merge(messages,categories, on='id', how='left')
    # clean categories
    categories = clean_categories(categories)
    # drop the original categories column from `df`
    df = df.drop(columns="categories")
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='outer')
    # remove duplicates
    df = remove_duplicates(df)

    # save cleaned data set to sql data base
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)

def remove_duplicates(df):
    # remove "original message column"
    ## Assume original message is not needed (different language.) 60% missing, so drop whole column first.
    ## This will ensure we don't need to drop 60% of data.
    df = df.drop(columns="original")
    # Drop rows hvaing any missing values
    df = df.dropna()
    return df

def clean_categories(categories):
    """ This function taks in a raw data frame of categories,
        creates 36 individual categories columns and cleans the row values
        to have integer 0 or 1.

        input categories (pandas data frame)

        returns cleaned categories (pandas data frame)
    """
    # create a dataframe of the 36 individual category columns
    categories = categories["categories"].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split("-")[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Loop through all category columns and convert values to only 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        # first ensure that it is string values by setting type.
        # value of format "aid_related-0" str[-1] gives 0.
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric (int32)
        categories[column] = categories[column].astype("int32")
    return categories
