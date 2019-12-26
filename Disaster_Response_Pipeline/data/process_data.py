import sys
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories, on='id', how='left')
    return df

def clean_data(df):
    # clean categories
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
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
    # drop the original categories column from `df`
    df = df.drop(columns="categories")
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='outer')
    # remove duplicates
    df = df.drop_duplicates()
    # remove missing and unused columns
    df = df.drop(columns=["id","original","genre"])
    # drop rows with any missing values
    df = df.dropna()
    # concatenation casted the categories to floats.
    # want them as itegers
    # Convert category columns to int again (after converted to float after concatenation)
    to_int_columns = list(set(df.columns).difference(set(['message'])))
    df[to_int_columns] = df[to_int_columns].astype("int32")
    return df

def save_data(df, database_filename):
    # save cleaned data set to sql data base
    engine = create_engine('sqlite:///data/{}'.format(database_filepath))
    df.to_sql("coded_responses", engine,if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
