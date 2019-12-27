# import modules
import pandas as pd
import plotly.graph_objs as go

def return_figures(df):
    """Creates two plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the two plotly visualizations

    """

# first chart plots messages by genres as bar chart
    graph_one = []
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    graph_one.append(
        go.Bar(
            x = genre_names,
            y = genre_counts,
    ))
    layout_one = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = "Genre"),
                yaxis = dict(title = 'Count'),
                )

# second chart plots labels and number of messages having these labels
    graph_two = []
    y_labels = list(df.drop(columns=["message","id","genre","original"]).columns)
    df2 = df[y_labels].sum().sort_values()
    graph_two.append(
        go.Bar(
            x = df2.keys().tolist(),
            y = df2.tolist(),
        )
    )

    layout_two = dict(title = 'Distribution of Message per Labels',
                xaxis = dict(title = 'Label',),
                yaxis = dict(title = 'Count'),
                )

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    # TODO: append the figure five information to the figures list

    return figures
