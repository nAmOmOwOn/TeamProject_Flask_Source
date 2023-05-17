import pandas as pd
import plotly.express as px


def displaygraph():
    df = px.data.gapminder().query("country=='Canada'")
    fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
    
    return fig