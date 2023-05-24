import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.subplots as sp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats # 상관관계 분석 import
from datetime import date, timedelta

def Read_data(filename):
    df = pd.read_csv("C:/Users/jw/Desktop/TeamProject_Flask_Prac/{}.csv".format(filename))
#     df.set_index('Date', inplace=True)
#     df.sort_index(ascending=True, inplace=True)
    df['Name'] = filename # 식별자 추가(나중에 분석 돌리기 위해서 추가함..)
    return df


def Make_Data_Polynomial(filename):
    df = Read_data(filename)
    X = df['Date'].values # 처음부터 마지막 컬럼 직전까지의 데이터 (독립 변수 - 원인)
    y = df['Close'].values # 마지막 컬럼 데이터 (종속 변수 - 결과)
    df = pd.DataFrame(X, columns=['Date'])
    # 'Date' 열의 데이터를 날짜로 변환
    df['Date'] = pd.to_datetime(df['Date'])
    # 'Date' 열의 데이터를 숫자로 변환
    df['Date_numeric'] = pd.factorize(df['Date'])[0]
    X = df['Date'].values 
    X = df['Date_numeric'].values
    X = X.reshape(-1, 1)
    return X, y



def Polynomial_graph(filename, degree):
    X = Make_Data_Polynomial(filename)[0]
    y = Make_Data_Polynomial(filename)[1]
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    x_range = np.linspace(X.min(), X.max(), 100)
    X_range_poly = poly_reg.transform(x_range.reshape(-1, 1))
    y_range = lin_reg.predict(X_range_poly)

    # 등락률 계산
    y_diff = np.diff(y)  # 이전 값과의 차이 계산
    y_rate = (y_diff / y[:-1]) * 100  # 등락률 계산 (백분율)

    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2])
    
    colors = ['blue' if rate < 0 else 'red' for rate in y_rate]

    fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', marker=dict(color='green'), opacity=0.65, name='Value'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_range, y=y_range, name='Polynomial Fit'), row=1, col=1)
    fig.add_trace(go.Bar(x=X[:-1].flatten(), y=y_rate, name='등락률', marker=dict(color=colors, opacity=0.5)), row=2, col=1)

    fig.update_layout(
        title='Polynomial Model {}'.format(filename),
        # xaxis_title='date',
        yaxis_title='value',
        autosize=True
    )

    return fig


def Polynomial_Predict_days(filename, degree, end):# degree 는 숫자 , end일 후 까지 예측
    X = Make_Data_Polynomial(filename)[0]
    y = Make_Data_Polynomial(filename)[1]
    result = []
    poly_reg = PolynomialFeatures(degree=degree) # 그래프의 X 계수를 결정
    X_poly = poly_reg.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y) # 변환된 X 와 y 를 가지고 모델 생성 (학습)
    for i in range(int(Make_Data_Polynomial(filename)[0][-1]) + 1, int(Make_Data_Polynomial(filename)[0][-1]) + 1 + end):
        predict_value = lin_reg.predict(poly_reg.fit_transform([[i]]))
        result.append(predict_value)
    return result


def Change_DataFrame(filename, degree, end):
    df = pd.DataFrame(Polynomial_Predict_days(filename, degree, end), columns=['Close'])
    df['Name'] = filename
    return df


def anal_scipy(*dfs):
    result = []
    for i in range(len(dfs)):
        for j in range(i+1, len(dfs)):
            x = dfs[i]
            y = dfs[j]
            result_key = str(x['Name'].iloc[0]) + ", " + str(y['Name'].iloc[0]) # UK,KS200 이런 형태로 만들어줌
            result_value = (result_key, scipy.stats.pearsonr(x['Close'], y['Close']))
            result.append(result_value)

    return result


def Change_DataFrame(filename, degree, end):
    df = pd.DataFrame(Polynomial_Predict_days(filename, degree, end), columns=['Close'])
    df['Name'] = filename
    return df

def spicy_graph(*dfs):
    My_data = anal_scipy(*dfs)
    result_index = []
    result_value = []
    for x in range(len(My_data)):
        result_index.append(My_data[x][0])
        result_value.append(My_data[x][1][0])
        
    table_data = [['', 'Value']]
    teams = []
    Value = []

    for i in range(len(result_index)):
        table_data.append([result_index[i], result_value[i]])
        teams.append(result_index[i])
        Value.append(result_value[i])

    fig = ff.create_table(table_data, height_constant=60)

    # Make traces for graph
    max_abs_value_index = max(range(len(Value)), key=lambda i: abs(Value[i]))
    colors = ['#0099ff' if idx != max_abs_value_index else '#ff0000' for idx in range(len(Value))]

    trace1 = go.Bar(x=teams, y=Value, xaxis='x2', yaxis='y2',
                    marker=dict(color=colors),
                    name='Value')

    # Add trace data to figure
    fig.add_traces([trace1])

    # initialize xaxis2 and yaxis2
    fig['layout']['xaxis2'] = {}
    fig['layout']['yaxis2'] = {}

    # Edit layout for subplots
    fig.layout.yaxis.update({'domain': [0, .45]})
    fig.layout.yaxis2.update({'domain': [.6, 1]})

    # The graph's yaxis2 MUST BE anchored to the graph's xaxis2 and vice versa
    fig.layout.yaxis2.update({'anchor': 'x2'})
    fig.layout.xaxis2.update({'anchor': 'y2'})
    fig.layout.yaxis2.update({'title': 'Value'})

    # Update the margins to add a title and see graph x-labels.
    fig.layout.margin.update({'t':75, 'l':50})
    fig.layout.update({'title': 'scipy Analysis'})

    # Update the height because adding a graph vertically will interact with
    # the plot height calculated for the table
    fig.layout.update({'height':800})

    # Plot!
    return fig


def spicy_graph_avg(index_values, value_values):
    table_data = [['', 'Value']]
    teams = []
    Value = []

    for i in range(len(index_values)):
        table_data.append([index_values[i], value_values[i]])
        teams.append(index_values[i])
        Value.append(value_values[i])

    fig = ff.create_table(table_data, height_constant=60)

    # Make traces for graph
    max_abs_value_index = max(range(len(Value)), key=lambda i: abs(Value[i]))
    colors = ['#0099ff' if idx != max_abs_value_index else '#ff0000' for idx in range(len(Value))]

    trace1 = go.Bar(x=teams, y=Value, xaxis='x2', yaxis='y2',
                    marker=dict(color=colors),
                    name='Value')

    # Add trace data to figure
    fig.add_traces([trace1])

    # initialize xaxis2 and yaxis2
    fig['layout']['xaxis2'] = {}
    fig['layout']['yaxis2'] = {}

    # Edit layout for subplots
    fig.layout.yaxis.update({'domain': [0, .45]})
    fig.layout.yaxis2.update({'domain': [.6, 1]})

    # The graph's yaxis2 MUST BE anchored to the graph's xaxis2 and vice versa
    fig.layout.yaxis2.update({'anchor': 'x2'})
    fig.layout.xaxis2.update({'anchor': 'y2'})
    fig.layout.yaxis2.update({'title': 'Value'})

    # Update the margins to add a title and see graph x-labels.
    fig.layout.margin.update({'t':75, 'l':50})
    fig.layout.update({'title': 'scipy Analysis'})

    # Update the height because adding a graph vertically will interact with
    # the plot height calculated for the table
    fig.layout.update({'height':800})

    # Plot!
    return fig


