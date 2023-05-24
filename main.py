# 리스트 화 된 자료가 가져오기 편함

import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, redirect
from flask import url_for
import plotly.graph_objects as go
from My_Def.Total_def import *



UK_df = Read_data("UK")
KS200_df = Read_data("KS200")
OIL_df = Read_data("OIL")
CD91_df = Read_data("CD91")
SamSung_df = Read_data("SamSung")


UK_data_df = Change_DataFrame("UK", 5, 5)
KS200_data_df = Change_DataFrame("KS200", 4, 5)
OIL_data_df = Change_DataFrame("OIL", 7, 5)
CD91_data_df = Change_DataFrame("CD91", 5, 5)
SamSung_data_df = Change_DataFrame("SamSung", 4, 5)


Statistic_avg = []
Statistic_index = []
Data1 = anal_scipy(UK_data_df, KS200_data_df, OIL_data_df, CD91_data_df)
Data2 = anal_scipy(UK_df, KS200_df, OIL_df, CD91_df)

for x in range(len(anal_scipy(UK_df, KS200_df, OIL_df, CD91_df))):
    Statistic_avg.append((Data1[x][1][0] * (95/100)) + (Data2[x][1][0] * (5/100)))
    Statistic_index.append(Data1[x][0])


Standard_index = UK_df, KS200_df, CD91_df
Standard_predict_index = UK_data_df, KS200_data_df, CD91_data_df


Standard_result = []
for x in Standard_index:
    Standard_result.append(anal_scipy(SamSung_df, x))


Standard_result = [item for sublist in Standard_result for item in sublist]



Standard_predict_result = []
for x in Standard_predict_index:
    Standard_predict_result.append(anal_scipy(SamSung_data_df, x))


Standard_predict_result = [item for sublist in Standard_predict_result for item in sublist]



SamSung_Graph_index = []
SamSung_Graph_Value1 = []
SamSung_Graph_Value2 = []
for x in range(len(Standard_result)):
    SamSung_Graph_index.append(Standard_result[x][0])
    SamSung_Graph_Value1.append(Standard_result[x][1][0])
    SamSung_Graph_Value2.append(Standard_predict_result[x][1][0])


SamSung_Graph_avg = []
for x in range(len(SamSung_Graph_Value1)):
    SamSung_Graph_avg.append(
       (SamSung_Graph_Value1[x] * (95 / 100)) +  SamSung_Graph_Value2[x] * (5 / 100))



app = Flask("Project")



@app.route("/")
def SPS_main():
  graph_html_1 = Polynomial_graph("SamSung",4).to_html(full_html=False)
  graph_html_2 = Polynomial_graph("UK",5).to_html(full_html=False)
  graph_html_3 = Polynomial_graph("KS200",4).to_html(full_html=False)
  graph_html_4 = Polynomial_graph("OIL",7).to_html(full_html=False)
  graph_html_5 = Polynomial_graph("CD91",5).to_html(full_html=False)
  url1 = url_for('SPS_main02')
  return render_template("SPS_main01.html", graph_html_1=graph_html_1, 
                         graph_html_2=graph_html_2, graph_html_3=graph_html_3,
                         graph_html_4=graph_html_4, graph_html_5=graph_html_5,
                         url1=url1)


@app.route("/SPS_main02")
def SPS_main02():
  graph_html_1 = spicy_graph(UK_df, KS200_df, OIL_df, CD91_df).to_html(full_html=False)
  graph_html_2 = spicy_graph(UK_data_df, KS200_data_df, OIL_data_df, CD91_data_df
                             ).to_html(full_html=False)
  graph_html_3 = spicy_graph_avg(Statistic_index, Statistic_avg).to_html(full_html=False)
  url1 = url_for("SPS_main03")

  return render_template("SPS_main02.html", graph_html_1=graph_html_1, graph_html_2=graph_html_2,
                         graph_html_3=graph_html_3, url1=url1)

@app.route("/SPS_main03")
def SPS_main03():
   graph_html_1 = spicy_graph_avg(SamSung_Graph_index, SamSung_Graph_avg).to_html(
      full_html=False)
   
   return render_template("SPS_main03.html", graph_html_1=graph_html_1)

app.run("0.0.0.0")
