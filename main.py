# 리스트 화 된 자료가 가져오기 편함

import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, redirect
from flask import url_for
from random import *
import plotly.graph_objects as go
from My_Def.lotto import My_lotto
from My_Def.showgraph import displaygraph

app = Flask("Project")


# @app.route("/")
# def home():
#   return render_template("home.html")

@app.route("/")
def SPS_main():
  graph_html_1 = displaygraph().to_html(full_html=False)
  graph_html_2 = displaygraph().to_html(full_html=False)
  graph_html_3 = displaygraph().to_html(full_html=False)
  graph_html_4 = displaygraph().to_html(full_html=False)
  return render_template("SPS_main.html", graph_html_1=graph_html_1, 
                         graph_html_2=graph_html_2, graph_html_3=graph_html_3,
                         graph_html_4=graph_html_4)


@app.route("/data1")
def data1():
  total = [x for x in range(1, 46)]
  result = []
  shuffle(total)
  for x in range(0, 6):
    x = total.pop()
    result.append(x)

  return render_template("data1.html", numbers=result)


@app.route("/data2")
def data2():
  total = [x for x in range(1, 46)]
  result = []
  shuffle(total)
  for x in range(0, 6):
    x = total.pop()
    result.append(x)

  return render_template("data2.html", numbers=result)

@app.route("/image")
def image():
  return render_template("image.html")


@app.route("/Prac")
def Prac():
  url1 = url_for('home')
  url2 = url_for('data1')
  url3 = url_for('data2')
  return render_template("Prac.html", url1=url1, url2=url2, url3=url3)

@app.route("/graph")
def graph():
  graph_html = displaygraph().to_html(full_html=False)
  return render_template("graph.html", graph_html=graph_html)


app.run("0.0.0.0")
