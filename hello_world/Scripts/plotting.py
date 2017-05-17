import sqlite3
import numpy as np
from datetime import datetime
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter
from math import pi
from django_world.settings import DATABASE_PATH

width = 1050
height = 350


def plot_feedforward(label):
    indexes = 'Nasdaq DowJones SnP500 Rates'.split()
    with sqlite3.connect(DATABASE_PATH) as con:
        cursor = con.cursor()

        select_dates = "SELECT DateTime FROM historic_indexes WHERE Id > 11;"
        cursor.execute(select_dates)
        dates = np.array([datetime.strptime(x[0], '%Y-%m-%d') for x in cursor.fetchall()])

        select_historic = "SELECT Nasdaq, Dowjones, SnP500, Rates FROM historic_indexes WHERE Id > 11;"
        cursor.execute(select_historic)
        historic = np.array(cursor.fetchall())

        select_predictions = "SELECT Nasdaq, Dowjones, SnP500, Rates FROM predicted_indexes WHERE Label = {};".format(
            label
        )
        cursor.execute(select_predictions)
        predicted = np.array(cursor.fetchall())

        extr = dict([(index, {}) for index in indexes])
        for index in indexes:
            for m in 'min max'.split():
                cursor.execute("""SELECT {} FROM extremum_indexes
                                  WHERE ind = ?""".format(m), (index,))
                extr[index][m] = cursor.fetchone()[0]

    # De-normalize datas
    # Data normalization formula : y = (x - Min(X))/(Max(X) - Min(X))
    # To de-normalize, we use : (Max(X) - Min(X)) * y + Min(X)
    def f(x, extrems):
        return (extrems['max'] - extrems['min']) * x + extrems['min']

    historic_unnormalised = np.empty(historic.shape)
    predicted_unnormalised = np.empty(predicted.shape)
    for i, index in enumerate(indexes):
        historic_unnormalised[:, i] = f(historic[:, i], extr[index])
        predicted_unnormalised[:, i] = f(predicted[:, i], extr[index])

    error = np.sqrt(np.square(historic - predicted))

    resp = False

    axenorm = [None for i in range(4)]
    axerr = [None for i in range(4)]
    axes = [None for i in range(4)]
    for i in range(4):
        TOOLS = 'pan,wheel_zoom,box_zoom,resize,reset'

        axerr[i] = figure(responsive=resp, height=height, width=width, x_axis_type="datetime", tools=TOOLS)
        axenorm[i] = figure(responsive=resp, height=height, width=width, x_axis_type="datetime", tools=TOOLS)
        axes[i] = figure(responsive=resp, height=height, width=width, x_axis_type="datetime", tools=TOOLS)


        options = {
            'line_join': 'round'
        }

        axes[i].line(dates, historic_unnormalised[:, i], legend="Predictions", line_color='navy', **options)
        axes[i].line(dates, predicted_unnormalised[:, i], legend="Données historiques", color='red', **options)

        axerr[i].line(dates, error[:, i], legend="Racine de l'erreur", line_color='orange', **options)
        axenorm[i].line(dates, historic[:, i], legend="Predictions normalisées", line_color='navy', **options)
        axenorm[i].line(dates, predicted[:, i], legend="Données historiques normalisées", line_color='red', **options)

        axes[i].xaxis.formatter = DatetimeTickFormatter(
            days=["%d %B %Y"],
            months=["%d %B %Y"],
            years=["%d %B %Y"],
        )
        axerr[i].xaxis.formatter = DatetimeTickFormatter(
            days=["%d %B %Y"],
            months=["%d %B %Y"],
            years=["%d %B %Y"],
        )
        axenorm[i].xaxis.formatter = DatetimeTickFormatter(
            days=["%d %B %Y"],
            months=["%d %B %Y"],
            years=["%d %B %Y"],
        )
        axerr[i].xaxis.major_label_orientation = pi / 4

        axes[i].xaxis.major_label_orientation = pi / 4
        axes[i].xaxis.bounds = (dates[0], dates[-1])
        axes[i].yaxis.bounds = (extr[indexes[i]]['min'],
                                extr[indexes[i]]['max'])

        axenorm[i].xaxis.major_label_orientation = pi / 4
        axenorm[i].xaxis.bounds = (dates[0], dates[-1])
        axenorm[i].yaxis.bounds = (0, 1)

        axes[i].legend.border_line_width = 1
        axes[i].legend.border_line_color = "black"
        axes[i].legend.border_line_alpha = 0.5

    print(historic[:, 3])
    print(historic_unnormalised[:, 3])

    return dict([(indexes[i], axes[i]) for i in range(4)]), \
           dict([(indexes[i], axerr[i]) for i in range(4)]), \
           dict([(indexes[i], axenorm[i]) for i in range(4)])


def plot_lmst():
    TOOLS = 'pan,wheel_zoom,box_zoom,resize,reset'
    with sqlite3.connect(DATABASE_PATH) as con:
        cursor = con.cursor()
        cursor.execute("""SELECT historic FROM lstm_single;""")
        historic = np.array(cursor.fetchall())
        cursor = con.cursor()
        cursor.execute("""SELECT predicted FROM lstm_single;""")
        predicted = np.array(cursor.fetchall())

    resp = False

    axes = figure(responsive=resp, height=height, width=width, tools=TOOLS)
    x = [i for i in range(len(predicted))]
    axes.line(x, list(predicted), legend="Predictions", line_color='violet', line_join='round')
    axes.line(x, list(historic), legend="Données réelles", line_color='green', line_join='round')

    error = np.sqrt(np.square(historic - predicted))

    axerr = figure(responsive=resp, height=height, width=width, tools=TOOLS)
    axerr.line(x, list(error), legend="Racine de l'erreur", line_color='orange', line_join='round')

    return axes, axerr


if __name__ == '__main__':
    print(plot_feedforward("1"))
    # print(plot_lmst())
