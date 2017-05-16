import sqlite3
import numpy as np
from datetime import datetime
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter
from math import pi
from django_world.settings import DATABASE_PATH

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


    for i, index in enumerate(indexes):
        historic[:, i] = f(historic[:, i], extr[index])
        predicted[:, i] = f(predicted[:, i], extr[index])

    error = np.square(historic - predicted)

    width = 800
    height = 350

    axes = [None for i in range(4)]
    for i in range(4):
        TOOLS = 'pan,wheel_zoom,box_zoom,resize,reset'

        if i == 0:
            axes[i] = figure(responsive=True, height=height, width=width, x_axis_type="datetime", tools=TOOLS)
        else:
            axes[i] = figure(responsive=True, height=height, width=width, x_axis_type="datetime", tools=TOOLS,
                             x_range=axes[0].x_range, y_range=axes[0].y_range)

        options = {
            'line_join': 'round'
        }

        axes[i].line(dates, predicted[:, i], legend="Predictions", line_color='navy', **options)
        axes[i].line(dates, historic[:, i], legend="Historic Data", color='red', **options)

        axes[i].xaxis.formatter = DatetimeTickFormatter(
            days=["%d %B %Y"],
            months=["%d %B %Y"],
            years=["%d %B %Y"],
        )
        axes[i].xaxis.major_label_orientation = pi / 4
        axes[i].xaxis.bounds = (dates[0], dates[-1])
        print((extr[indexes[i]]['min'],
                                extr[indexes[i]]['max']))
        axes[i].yaxis.bounds = (extr[indexes[i]]['min'],
                                extr[indexes[i]]['max'])

        axes[i].legend.border_line_width = 1
        axes[i].legend.border_line_color = "black"
        axes[i].legend.border_line_alpha = 0.5

    return dict([(indexes[i], axes[i]) for i in range(4)])

if __name__ == '__main__':
    print(plot_feedforward("1"))