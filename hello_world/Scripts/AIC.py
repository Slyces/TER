import sqlite3 as sql
from django_world.settings import DATABASE_PATH
import numpy as np
from math import log as ln

def AIC():
    conn = sql.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""SELECT Nasdaq, Dowjones, SnP500, Rates FROM historic_indexes;""")
    datas = np.array(cursor.fetchall())[11:]

    cursor.execute("""SELECT Nasdaq, Dowjones, SnP500, Rates FROM predicted_indexes
                      WHERE Label=1;""")
    predicted = np.array(cursor.fetchall())

    lol = datas - predicted
    rss = 0
    for x in lol:
        for y in x:
            rss += y**2
    k = 40 * 4
    n = len(lol)

    print(n * ln(rss/n) + 2*k)


if __name__ == '__main__':
    AIC()