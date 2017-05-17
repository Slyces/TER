import pandas
import sqlite3 as sql
from django_world.settings import DATABASE_PATH

# df = pandas.read_csv('sp500.csv')
con = sql.connect(DATABASE_PATH)

# df.to_sql('historic_snp', con)

df = pandas.read_csv('datas.csv')
print(df)

req = """INSERT INTO extremum_indexes (ind, min, max) VALUES (?, ?, ?);"""
for index in 'Nasdaq Dow S&P500 Rates'.split():
    con.execute(req, (index, min(df[index]), max(df[index])))
con.commit()