import requests, csv
from datetime import datetime, timedelta


def download(index, filename, start, end):
    correspondances = {
        "S&P500": '^GSPC'
    }

    if index in correspondances:
        index = correspondances[index]

    url = "http://chart.finance.yahoo.com/table.csv?s={index}" \
          "&a={sm}&b={sd}&c={sy}&d={em}&e={ed}&f={ey}&g=d" \
          "&ignore=.csv".format(
        index=index, sd=start.day, sm=start.month, sy=start.year,
        ed=end.day, em=end.month, ey=end.year
    )
    print(url)
    print("http://chart.finance.yahoo.com/table.csv?s=^GSPC&a=1&b=18&c=2017&d=2&e=18&f=2017&g=d&ignore=.csv")

    with requests.Session() as s:
        dl = s.get(url)

        decoded_content = dl.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        with open(filename, 'w') as file:
            for line in list(cr):
                # print(line)
                file.write(','.join(line) + '\n')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="downloads datas of daily values of "
                                        "the given index to the given file "
                                        "from yahoo!finance")
    parser.add_argument('-s', '--start', dest="start_date", default=None,
                        help="starting date [yyyy-mm-dd]")
    parser.add_argument('-e', '--end', dest="end_date", default=None,
                        help="ending date [yyyy-mm-dd]")
    parser.add_argument("index", help="name")
    parser.add_argument("-f", "--file", dest="filename", default='data.csv',
                        help="name of the file where the data will be"
                             "downloaded")

    args = parser.parse_args()

    if args.end_date is None:
        end = datetime.now()
    else:
        end = datetime(*[int(x) for x in args.end_date.split('-')])
    if args.start_date is None:
        start = datetime(end.year - (1 if end.month == 1 else 0),
                         end.month - (1 if end.month > 1 else -11),
                         end.day)
    else:
        start = datetime(*[int(x) for x in args.start_date.split('-')])
    assert start < end

    # download(args.index, args.filename, start, end)
    import yahoo_finance
    index = yahoo_finance.Share(args.index)
    # print(index.get_historical('2014-04-25', '2014-04-29'))
    # for year in range(1990, 2016):
    #     print('{}-01-01'.format(year))
    #     try:
    #         index.get_historical('2014-04-25', '2014-04-29')
    #         print('{}-01-01'.format(year))
    #         break
    #     except:
    #         pass
    # print('finished')
    import pandas
    frame = pandas.DataFrame(columns=['Nasdaq', 'Dowjones', 'S&P500', 'Rates'])
    print(len(frame))