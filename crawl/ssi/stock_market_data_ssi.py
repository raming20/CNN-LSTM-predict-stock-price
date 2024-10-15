import sys
sys.path.append("../../")

from utils.read_har import *
from datetime import datetime
import pandas as pd


def get_stock_market_data(stock_symbol, resolution, from_time=datetime(2000, 1, 1), to_time=datetime.now()):
    """
    Args:
        stock_symbol (String): Stock symbol, Example: "FPT"
        reesolution (String): Example:
            # ONE_MINUTE = "1"
            # FIVE_MINUTES = "5"
            # FIFTEEN_MINUTES = "15"
            # THIRTY_MINUTES = "30"
            # ONE_HOUR = "1H"
            # ONE_DAY = "1D"
            # ONE_WEEK = "1W"
            # ONE_MONTH = "1M"
        from_time (Datetime): Example: datetime(2000, 1, 1)
        to_time (Datetime): Example: datetime(2000, 1, 1)

    Returns:
        pd.Dataframe: this function get data from SSI API and then convert it to pandas Dataframe with columns: 
            t: time,
            c: close price,
            o: open price,
            h: high price,
            l: low price,
            v: volume
    """

    headers = {'accept': 'application/json, text/plain, */*',
               'accept-encoding': 'gzip, deflate, br, zstd',
               'accept-language': 'en-US,en;q=0.9',
               'origin': 'https://iboard.ssi.com.vn',
               'priority': 'u=1, i',
               'referer': 'https://iboard.ssi.com.vn/',
               'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Microsoft Edge";v="126"',
               'sec-ch-ua-mobile': '?0',
               'sec-ch-ua-platform': '"Windows"',
               'sec-fetch-dest': 'empty',
               'sec-fetch-mode': 'cors',
               'sec-fetch-site': 'same-site',
               'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'}

    cockies = {}

    query_string = {'resolution': resolution,
                    'symbol': stock_symbol,
                    'from': str(from_time.timestamp()),
                    'to': str(to_time.timestamp())}

    url_without_params = 'https://iboard-api.ssi.com.vn/statistics/charts/history'

    response = requests.get(
        url=url_without_params,
        headers=headers,
        params=query_string
    )

    response.status_code

    return pd.DataFrame(response.json()["data"])
