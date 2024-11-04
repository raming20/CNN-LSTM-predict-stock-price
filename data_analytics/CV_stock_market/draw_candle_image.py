# %%
import sys

import matplotlib.axes
import matplotlib.pyplot
import ta.trend
sys.path.append("../../")


from datetime import datetime
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from ta.utils import dropna
import ta
import ta.momentum
import matplotlib


def convert_date_index(df: pd.DataFrame):
    """
    df is a pandas.DataFrame and assumed to have a columns: ['Date']
    """
    df = df.copy()
    
    if "Date" not in df.columns:
        return df
    
    assert "Date" in df.columns
    
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    
    return df
    


def draw_candle_image(
    df_draw: pd.DataFrame, 
    save_to_file=None, 
    show_macd=False, 
    df_calculate_macd=None, 
    show_x_y = False, 
    show_volume = False):
    """
    df is a pandas.DataFrame and assumed to have a columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] 
    or ['Open', 'High', 'Low', 'Close', 'Volume'] and index is Date data.
    
    If macd = True then df must have additional columns: ["Macd", "Macd_signal", "Macd_histogram"], if not then 
    function automatically calculates Macd indices.
    """
    df_draw = df_draw.copy()
    
    if df_calculate_macd is None:
        df_calculate_macd = df_draw
    
    df_calculate_macd = df_calculate_macd.copy()
    
    if 'Date' in df_draw.columns:
        df_draw = convert_date_index(df_draw)
    
    if 'Date' in df_calculate_macd.columns:
        df_calculate_macd = convert_date_index(df_calculate_macd)
    
    mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='inherit')
    style = mpf.make_mpf_style(marketcolors=mc)
    
    adps = []
    if show_macd:
        if "Macd" not in df_calculate_macd.columns or \
            "Macd_signal" not in df_calculate_macd.columns or \
            "Macd_histogram" not in df_calculate_macd.columns:
            
            df_calculate_macd["Macd"] = ta.trend.macd(df_calculate_macd["Close"], fillna=True)
            df_calculate_macd["Macd_signal"]= ta.trend.macd_signal(df_calculate_macd["Close"], fillna=True)
            df_calculate_macd["Macd_histogram"] = ta.trend.macd_diff(df_calculate_macd["Close"], fillna=True)
        
        df_calculate_macd_correspond = df_calculate_macd[df_draw.index.min():df_draw.index.max()]
        df_draw["Macd"] = df_calculate_macd_correspond["Macd"]
        df_draw["Macd_signal"] = df_calculate_macd_correspond["Macd_signal"]
        df_draw["Macd_histogram"] = df_calculate_macd_correspond["Macd_histogram"]
        
    
    fig: plt.Figure
    axlist: list[matplotlib.axes.Axes]
    
    panel_ratios = [6]
    if show_volume:
        panel_ratios.append(2)
        
    if show_macd:
        panel_ratios.append(2)
        panel_order_of_macd = len(panel_ratios) - 1 # display the macd in the bottom of image
        adps = [
            mpf.make_addplot(df_draw["Macd"], panel=panel_order_of_macd, color="blue", ylabel="MACD"),
            mpf.make_addplot(df_draw["Macd_signal"], panel=panel_order_of_macd, color="red"),
            mpf.make_addplot(df_draw["Macd_histogram"], type="bar", panel=panel_order_of_macd, color="gray", alpha=0.5),
        ]
        
    
    # Vẽ biểu đồ nến với khối lượng và ẩn các trục
    fig, axlist = mpf.plot(
        df_draw, 
        type='candle', 
        style=style, 
        volume=show_volume,  # Hiển thị khối lượng
        axisoff=not show_x_y,  # Bỏ trục x và y
        returnfig=True,  # Trả về đối tượng Figure để tùy chỉnh
        figratio=(5,5),  # Điều chỉnh tỷ lệ khung hình để thu hẹp khoảng cách
        figscale=1,  # Tăng kích thước biểu đồ để làm các nến gần nhau hơn
        addplot=adps,
        panel_ratios=panel_ratios
    )

    # Loại bỏ các nhãn trục và tiêu đề
    if not show_x_y:
        for ax in axlist:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")

    # Hiển thị biểu đồ
    # fig.show()
    
    if type(save_to_file) == str:
        fig.savefig(save_to_file, format="png", bbox_inches="tight", pad_inches=0)


def add_days_around_date(date: str, df_original: pd.DataFrame, previous_days = 15, next_days = 15):
    """
    date assumed to be have shape: YYYY-MM-DD 00:00:00+00:00
    """
    
    # date_find là một giá trị datetime
    date_find = pd.to_datetime(date)  # Thay đổi thành ngày bạn muốn

    # Tính khoảng thời gian 15 ngày trước và 15 ngày sau date_find
    start_date = date_find - pd.Timedelta(days=previous_days)
    end_date = date_find + pd.Timedelta(days=next_days)

    # Lọc DataFrame để lấy các hàng nằm trong khoảng thời gian này
    if "Date" in df_original.columns:
        df_original = convert_date_index(df_original)
        
    filtered_df = df_original[start_date:end_date]
    
    return filtered_df


def add_days_around_df(df: pd.DataFrame, df_original: pd.DataFrame, previous_days = 15, next_days = 15):
    if "Date" in df.columns:
        df = convert_date_index(df)
    
    date_column = df.index
    date_min = pd.to_datetime(date_column.min())
    date_max = pd.to_datetime(date_column.max())

    # Tính khoảng thời gian 15 ngày trước và 15 ngày sau date_find
    start_date = date_min - pd.Timedelta(days=previous_days)
    end_date = date_max + pd.Timedelta(days=next_days)

    # Lọc DataFrame để lấy các hàng nằm trong khoảng thời gian này
    if "Date" in df_original.columns:
        df_original = convert_date_index(df_original)

    filtered_df = df_original[start_date:end_date]
    
    return filtered_df

