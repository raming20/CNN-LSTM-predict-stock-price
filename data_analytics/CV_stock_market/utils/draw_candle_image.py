from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import keras
import tensorflow as tf
import numpy as np
import matplotlib
import ta.momentum
import ta
from ta.utils import dropna
from ta import add_all_ta_features
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
from datetime import datetime
import ta.trend
import matplotlib.pyplot
import matplotlib.axes
import sys
sys.path.append("../../")


def convert_number_index(df: pd.DataFrame):
    """
    df is a pandas.DataFrame and assumed to have a columns: ['Date']
    """
    df = df.copy()

    assert "Date" in df.columns, "Date must in columns"

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.reset_index(drop=True)

    return df


def convert_date_index(df: pd.DataFrame):
    """
    df is a pandas.DataFrame and assumed to have a columns: ['Date']
    """
    df = df.copy()

    assert "Date" in df.columns, "Date must in columns"

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date", drop=False)

    return df


def draw_candle_image(
        df_draw: pd.DataFrame,
        save_to_file=None,
        
        show_macd=False,
        show_macd_signal=False,
        show_macd_histogram=False,
        
        show_BB_avg=False,
        show_BB_high=False,
        show_BB_low=False,
        
        color_BB_avg="red",
        color_BB_high="orange",
        color_BB_low="orange",
        
        show_SMA=False,
        show_EMA_9=False,
        show_EMA_50=False,
        show_EMA_200=False,
        
        color_SMA="Aqua",
        color_EMA_9="navy",
        color_EMA_50="maroon",
        color_EMA_200="navy",
        
        show_x_y=False,
        show_volume=False,
        figscale=0.5,
        figcolor="white",
        figratio=(5, 5),
        preview_image=True,
        return_image_tensor=False):
    """
    df is a pandas.DataFrame and assumed to have a columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] 
    or ['Open', 'High', 'Low', 'Close', 'Volume'] and index is Date data.
    
    If macd = True then df must have additional columns: ["Macd", "Macd_signal", "Macd_histogram"], if not then 
    function automatically calculates Macd indices.
    """
    df_draw = df_draw.copy()

    if 'Date' in df_draw.columns:
        df_draw = convert_number_index(df_draw)

    mc = mpf.make_marketcolors(
        up='green', down='red', wick='inherit', edge='inherit', volume='inherit', )
    style = mpf.make_mpf_style(marketcolors=mc, figcolor=figcolor)

    adps = []
    if show_macd:
        assert "Macd" in df_draw.columns
        assert "Macd_signal" in df_draw.columns
        assert "Macd_histogram" in df_draw.columns

    fig: plt.Figure
    axlist: list[matplotlib.axes.Axes]

    panel_ratios = [6]
    if show_volume:
        panel_ratios.append(2)

    if show_macd or show_macd_signal or show_macd_histogram:
        panel_ratios.append(2)
        # display the macd in the bottom of image
        panel_order_of_macd = len(panel_ratios) - 1

        if show_macd:
            adps.append(
                mpf.make_addplot(
                    df_draw["Macd"], panel=panel_order_of_macd, color="blue", ylabel="MACD")
            )
        if show_macd_signal:
            adps.append(
                mpf.make_addplot(df_draw["Macd_signal"],
                                         panel=panel_order_of_macd, color="red")
            )
        if show_macd_histogram:
            adps.append(
                mpf.make_addplot(df_draw["Macd_histogram"], type="bar",
                                         panel=panel_order_of_macd, color="gray", alpha=0.5)
            )
    
    if True in [show_BB_avg, show_BB_high, show_BB_low, show_SMA, show_EMA_9, show_EMA_50, show_EMA_200]:
        if show_BB_avg:
            adps.append(
                mpf.make_addplot(df_draw['BB_avg'], color='red')
            )
        
        if show_BB_high:
            adps.append(
                mpf.make_addplot(df_draw['BB_high'], color='orange')
            )
        
        if show_BB_low:
            adps.append(
                mpf.make_addplot(df_draw['BB_low'], color='orange')
            )
        
        if show_SMA:
            adps.append(
                mpf.make_addplot(df_draw['SMA'], color='Aqua')
            )
        
        if show_EMA_9:
            adps.append(
                mpf.make_addplot(df_draw['EMA_9'], color='navy')
            )
        
        if show_EMA_50:
            adps.append(
                mpf.make_addplot(df_draw['EMA_50'], color='maroon')
            )
        
        if show_EMA_200:
            adps.append(
                mpf.make_addplot(df_draw['EMA_200'], color='navy')
            )


    # Vẽ biểu đồ nến với khối lượng và ẩn các trục
    fig, axlist = mpf.plot(
        convert_date_index(df_draw),
        type='candle',
        style=style,
        volume=show_volume,  # Hiển thị khối lượng
        axisoff=not show_x_y,  # Bỏ trục x và y
        returnfig=True,  # Trả về đối tượng Figure để tùy chỉnh
        figratio=figratio,  # Điều chỉnh tỷ lệ khung hình để thu hẹp khoảng cách
        figscale=figscale,  # Tăng kích thước biểu đồ để làm các nến gần nhau hơn
        addplot=adps,
        panel_ratios=panel_ratios
    )

    # Loại bỏ các nhãn trục và tiêu đề
    if not show_x_y:
        for ax in axlist:
            ax.axis("off")

    if type(save_to_file) == str:
        fig.savefig(
            save_to_file,
            format="png",
            bbox_inches="tight",
            pad_inches=0
        )

    image_array = None
    if return_image_tensor:
        canvas = FigureCanvas(fig)
        canvas.draw()

        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image_array = image_array.reshape(
            canvas.get_width_height()[::-1] + (3,))

    if not preview_image:
        fig.clear()
        plt.close(fig)

    return image_array


def normalize_date(date):
    date = str(date)
    if len(date.strip()) <= 10:
        date = f"{date.strip()} 00:00:00+00:00"
    return pd.to_datetime(date)


def get_date_index(df_original, date):
    date = normalize_date(date)
    date_str = str(date.date())
    index_of_date = df_original.loc[df_original["Date"] == date_str].index[0]
    return index_of_date


def add_days_around_date(date: str, df_original: pd.DataFrame, previous_days=15, next_days=15):
    """
    date assumed to be have shape: YYYY-MM-DD 00:00:00+00:00
    """

    date = normalize_date(date)

    # Lọc DataFrame để lấy các hàng nằm trong khoảng thời gian này
    if "Date" in df_original.columns:
        df_original = convert_number_index(df_original)

    index_of_date = get_date_index(df_original, date)
    start_index = index_of_date - previous_days
    end_index = index_of_date + next_days
    
    filtered_df = df_original[start_index:end_index+1]

    return filtered_df


def normalize_candle_image(
    image_tensor: np.ndarray, 
    combine_into_one_image=False,
    convert_red=False,
    convert_green=False):
    
    image_tensor = image_tensor.copy()

    if convert_red:
        red_face = image_tensor[:, :, 0]
        red_face[(red_face > 0) & (red_face != 255)] = 255
        image_tensor[:, :, 0] = red_face

    if convert_green:
        green_face = image_tensor[:, :, 1]
        green_face[(green_face > 0) & (green_face != 128)] = 128
        image_tensor[:, :, 1] = green_face

    if combine_into_one_image:
        image_tensor = np.max(image_tensor, axis=2)
        image_tensor = tf.expand_dims(image_tensor, axis=2)

    image_tensor = image_tensor / 255

    return image_tensor


def duplicate_last_row(
    df: pd.DataFrame, 
    time_duplicate: int):
    
    # Xác định dòng cuối cùng và số lần duplicate
    last_row = df.iloc[-1]

    # Tạo DataFrame mới chứa các bản duplicate của dòng cuối
    duplicates = pd.DataFrame([last_row] * time_duplicate, columns=df.columns)

    # Kết hợp DataFrame ban đầu với các dòng duplicate
    df_extended = pd.concat([df, duplicates], ignore_index=True)

    return df_extended



