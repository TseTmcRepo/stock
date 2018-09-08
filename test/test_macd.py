import asyncio
import uvloop
import matplotlib.pyplot as plt
from src.technical import SignalDriver


if __name__ == "__main__":
    s = SignalDriver("واحیا")
    s.calculate_macd_extremum_signal()
    s.df["slopes"] = s.df["MACD"].rolling(7, 7).apply(lambda x: x[6]-2*x[3]+x[0])
    plt.figure(0)
    ax = s.df[-100:].plot(y=["MACD"])
    ax2 = ax.twinx()
    s.df[-100:].plot(y=["slopes"], ax=ax2, color="r")
    # s.df[-100:].plot(y=["macd_extremum_signal"], ax=ax2, drawstyle="steps", color="r")
    plt.show()
