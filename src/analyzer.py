import logging
import uvloop
import asyncio
import jdatetime
import pandas as pd
from src.settings import macd_settings, pp_settings
from src.db_manager import MongodbManager, AsyncMongodbManager

# create logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(filename)s:%(lineno)s - %(message)s")
logger = logging.getLogger(__name__)


class MACDAnalyzer:
    """class to perform MACD analysis on historical data"""

    def __init__(self, slow_range=macd_settings.get("slow_range"), fast_range=macd_settings.get("fast_range"),
                 signal_range=macd_settings.get("signal_range")):
        self.dbmanager = MongodbManager()
        self.slow_range = slow_range
        self.fast_range = fast_range
        self.signal_range = signal_range

    def calulate_macd(self, symbol, days=None):
        """
        MACD related calculations of a symbol for last #days trading days. It calculates slow and fast EMAs, and then
        drives MACD and signal values.
        :param str symbol:
        :param int days:
        :return:
        :rtype: pandas.DataFrame
        """
        days = days or self.slow_range
        logger.info("calculating MACD for last {0} trading days of {1}".format(days, symbol))
        try:
            data = self.dbmanager.get_last_n_historical_data(symbol=symbol, days=days + self.slow_range)
            data.reverse()  # to sort from earliest to latest
            df = pd.DataFrame({
                "close": [item.get("close") for item in data],
                "first": [item.get("first") for item in data],
                "datetime": [item.get("datetime") for item in data]
            })
            df["slow_ema"] = df["close"].ewm(span=self.slow_range).mean()
            df["fast_ema"] = df["close"].ewm(span=self.fast_range).mean()
            df["macd"] = df["fast_ema"] - df["slow_ema"]
            df["signal"] = df["macd"].ewm(span=self.signal_range).mean()

            return df
        except Exception as e:
            logger.error("error calculating MACD for {0}: {1}".format(symbol, str(e)))

    def signal_on_cross(self, df):
        """
        Generate `BUY`, `SELL`, or `NEUTRAL` signals based on cross point of MACD and signal lines.

        :param pandas.DataFrame df: containing MACD and signal values for two consecutive days
        :return:
        """
        if df["macd"].iat[-1] <= df["signal"].iat[-1] and df["macd"].iat[-2] > df["signal"].iat[-2]:
            return "SELL"
        elif df["macd"].iat[-1] < df["signal"].iat[-1] and df["macd"].iat[-2] >= df["signal"].iat[-2]:
            return "SELL"
        elif df["macd"].iat[-1] >= df["signal"].iat[-1] and df["macd"].iat[-2] < df["signal"].iat[-2]:
            return "BUY"
        elif df["macd"].iat[-1] > df["signal"].iat[-1] and df["macd"].iat[-2] <= df["signal"].iat[-2]:
            return "BUY"
        else:
            return "NEUTRAL"

    def signal_on_extremum(self, df, neutral_tol=0.0, forecast_tol=0.0):
        """
        Generate `BUY`, `SELL`, or `NEUTRAL` signals based on analysis of MACD line

        :param pandas.DataFrame df: containing MACD values for three consecutive days
        :param float neutral_tol: a float in range [0, 1]. An absolute relative slope change of r=abs((s2-s1)/s1)
        is ALWAYS considered as a NEUTRAL signal, if r < neutral_tol.
        :param float forecast_tol: a float in range [0, 1]. A relative slope of r=|s2/s1| is ALWAYS considered as a
        BUY/SELL signal, if r < forecast_tol
        :return:
        """
        signal = ""
        sf = df.drop("fast_ema", axis=1).drop("slow_ema", axis=1).drop("signal", axis=1)
        sf["jalali"] = df["datetime"].apply(lambda x: jdatetime.datetime.fromgregorian(datetime=x))
        sf.drop("datetime", axis=1, inplace=True)
        slope_21 = df["macd"].iat[-2] - df["macd"].iat[-3]
        slope_10 = (df["macd"].iat[-1] - df["macd"].iat[-2])

        # add 0.0000001 to prevent div by zero
        if abs(slope_21) < 0.0000001:
            slope_21 = 0.0000001 if slope_21 >= 0 else -0.0000001

        if abs((slope_10 - slope_21) / slope_21) <= neutral_tol:
            signal = "NEUTRAL"
        elif abs(slope_10/slope_21) <= forecast_tol:
            signal = "BUY" if slope_21 < 0 else "SELL"
        elif slope_21 > 0 > slope_10:
            signal = "SELL"
        elif slope_21 < 0 < slope_10:
            signal = "BUY"
        else:
            signal = "NEUTRAL"

        print(sf)
        print(signal, slope_21, slope_10, slope_10/slope_21, (slope_10 - slope_21)/slope_21)
        print("="*10)
        return signal

    def run_backtest_with_extremum(self, symbol, days=1, initial_capital=10000):

        df = self.calulate_macd(symbol=symbol, days=days)
        capital = initial_capital
        portfo = 0
        pd.set_option('display.max_columns', None)
        for index, row in df.iterrows():
            try:
                if index < self.slow_range:
                    continue
                slice = df[index-3:index]
                signal = self.signal_on_extremum(df=slice)
                if signal == "BUY":
                    portfo = capital // df["first"].iat[index+1]
                    capital -= portfo * df["first"].iat[index+1]
                elif signal == "SELL":
                    capital += portfo * df["first"].iat[index+1]
            except Exception as e:
                continue

        capital = capital + portfo * df["close"].iat[-1]
        return capital, capital/initial_capital, df["close"].iat[-1]/df["close"].iat[self.slow_range]


class PivotPointAnalayzer:

    def __init__(self, mid_range=pp_settings.get("mid_range"), long_range=pp_settings.get("long_range")):
        self.async_dbmanager = None
        self.mid_range = mid_range
        self.long_range = long_range

    def _initiate_async_db_manager(self, event_loop):
        """
        Initiate an async db manager, passing a pre-built event loop
        :param asyncio.Loop event_loop:
        :return:
        """
        self.async_dbmanager = AsyncMongodbManager(event_loop=event_loop)

    async def analyze_pivot_point(self, symbol, days=0):
        """
        Calculate pivot point of a symbol for a #days session, and drive resistances and supports.
        :param str symbol:
        :param int days:
        :return:
        :rtype: tuple
        """
        logger.info("pivot point analysis for last {0} days of {1}".format(days, symbol))
        try:
            data = await self.async_dbmanager.get_last_n_historical_data(symbol=symbol, days=days)
            data.reverse()  # to sort from earliest to latest
            df = pd.DataFrame({
                "close": [item.get("close") for item in data],
                "low": [item.get("low") for item in data],
                "high": [item.get("high") for item in data],
                "datetime": [item.get("datetime") for item in data]
            })

            # calculate pivot point, supports and resistances
            high = df["high"].max()
            low = df["low"].min()
            close = df["close"].iat[-1]

            print(high, low, close)

            pp = 0.25 * (high + low + 2*close)
            r1 = 2*pp - low
            r2 = pp + high - low
            r3 = high + 2*(pp - low)
            s1 = 2*pp - high
            s2 = pp + low - high
            s3 = low - 2*(high - pp)

            return pp, r1, r2, r3, s1, s2, s3
        except Exception as e:
            logger.error("error calculating pivot point for {0}: {1}".format(symbol, str(e)))
