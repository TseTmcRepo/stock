import logging
import numpy as np
import pandas as pd
from src.settings import COMMISSIONS
from src.db_manager import MongodbManager

# create logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(filename)s:%(lineno)s - %(message)s")
logger = logging.getLogger(__name__)


class IndicatorAnalyser:

    def __init__(self, symbol):
        self.symbol = symbol
        self.db_manager = MongodbManager()
        self.df = self.read_data()
        self.indicator_kwargs = {}

    def read_data(self, days=3650):
        """
        Read historical data of a symbol and return in a pandas.DataFrame
        :param int days: number of last trading days
        :rtype: pandas.DataFrame
        """
        data = self.db_manager.get_last_n_historical_data(symbol=self.symbol, days=days)
        return pd.DataFrame(data)

    def update_data(self):
        """update self.df with recent trade info"""
        last_date = self.df["datetime"].iat[-1]
        new_data = self.db_manager.get_recent_historical_data(symbol=self.symbol, start_date=last_date)
        n = len(new_data)
        logger.info("{} new data".format(n))

        if n > 0:
            self.df = self.df.append(pd.DataFrame(new_data), ignore_index=True, sort=False)
            for indicator in self.indicator_kwargs.keys():
                self.__getattribute__(str(indicator) + "_update")(n)

    def calculate_macd(self, n_fast=12, n_slow=26, n_signal=9):
        """
        calculate MACD, MACD Signal and MACD difference

        :param int n_fast:
        :param int n_slow:
        :param int n_signal:
        """
        self.indicator_kwargs["macd"] = {"n_fast": n_fast, "n_slow": n_slow, "n_signal": n_signal}

        self.df["EMA_fast"] = self.df['Last'].ewm(span=n_fast, min_periods=n_slow).mean()
        self.df["EMA_slow"] = self.df['Last'].ewm(span=n_slow, min_periods=n_slow).mean()
        self.df["MACD"] = self.df["EMA_fast"] - self.df["EMA_slow"]
        self.df["MACD_signal"] = self.df["MACD"].ewm(span=n_signal, min_periods=n_signal).mean()
        self.df["MACD_diff"] = self.df["MACD"] - self.df["MACD_signal"]

    def macd_update(self, n):
        """
        update recently added data with calculating MACD, MACD Signal and MACD difference
        :param int n: number of recently added rows
        """
        macd_kwargs = self.indicator_kwargs.get("macd")

        n_fast = macd_kwargs.get("n_fast")
        n_slow = macd_kwargs.get("n_slow")
        n_signal = macd_kwargs.get("n_signal")

        r_fast = 2.0 / (n_fast + 1)
        r_slow = 2.0 / (n_slow + 1)
        r_signal = 2.0 / (n_signal + 1)

        for i in range(-n, 0):
            self.df["EMA_fast"].iat[i] = r_fast * self.df["Last"].iat[i] + (1 - r_fast) * self.df["EMA_fast"].iat[i-1]
            self.df["EMA_slow"].iat[i] = r_slow * self.df["Last"].iat[i] + (1 - r_slow) * self.df["EMA_slow"].iat[i-1]
            self.df["MACD"].iat[i] = self.df["EMA_fast"].iat[i] - self.df["EMA_slow"].iat[i]
            self.df["MACD_signal"].iat[i] = r_signal * self.df["MACD"].iat[i] + (1 - r_signal) * self.df["MACD_signal"].iat[i-1]
            self.df["MACD_diff"].iat[i] = self.df["MACD"].iat[i] - self.df["MACD_signal"].iat[i]

    def calculate_adx(self, n=12, n_adx=25):
        """
        calculate the Average Directional Movement Index for given data.

        :param int n:
        :param int n_adx:
        """
        self.indicator_kwargs["adx"] = {"n": n, "n_adx": n_adx}

        i = 0
        up_i = []
        down_i = []
        while i + 1 <= self.df.index[-1]:
            up_move = self.df.loc[i + 1, 'High'] - self.df.loc[i, 'High']
            down_move = self.df.loc[i, 'Low'] - self.df.loc[i + 1, 'Low']
            if up_move > down_move and up_move > 0:
                up_d = up_move
            else:
                up_d = 0
            up_i.append(up_d)
            if down_move > up_move and down_move > 0:
                down_d = down_move
            else:
                down_d = 0
            down_i.append(down_d)
            i = i + 1
        i = 0
        tr_list = [0]
        while i < self.df.index[-1]:
            tr = max(self.df.loc[i + 1, 'High'], self.df.loc[i, 'Last']) - min(self.df.loc[i + 1, 'Low'],
                                                                               self.df.loc[i, 'Last'])
            tr_list.append(tr)
            i = i + 1
        tr_series = pd.Series(tr_list)
        tr_emw_series = pd.Series(tr_series.ewm(span=n, min_periods=n).mean())
        up_i = pd.Series(up_i)
        down_i = pd.Series(down_i)
        positive_di = pd.Series(up_i.ewm(span=n, min_periods=n).mean() / tr_emw_series)
        negative_di = pd.Series(down_i.ewm(span=n, min_periods=n).mean() / tr_emw_series)
        self.df["ADX"] = pd.Series(
            (abs(positive_di - negative_di) / (positive_di + negative_di)).ewm(span=n_adx, min_periods=n_adx).mean())

    def adx_update(self, n):
        """
        update recently added data with calculating the Average Directional Movement Index
        :param int n: number of recently added rows
        """
        # TODO: this method needs optimization
        adx_kwargs = self.indicator_kwargs.get("adx")
        self.calculate_adx(**adx_kwargs)

    def calculate_ppsr(self, period=12):
        """
        calculate Pivot Points, Supports and Resistances for given data

        :param int period:
        """
        self.indicator_kwargs["ppsr"] = {"period": period}

        highs = self.df["High"].rolling(window=period, min_periods=period).max()
        lows = self.df["Low"].rolling(window=period, min_periods=period).min()
        closes = self.df["Last"].rolling(window=period, min_periods=period).apply(lambda x: x[-1], raw=True)
        self.df["PP"] = (highs + lows + 2 * closes) / 4
        self.df["R1"] = 2 * self.df["PP"] - lows
        self.df["S1"] = 2 * self.df["PP"] - highs
        self.df["R2"] = self.df["PP"] + highs - lows
        self.df["S2"] = self.df["PP"] - highs + lows
        self.df["R3"] = highs + 2 * (self.df["PP"] - lows)
        self.df["S3"] = lows - 2 * (highs - self.df["PP"])

    def ppsr_update(self, n):
        """
        update recently added data with calculating Pivot Point Supports and Resistances

        :param int n: number of recently added rows
        """
        ppsr_kwarges = self.indicator_kwargs.get("ppsr")
        period = ppsr_kwarges.get("period")

        highs = self.df[1 - n - period:]["High"].rolling(window=period, min_periods=period).max()
        lows = self.df[1 - n - period:]["Low"].rolling(window=period, min_periods=period).min()
        closes = self.df[1 - n - period:]["Last"].rolling(window=period, min_periods=period).apply(lambda x: x[-1], raw=1)
        pp = (highs + lows + 2 * closes) / 4
        r1 = pd.Series(2 * pp - lows)
        s1 = pd.Series(2 * pp - highs)
        r2 = pd.Series(pp + highs - lows)
        s2 = pd.Series(pp - highs + lows)
        r3 = pd.Series(highs + 2 * (pp - lows))
        s3 = pd.Series(lows - 2 * (highs - pp))
        psr = {"PP": pp, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}
        self.df.update(pd.DataFrame(psr))


class SignalDriver(IndicatorAnalyser):
    BUY_SIG = 1
    SELL_SIG = -1
    NEUTRAL_SIG = 0

    def generate_macd_cross_signal(self, macd_pre, macd_now, signal_pre, signal_now):
        if macd_now <= signal_now and macd_pre > signal_pre:
            return self.SELL_SIG
        elif macd_now < signal_now and macd_pre >= signal_pre:
            return self.SELL_SIG
        elif macd_now >= signal_now and macd_pre < signal_pre:
            return self.BUY_SIG
        elif macd_now > signal_now and macd_pre <= signal_pre:
            return self.BUY_SIG
        else:
            return self.NEUTRAL_SIG

    def calculate_macd_cross_signal(self):
        """
        Generate signals based on cross point of MACD and signal lines.
        """
        if "macd" not in self.indicator_kwargs:
            self.calculate_macd()

        self.df["macd_cross_signal"] = pd.Series([])
        for i in range(1, self.df.index.max() + 1):
            self.df["macd_cross_signal"].iat[i] = self.generate_macd_cross_signal(
                self.df["MACD"].iat[i-1], self.df["MACD"].iat[i],
                self.df["MACD_signal"].iat[i-1], self.df["MACD_signal"].iat[i])

    def latest_macd_cross_signal(self):
        """
        Generate latest signal based on cross point of MACD and signal lines.
        """
        if "macd" not in self.indicator_kwargs:
            self.calculate_macd()

        return self.generate_macd_cross_signal(
            self.df["MACD"].iat[-2], self.df["MACD"].iat[-1],
            self.df["MACD_signal"].iat[-2], self.df["MACD_signal"].iat[-1])

    def generate_macd_extremum_signal(self, s_pre, s_after, max_slope_diff=0.0, min_slope_diff=0.0):
        """
        Generate signals based on slope values of MACD line

        :param float s_pre:
        :param float s_after:
        :param float max_slope_diff: minimum of absolute value of slope difference at MACD line's maximum
        :param float min_slope_diff: minimum of absolute value of slope difference at MACD line's minimum
        """
        if s_pre > 0 > s_after and abs(s_pre - s_after) >= max_slope_diff:
            return self.SELL_SIG
        elif s_pre < 0 < s_after and abs(s_pre - s_after) >= min_slope_diff:
            return self.BUY_SIG
        else:
            return self.NEUTRAL_SIG

    def calculate_macd_extremum_signal(self, max_slope_diff=0.0, min_slope_diff=0.0):
        """
        Generate signals based on analysis of MACD line extremums

        :param float max_slope_diff: minimum of absolute value of slope difference at MACD line's maximum
        :param float min_slope_diff: minimum of absolute value of slope difference at MACD line's minimum
        """
        if "macd" not in self.indicator_kwargs:
            self.calculate_macd()

        self.df["macd_extremum_signal"] = self.df["MACD"].rolling(window=3, min_periods=3).apply(
            lambda x: self.generate_macd_extremum_signal(x[1] - x[0], x[2] - x[1], max_slope_diff, min_slope_diff),
            raw=True
        )

    def latest_macd_extremum_signal(self, max_slope_diff=0.0, min_slope_diff=0.0):
        """
        Generate latest signal based on analysis of MACD line extremums

        :param float max_slope_diff: minimum of absolute value of slope difference at MACD line's maximum
        :param float min_slope_diff: minimum of absolute value of slope difference at MACD line's minimum
        """
        if "macd" not in self.indicator_kwargs:
            self.calculate_macd()

        s_after = self.df["MACD"].iat[-1] - self.df["MACD"].iat[-2]
        s_pre = self.df["MACD"].iat[-2] - self.df["MACD"].iat[-3]

        return self.generate_macd_extremum_signal(s_pre, s_after, max_slope_diff, min_slope_diff)

    def calculate_adx_signal(self, threshold=0.25):
        """
        Generate signals based on ADX value. this is not a BUY/SELL signal.

        :param float threshold: signal for adx >= threshold is 1, otherwise 0
        :return:
        """
        if "adx" not in self.indicator_kwargs:
            self.calculate_adx()

        self.df["adx_signal"] = self.df["ADX"].apply(lambda x: int(x >= threshold))

    def latest_adx_signal(self, threshold=0.25):
        """
        latest signal based on ADX value. this is not a BUY/SELL signal.

        :param float threshold: signal for adx >= threshold is 1, otherwise 0
        :return:
        """
        if "adx" not in self.indicator_kwargs:
            self.calculate_adx()

        return int(self.df["ADX"].iat[-1] >= threshold)

    def trade(self, signal, capital, share, price, use_commission=False):
        """
        trade the symbol based on the signal.

        :param int signal:
        :param int capital:
        :param int share:
        :param float price: price to perform trading on
        :param bool use_commission: take buy/sell commissions into effect
        :return: (new_capital, new_share)
        """
        sell_coeff = 1 if not use_commission else 1 - COMMISSIONS.get("sell_bours")
        buy_coeff = 1 if not use_commission else 1 + COMMISSIONS.get("buy_bours")

        if signal == self.BUY_SIG:
            logger.info("buying shares with capital: {0} and price: {1}".format(capital, price))
            price = price * buy_coeff
            new_share = capital // price
            new_capital = capital - new_share * price
            logger.info("bought {0} new shares. total shares: {1}, remaining capital: {2}".format(
                new_share, new_share+share, new_capital
            ))
            new_share += share
        elif signal == self.SELL_SIG:
            logger.info("selling {0} shares with price: {1}".format(share, price))
            price = price * sell_coeff
            new_capital = capital + share * price
            logger.info("new capital is {}".format(new_capital))
            new_share = 0
        else:
            new_capital = capital
            new_share = share

        return new_capital, new_share

    def run_back_test(self, signal_list, initial_capital=10000000, use_commission=False, days=None):
        """
        Run back-test for last #days, using multiplication of signals given in signal list

        :param list signal_list: e.g. ["macd_cross_signal"] , ["macd_extremum_signal", "adx_signal"]
        :param int days:
        :param int initial_capital:
        :param bool use_commission: take buy/sell commissions into effect
        :return:
        """
        share = 0
        capital = initial_capital
        last_price = 0
        first_signal_usage_index = 0
        df = self.df if not days else self.df[-days:]

        if not all(signal in self.df for signal in signal_list):
            raise Exception("one or more signal is missing in self.df")

        for i, row in df.iterrows():
            signals = [row[sig] for sig in signal_list]
            if any(pd.isnull(sig) for sig in signals):
                first_signal_usage_index += 1
                continue
            signal = np.array(signals).prod()
            capital, share = self.trade(int(signal), capital, share, row["Last"], use_commission)
            last_price = row["Last"]

        capital += share * last_price
        return capital / initial_capital, last_price / self.df["Last"].iat[first_signal_usage_index]

# class IndicatorAnalyser:
#
#     def __init__(self, symbol):
#         self.symbol = symbol
#         self.db_manager = MongodbManager()
#         self.df = self.read_data()
#
#     def read_data(self, days=3650):
#         """
#         Read historical data of a symbol and return in a pandas.DataFrame
#         :param int days: number of last trading days
#         :rtype: pandas.DataFrame
#         """
#         data = self.db_manager.get_last_n_historical_data(symbol=self.symbol, days=days)
#         return pd.DataFrame(data)
#
#     def update_data(self):
#         """update self.df with recent trade info"""
#         last_date = self.df["datetime"].iat[-1]
#         new_data = self.db_manager.get_recent_historical_data(symbol=self.symbol, start_date=last_date)
#         self.df.append(new_data)
#
#     def moving_average(self, n):
#         """
#         calculate the moving average
#
#         :param n:
#         """
#         m_a = pd.Series(self.df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
#         self.df.join(m_a)
#
#     def exponential_moving_average(self, n):
#         """
#         calculate exponential moving average
#
#         :param n:
#         """
#         e_m_a = pd.Series(self.df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
#         self.df.join(e_m_a)
#
#     def momentum(self, n):
#         """
#         calculate momentum
#
#         :param n:
#         """
#         m = pd.Series(self.df['Close'].diff(n), name='Momentum_' + str(n))
#         self.df.join(m)
#
#     def rate_of_change(self, n):
#         """
#         caluclate rate of change
#
#         :param n:
#         """
#         m = self.df['Close'].diff(n - 1)
#         n = self.df['Close'].shift(n - 1)
#         r_o_c = pd.Series(m / n, name='ROC_' + str(n))
#         self.df.join(r_o_c)
#
#     def average_true_range(self, n):
#         """
#         calculate average true range
#
#         :param n:
#         """
#         i = 0
#         tr_list = [0]
#         while i < self.df.index[-1]:
#             t_r = max(self.df.loc[i + 1, 'High'], self.df.loc[i, 'Close']) \
#                  - min(self.df.loc[i + 1, 'Low'], self.df.loc[i, 'Close'])
#             tr_list.append(t_r)
#             i = i + 1
#         tr_series = pd.Series(tr_list)
#         a_t_r = pd.Series(tr_series.ewm(span=n, min_periods=n).mean(), name='ATR_' + str(n))
#         self.df.join(a_t_r)
#
#     def bollinger_bands(self, n):
#         """
#         calculate Bolinger bands
#
#         :param n:
#         """
#         MA = pd.Series(self.df['Close'].rolling(n, min_periods=n).mean())
#         MSD = pd.Series(self.df['Close'].rolling(n, min_periods=n).std())
#         b1 = 4 * MSD / MA
#         B1 = pd.Series(b1, name='BollingerB_' + str(n))
#         self.df = self.df.join(B1)
#         b2 = (self.df['Close'] - MA + 2 * MSD) / (4 * MSD)
#         B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
#         self.df.join(B2)
#
#     def ppsr(self, n):
#         """calculate Pivot Points, Supports and Resistances for given data
#
#         :param n: period
#         """
#         HIGHS = self.df["High"].rolling(window=n, min_periods=n).max()
#         LOWS = self.df["Low"].rolling(window=n, min_periods=n).min()
#         CLOSES = self.df["Close"].rolling(window=n, min_periods=n).apply(lambda x: x[-1])
#         PP = (HIGHS + LOWS + 2 * CLOSES) / 4
#         R1 = pd.Series(2 * PP - LOWS)
#         S1 = pd.Series(2 * PP - HIGHS)
#         R2 = pd.Series(PP + HIGHS - LOWS)
#         S2 = pd.Series(PP - HIGHS + LOWS)
#         R3 = pd.Series(HIGHS + 2 * (PP - LOWS))
#         S3 = pd.Series(LOWS - 2 * (HIGHS - PP))
#         psr = {'PP': PP, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}
#         PSR = pd.DataFrame(psr)
#         self.df.join(PSR)
#
#     def stochastic_oscillator_k(self):
#         """
#         calculate stochastic oscillator %K for given data.
#         """
#         SOk = pd.Series((self.df['Close'] - self.df['Low']) / (self.df['High'] - self.df['Low']), name='SO%k')
#         self.df.join(SOk)
#
#     def stochastic_oscillator_d(self, n):
#         """
#         calculate stochastic oscillator %D for given data.
#
#         :param n:
#         """
#         SOk = pd.Series((self.df['Close'] - self.df['Low']) / (self.df['High'] - self.df['Low']), name='SO%k')
#         SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SO%d_' + str(n))
#         self.df.join(SOd)
#
#     def trix(self, n):
#         """
#         calculate TRIX for given data.
#
#         :param n:
#         """
#         EX1 = self.df['Close'].ewm(span=n, min_periods=n).mean()
#         EX2 = EX1.ewm(span=n, min_periods=n).mean()
#         EX3 = EX2.ewm(span=n, min_periods=n).mean()
#         i = 0
#         ROC_l = [np.nan]
#         while i + 1 <= self.df.index[-1]:
#             ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
#             ROC_l.append(ROC)
#             i = i + 1
#         Trix = pd.Series(ROC_l, name='Trix_' + str(n))
#         self.df.join(Trix)
#
#     def average_directional_movement_index(self, n, n_ADX):
#         """
#         calculate the Average Directional Movement Index for given data.
#
#         :param n:
#         :param n_ADX:
#         """
#         i = 0
#         UpI = []
#         DoI = []
#         while i + 1 <= self.df.index[-1]:
#             UpMove = self.df.loc[i + 1, 'High'] - self.df.loc[i, 'High']
#             DoMove = self.df.loc[i, 'Low'] - self.df.loc[i + 1, 'Low']
#             if UpMove > DoMove and UpMove > 0:
#                 UpD = UpMove
#             else:
#                 UpD = 0
#             UpI.append(UpD)
#             if DoMove > UpMove and DoMove > 0:
#                 DoD = DoMove
#             else:
#                 DoD = 0
#             DoI.append(DoD)
#             i = i + 1
#         i = 0
#         TR_l = [0]
#         while i < self.df.index[-1]:
#             TR = max(self.df.loc[i + 1, 'High'], self.df.loc[i, 'Close']) - min(self.df.loc[i + 1, 'Low'],
#                                                                                 self.df.loc[i, 'Close'])
#             TR_l.append(TR)
#             i = i + 1
#         TR_s = pd.Series(TR_l)
#         ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean())
#         UpI = pd.Series(UpI)
#         DoI = pd.Series(DoI)
#         PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean() / ATR)
#         NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean() / ATR)
#         ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span=n_ADX, min_periods=n_ADX).mean(),
#                         name='ADX_' + str(n) + '_' + str(n_ADX))
#         self.df.join(ADX)
#
#     def macd(self, n_fast, n_slow, n_signal):
#         """
#         calculate MACD, MACD Signal and MACD difference
#
#         :param n_fast:
#         :param n_slow:
#         :param n_signal:
#         """
#         ema_fast = pd.Series(self.df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
#         ema_slow = pd.Series(self.df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
#         macd = pd.Series(ema_fast - ema_slow, name='MACD_{0}_{1}'.format(n_fast, n_slow))
#         macd_signal = pd.Series(macd.ewm(span=n_signal, min_periods=n_signal).mean(), name='MACDsign_' + str(n_signal))
#         macd_diff = pd.Series(macd - macd_signal, name='MACDdiff_{0}_{1}_{2}'.format(n_fast, n_slow, n_signal))
#         self.df.join(macd)
#         self.df.join(macd_signal)
#         self.df.join(macd_diff)
#
#     def mass_index(self):
#         """
#         calculate the Mass Index for given data.
#         """
#         Range = self.df['High'] - self.df['Low']
#         EX1 = Range.ewm(span=9, min_periods=9).mean()
#         EX2 = EX1.ewm(span=9, min_periods=9).mean()
#         Mass = EX1 / EX2
#         MassI = pd.Series(Mass.rolling(25).sum(), name='Mass Index')
#         self.df.join(MassI)
#
#     def vortex_indicator(self, n):
#         """
#         calculate the Vortex Indicator for given data.
#
#         Vortex Indicator described here:
#             http://www.vortexindicator.com/VFX_VORTEX.PDF
#         :param n:
#         """
#         i = 0
#         TR = [0]
#         while i < self.df.index[-1]:
#             Range = max(self.df.loc[i + 1, 'High'], self.df.loc[i, 'Close']) - min(self.df.loc[i + 1, 'Low'],
#                                                                                    self.df.loc[i, 'Close'])
#             TR.append(Range)
#             i = i + 1
#         i = 0
#         VM = [0]
#         while i < self.df.index[-1]:
#             Range = abs(self.df.loc[i + 1, 'High'] - self.df.loc[i, 'Low']) - abs(
#                 self.df.loc[i + 1, 'Low'] - self.df.loc[i, 'High'])
#             VM.append(Range)
#             i = i + 1
#         VI = pd.Series(pd.Series(VM).rolling(n).sum() / pd.Series(TR).rolling(n).sum(), name='Vortex_' + str(n))
#         self.df.join(VI)
#
#     def kst_oscillator(self, r1, r2, r3, r4, n1, n2, n3, n4):
#         """
#         calculate KST Oscillator for given data.
#
#         :param r1:
#         :param r2:
#         :param r3:
#         :param r4:
#         :param n1:
#         :param n2:
#         :param n3:
#         :param n4:
#         """
#         M = self.df['Close'].diff(r1 - 1)
#         N = self.df['Close'].shift(r1 - 1)
#         ROC1 = M / N
#         M = self.df['Close'].diff(r2 - 1)
#         N = self.df['Close'].shift(r2 - 1)
#         ROC2 = M / N
#         M = self.df['Close'].diff(r3 - 1)
#         N = self.df['Close'].shift(r3 - 1)
#         ROC3 = M / N
#         M = self.df['Close'].diff(r4 - 1)
#         N = self.df['Close'].shift(r4 - 1)
#         ROC4 = M / N
#         KST = pd.Series(
#             ROC1.rolling(n1).sum() + ROC2.rolling(n2).sum() * 2 + ROC3.rolling(n3).sum() * 3 + ROC4.rolling(
#                 n4).sum() * 4,
#             name='KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(
#                 n2) + '_' + str(n3) + '_' + str(n4))
#         self.df.join(KST)
#
#     def relative_strength_index(self, n):
#         """calculate Relative Strength Index(RSI) for given data.
#
#         :param n:
#         """
#         i = 0
#         UpI = [0]
#         DoI = [0]
#         while i + 1 <= self.df.index[-1]:
#             UpMove = self.df.loc[i + 1, 'High'] - self.df.loc[i, 'High']
#             DoMove = self.df.loc[i, 'Low'] - self.df.loc[i + 1, 'Low']
#             if UpMove > DoMove and UpMove > 0:
#                 UpD = UpMove
#             else:
#                 UpD = 0
#             UpI.append(UpD)
#             if DoMove > UpMove and DoMove > 0:
#                 DoD = DoMove
#             else:
#                 DoD = 0
#             DoI.append(DoD)
#             i = i + 1
#         UpI = pd.Series(UpI)
#         DoI = pd.Series(DoI)
#         PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
#         NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
#         RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
#         self.df.join(RSI)
#
#     def true_strength_index(self, r, s):
#         """
#         calculate True Strength Index (TSI) for given data.
#
#         :param r:
#         :param s:
#         """
#         M = pd.Series(self.df['Close'].diff(1))
#         aM = abs(M)
#         EMA1 = pd.Series(M.ewm(span=r, min_periods=r).mean())
#         aEMA1 = pd.Series(aM.ewm(span=r, min_periods=r).mean())
#         EMA2 = pd.Series(EMA1.ewm(span=s, min_periods=s).mean())
#         aEMA2 = pd.Series(aEMA1.ewm(span=s, min_periods=s).mean())
#         TSI = pd.Series(EMA2 / aEMA2, name='TSI_' + str(r) + '_' + str(s))
#         self.df.join(TSI)
#
#     def accumulation_distribution(self, n):
#         """
#         calculate Accumulation/Distribution for given data.
#
#         :param n:
#         """
#         ad = (2 * self.df['Close'] - self.df['High'] - self.df['Low']) / (self.df['High'] - self.df['Low']) * self.df[
#             'Volume']
#         M = ad.diff(n - 1)
#         N = ad.shift(n - 1)
#         ROC = M / N
#         AD = pd.Series(ROC, name='Acc/Dist_ROC_' + str(n))
#         self.df.join(AD)
#
#     def chaikin_oscillator(self):
#         """
#         calculate Chaikin Oscillator for given data.
#         """
#         ad = (2 * self.df['Close'] - self.df['High'] - self.df['Low']) / (self.df['High'] - self.df['Low']) * self.df[
#             'Volume']
#         Chaikin = pd.Series(ad.ewm(span=3, min_periods=3).mean() - ad.ewm(span=10, min_periods=10).mean(),
#                             name='Chaikin')
#         self.df.join(Chaikin)
#
#     def money_flow_index(self, n):
#         """
#         calculate Money Flow Index and Ratio for given data.
#
#         :param n:
#         """
#         PP = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
#         i = 0
#         PosMF = [0]
#         while i < self.df.index[-1]:
#             if PP[i + 1] > PP[i]:
#                 PosMF.append(PP[i + 1] * self.df.loc[i + 1, 'Volume'])
#             else:
#                 PosMF.append(0)
#             i = i + 1
#         PosMF = pd.Series(PosMF)
#         TotMF = PP * self.df['Volume']
#         MFR = pd.Series(PosMF / TotMF)
#         MFI = pd.Series(MFR.rolling(n, min_periods=n).mean(), name='MFI_' + str(n))
#         self.df.join(MFI)
#
#     def on_balance_volume(self, n):
#         """
#         calculate On-Balance Volume for given data.
#
#         :param n:
#         """
#         i = 0
#         OBV = [0]
#         while i < self.df.index[-1]:
#             if self.df.loc[i + 1, 'Close'] - self.df.loc[i, 'Close'] > 0:
#                 OBV.append(self.df.loc[i + 1, 'Volume'])
#             if self.df.loc[i + 1, 'Close'] - self.df.loc[i, 'Close'] == 0:
#                 OBV.append(0)
#             if self.df.loc[i + 1, 'Close'] - self.df.loc[i, 'Close'] < 0:
#                 OBV.append(-self.df.loc[i + 1, 'Volume'])
#             i = i + 1
#         OBV = pd.Series(OBV)
#         OBV_ma = pd.Series(OBV.rolling(n, min_periods=n).mean(), name='OBV_' + str(n))
#         self.df.join(OBV_ma)
#
#     def force_index(self, n):
#         """
#         calculate Force Index for given data.
#
#         :param n:
#         """
#         F = pd.Series(self.df['Close'].diff(n) * self.df['Volume'].diff(n), name='Force_' + str(n))
#         self.df.join(F)
#
#     def ease_of_movement(self, n):
#         """
#         calculate Ease of Movement for given data.
#
#         :param n:
#         """
#         EoM = (self.df['High'].diff(1) + self.df['Low'].diff(1)) * (self.df['High'] - self.df['Low']) / (
#                 2 * self.df['Volume'])
#         Eom_ma = pd.Series(EoM.rolling(n, min_periods=n).mean(), name='EoM_' + str(n))
#         self.df.join(Eom_ma)
#
#     def commodity_channel_index(self, n):
#         """
#         calculate Commodity Channel Index for given data.
#
#         :param n:
#         """
#         PP = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
#         CCI = pd.Series((PP - PP.rolling(n, min_periods=n).mean()) / PP.rolling(n, min_periods=n).std(),
#                         name='CCI_' + str(n))
#         self.df.join(CCI)
#
#     def coppock_curve(self, n):
#         """
#         calculate Coppock Curve for given data.
#
#         :param n:
#         """
#         M = self.df['Close'].diff(int(n * 11 / 10) - 1)
#         N = self.df['Close'].shift(int(n * 11 / 10) - 1)
#         ROC1 = M / N
#         M = self.df['Close'].diff(int(n * 14 / 10) - 1)
#         N = self.df['Close'].shift(int(n * 14 / 10) - 1)
#         ROC2 = M / N
#         Copp = pd.Series((ROC1 + ROC2).ewm(span=n, min_periods=n).mean(), name='Copp_' + str(n))
#         self.df.join(Copp)
#
#     def keltner_channel(self, n):
#         """
#         calculate Keltner Channel for given data.
#
#         :param n:
#         """
#         KelChM = pd.Series(((self.df['High'] + self.df['Low'] + self.df['Close']) / 3).rolling(n, min_periods=n).mean(),
#                            name='KelChM_' + str(n))
#         KelChU = pd.Series(
#             ((4 * self.df['High'] - 2 * self.df['Low'] + self.df['Close']) / 3).rolling(n, min_periods=n).mean(),
#             name='KelChU_' + str(n))
#         KelChD = pd.Series(
#             ((-2 * self.df['High'] + 4 * self.df['Low'] + self.df['Close']) / 3).rolling(n, min_periods=n).mean(),
#             name='KelChD_' + str(n))
#         self.df.join(KelChM)
#         self.df.join(KelChU)
#         self.df.join(KelChD)
#
#     def ultimate_oscillator(self):
#         """
#         calculate Ultimate Oscillator for given data.
#         """
#         i = 0
#         TR_l = [0]
#         BP_l = [0]
#         while i < self.df.index[-1]:
#             TR = max(self.df.loc[i + 1, 'High'], self.df.loc[i, 'Close']) - min(self.df.loc[i + 1, 'Low'],
#                                                                                 self.df.loc[i, 'Close'])
#             TR_l.append(TR)
#             BP = self.df.loc[i + 1, 'Close'] - min(self.df.loc[i + 1, 'Low'], self.df.loc[i, 'Close'])
#             BP_l.append(BP)
#             i = i + 1
#         UltO = pd.Series((4 * pd.Series(BP_l).rolling(7).sum() / pd.Series(TR_l).rolling(7).sum()) + (
#                 2 * pd.Series(BP_l).rolling(14).sum() / pd.Series(TR_l).rolling(14).sum()) + (
#                                  pd.Series(BP_l).rolling(28).sum() / pd.Series(TR_l).rolling(28).sum()),
#                          name='Ultimate_Osc')
#         self.df.join(UltO)
#
#     def donchian_channel(self, n):
#         """
#         calculate donchian channel of given pandas data frame.
#
#         :param n:
#         """
#         i = 0
#         dc_l = []
#         while i < n - 1:
#             dc_l.append(0)
#             i += 1
#
#         i = 0
#         while i + n - 1 < self.df.index[-1]:
#             dc = max(self.df['High'].ix[i:i + n - 1]) - min(self.df['Low'].ix[i:i + n - 1])
#             dc_l.append(dc)
#             i += 1
#
#         donchian_chan = pd.Series(dc_l, name='Donchian_' + str(n))
#         donchian_chan = donchian_chan.shift(n - 1)
#         self.df.join(donchian_chan)
#
#     def standard_deviation(self, n):
#         """
#         calculate Standard Deviation for given data.
#
#         :param n:
#         """
#         self.df.join(pd.Series(self.df['Close'].rolling(n, min_periods=n).std(), name='STD_' + str(n)))
