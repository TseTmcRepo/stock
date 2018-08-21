import asyncio
import uvloop
import matplotlib.pyplot as plt
from src.analyzer import MACDAnalyzer, PivotPointAnalayzer


if __name__ == "__main__":
    m = MACDAnalyzer()
    # p = PivotPointAnalayzer()
    # asyncio.set_event_loop(uvloop.new_event_loop())
    # loop = asyncio.get_event_loop()
    # p._initiate_async_db_manager(loop)
    res = m.run_backtest_with_extremum("فولاد", days=100)
    print(res)
    # res2 = loop.run_until_complete(p.analyze_pivot_point("فولاد", 100))
    # print(res2)
    # res['x'] = res["close"] / 10.0
    # res = m.calulate_macd("فولاد", 1000)
    # res['macd - signal'] = res["macd"] - res["signal"]
    # plt.figure(0)
    # ax = res.plot(y=["macd", "signal"])
    # res.plot(y=["macd - signal"], ax=ax, kind="bar")
    # plt.show()
