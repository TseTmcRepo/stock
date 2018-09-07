import csv
import time
import pytz
import random
import logging
import datetime
import requests
import persian
import aiohttp
import asyncio
import uvloop
import janus
from io import StringIO
from resources.data_resources import DataResource
from src.db_manager import MongodbManager, AsyncMongodbManager
from src.settings import PERSIAN_ALPHABET, DataStrings, TIMEZONE

# create logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(filename)s:%(lineno)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_symbol_name_code(symbol):
    return {
        "symbol": persian.convert_ar_characters(symbol.get("LVal18AFC", "").strip()),
        "symbol_raw": symbol.get("LVal18AFC", "").strip(),
        "ISIN": symbol.get("InstrumentID").strip()
    }


class CrawlSymbols:

    def __init__(self):
        self.dbmanager = MongodbManager()
        self.async_dbmanager = None
        self.symbol_queue_to_crawl = janus.Queue().async_q
        self.symbol_crawled_set = set()

    def _initiate_async_db_manager(self, event_loop):
        """
        Initiate an async db manager, passing a pre-built event loop
        :param asyncio.Loop event_loop:
        :return:
        """
        self.async_dbmanager = AsyncMongodbManager(event_loop=event_loop)

    def _is_market_open(self):
        """
        Return True if stock market is open now, otherwise False.
        :return:
        """
        now = datetime.datetime.now(tz=TIMEZONE)
        if now.weekday in [3, 4]:   # Thursday and Friday
            return False
        if datetime.time(8, 30, tzinfo=TIMEZONE) <= now.time() <= datetime.time(12, 30, tzinfo=TIMEZONE):
            return True
        else:
            return False

    async def _fetch_url(self, session, url, data=None, params=None, method="POST"):
        """
        Send an HTTP request and return data

        :param aiohttp.ClientSession session:
        :param str url:
        :param data: POST data
        :param params: GET parameters
        :param str method: "POST" or "GET"
        :return:
        """
        async with session.request(method=method, url=url, data=data, params=params) as response:
            if response.status == 200:
                return await response.text()
            else:
                logger.warning("response status {0} when fetching {1} with {2}".format(response.status, url, data))
                return None

    def _crawl_symbol_and_code(self):
        """
        Get a list of available symbols and their ISIN code

        :rtype: list
        """
        resource = DataResource.FIPIRAN_SYMBOL_INFO
        r = requests.post(resource.get("url"), data={"id": ""}, headers=resource.get("headers"))
        if r.status_code == 200:
            symbol_code_list = list(map(clean_symbol_name_code, r.json()))
            logger.info("Number of symbols: {}".format(len(symbol_code_list)))
            return symbol_code_list
        else:
            logger.error("Cannot retrieve list of available symbols. status code: {0}".format(r.status_code))
            return []

    def _extract_titles_and_tsetmc_codes(self, response):
        """
        extract title and tsetmc unique code from response returned from tsetmc.com

        :param str response: response text returning from tsetmc.com
        :return:
        :rtype: list
        """
        extracted_info = []
        all_symbols = response.strip().split(";")
        all_symbols.remove("")
        for item in all_symbols:
            info = item.split(",")
            extracted_info.append({
                "symbol": persian.convert_ar_characters(info[0].strip()),
                "title": info[1].strip(),
                "tsetmc_code": info[2].strip()
            })
        return extracted_info

    async def _gather_title_and_tsetmc_code(self, session, url):
        """
        Send post request to the given url, fetch response and extract title and tsetmc_code of symbols.
        finally stores info into the db

        :param aiohttp.ClientSession session:
        :param str url:
        :return:
        """
        if self.symbol_queue_to_crawl.qsize() > 0:
            logger.info("Remaining prefixes to process: {}".format(self.symbol_queue_to_crawl.qsize()))
            prefix_info = await self.symbol_queue_to_crawl.get()
        else:
            return

        prefix = prefix_info.get("prefix")
        try:
            response = await self._fetch_url(session, url, params={"skey": prefix}, method="GET")
            extracted_info = self._extract_titles_and_tsetmc_codes(response)
            logger.info("gather titles and tsetmc codes for prefix: {}".format(prefix))
            for info in extracted_info:
                if info.get("tsetmc_code") and info.get("symbol") not in self.symbol_crawled_set:
                    await self.async_dbmanager.upsert_symbol_info(symbol_info=info)
                    self.symbol_crawled_set.add(info.get("symbol"))

        except Exception as e:
            logger.error(str(e))
            if prefix_info.get("num_tries") < 20:
                prefix_info["num_tries"] += 1
                await self.symbol_queue_to_crawl.put(prefix_info)
            else:
                logger.error("request try exceeded limit for prefix: {}".format(prefix))

    async def _crawl_title_and_tsetmc_code(self, **kwargs):
        """
        Get title and tsetmc unique code for each symbol
        :return:
        """
        resource = DataResource.TSETMC_SYMBOL_INFO
        headers = resource.get("headers")
        url = resource.get("url")
        workers = 50  # number of concurrent requests to the server before sleep
        tasks = []
        async with aiohttp.ClientSession(headers=headers, read_timeout=10) as session:
            while self.symbol_queue_to_crawl.qsize() > 0:
                for _ in range(workers):
                    tasks.append(asyncio.ensure_future(self._gather_title_and_tsetmc_code(session, url)))

                await asyncio.gather(*tasks)
                # sleep to prevent our IP from being blocked. sleep more whe remaining symbols is fewer than 20
                time.sleep(max(20 - self.symbol_queue_to_crawl.qsize(), 1))

    def _extract_trade_info_csv(self, csv_row, symbol_info):
        """
        Extract trade information of a day, given a row of data. this is based on tsetmc.com data format
        :param dict csv_row:
        :param dict symbol_info:
        :return:
        :rtype: dict
        """
        date_time = datetime.datetime.strptime(csv_row["<DTYYYYMMDD>"].strip(), "%Y%m%d")
        date_time = date_time.replace(hour=12, minute=30, tzinfo=TIMEZONE)
        return {
            DataStrings.SYMBOL: symbol_info.get("symbol"),
            DataStrings.DATETIME: date_time,
            DataStrings.PRICELAST: int(float(csv_row["<LAST>"])),
            DataStrings.PRICECLOSING: int(float(csv_row["<CLOSE>"])),
            DataStrings.PRICELOW: int(float(csv_row["<LOW>"])),
            DataStrings.PRICEHIGH: int(float(csv_row["<HIGH>"])),
            DataStrings.PRICEFIRST: int(float(csv_row["<FIRST>"])),
            DataStrings.PRICELASTDAY: int(float(csv_row["<OPEN>"])),
            DataStrings.TRANSNUMBER: int(csv_row["<OPENINT>"]),
            DataStrings.TRANSVOLUME: int(csv_row["<VOL>"]),
            DataStrings.TRANSVALUE: int(csv_row["<VALUE>"]),
        }

    def _parse_historical_data(self, csv_text, symbol_info):
        """
        Get a CSV table and return a list of dictionaries, each containing a row of daily data
        :param str csv_text:
        :param dict symbol_info:
        :return:
        :rtype: list of dict
        """
        extracted_data_list = []
        file = StringIO(csv_text)
        csv_reader = csv.DictReader(file, delimiter=",")
        for row in csv_reader:
            extracted_data_list.append(self._extract_trade_info_csv(row, symbol_info))

        return extracted_data_list

    def _prepare_tsetmc_request_params(self, symbol_info):
        """
        Prepare params of a GET request to get historical data
        :param dict symbol_info:
        :return:
        :rtype: dict
        """
        return {
            "a": "1",
            "t": "i",
            "i": symbol_info.get("tsetmc_code"),
            "b": "0"
        }

    async def _gather_historical_data(self, session, url):
        """
        Send post request to the given url, fetch response and extract daily symbol trade data for all its history,
        finally stores the extracted data

        :param aiohttp.ClientSession session:
        :param str url:
        :return:
        """
        if self.symbol_queue_to_crawl.qsize() > 0:
            logger.info("Remaining symbols to process: {}".format(self.symbol_queue_to_crawl.qsize()))
            symbol_info = await self.symbol_queue_to_crawl.get()
        else:
            return

        if not symbol_info.get("tsetmc_code"):
            logger.warning("{} doe's not have tsetmc code".format(symbol_info.get("symbol")))
            return

        # create request data to sent to the tsetmc server
        request_params = self._prepare_tsetmc_request_params(symbol_info)
        try:
            csv_table = await self._fetch_url(session=session, url=url, params=request_params, method="GET")
            extracted_data_list = self._parse_historical_data(csv_table, symbol_info)
            logger.info("{0} rows extracted for {1}".format(len(extracted_data_list), symbol_info.get("symbol")))
            if bool(extracted_data_list):
                await self.async_dbmanager.add_historical_data(extracted_data_list)
        except Exception as e:
            logger.error(str(e))
            if symbol_info.get("num_tries") < 20:
                symbol_info["num_tries"] += 1
                await self.symbol_queue_to_crawl.put(symbol_info)
            else:
                logger.error("request try exceeded limit for symbol: {}".format(symbol_info.get("symbol")))

    async def _crawl_symbol_historical_data(self, **kwargs):
        """
        Schedule and execute getting all historical data of a symbols present
        in self.symbol_queue_to_crawl
        """
        resource = DataResource.TSETMC_HISTORY
        headers = resource.get("headers")
        url = resource.get("url")
        workers = 10  # number of concurrent requests to the server before sleep

        logger.info("crawl historical data for {0} symbols with {1} workers".format(
            self.symbol_queue_to_crawl.qsize(), workers))

        async with aiohttp.ClientSession(headers=headers, read_timeout=30) as session:
            while self.symbol_queue_to_crawl.qsize() > 0:
                tasks = []
                for _ in range(workers):
                    tasks.append(
                        asyncio.ensure_future(self._gather_historical_data(session, url)))

                await asyncio.gather(*tasks)

                # sleep to prevent our IP from being blocked
                time.sleep(random.randint(1, 2))

    def _extract_trade_info_tgju_list(self, info_list):
        """
        Extract trade info for the symbols given in info_list
        :param list info_list:
        :return:
        """
        extracted_data = []
        # TODO: The line bellow can be tricky when we repalce original time with market close time
        now = datetime.datetime.now(tz=TIMEZONE).replace(hour=12, minute=30, second=0, microsecond=0)
        for data in info_list:
            try:
                extracted_data.append({
                    DataStrings.SYMBOL: persian.convert_ar_characters(data.get("LVal18AFC")),
                    DataStrings.DATETIME: now,
                    DataStrings.PRICELAST: int(data.get("PDrCotVal")),
                    DataStrings.PRICECLOSING: int(data.get("PClosing")),
                    DataStrings.PRICELOW: int(data.get("PriceMin")),
                    DataStrings.PRICEHIGH: int(data.get("PriceMax")),
                    DataStrings.PRICEFIRST: int(data.get("PriceFirst")),
                    DataStrings.PRICELASTDAY: int(data.get("PriceYesterday")),
                    DataStrings.TRANSNUMBER: int(data.get("ZTotTran")),
                    DataStrings.TRANSVOLUME: int(data.get("QTotTran5J")),
                    DataStrings.TRANSVALUE: int(data.get("QTotCap")),
                })
            except Exception as e:
                logger.error("error extracting info for {0}: {1}".format(data.get("LVal18AFC"), str(e)))

        return extracted_data

    def _crawl_symbols_recent_data(self):
        """
        Get up-to-date data for currently trading symbols from tgju.org
        :return:
        """
        resource = DataResource.TGJU_SCREENER
        r = requests.get(resource.get("url"), params=resource.get("params"), headers=resource.get("headers"))
        if r.status_code == 200:
            response = r.json()
            logger.info("got update for {} symbols.".format(response.get("totalrecords")))
            extracted_data = self._extract_trade_info_tgju_list(response.get("rows", []))
            self.dbmanager.add_historical_data(extracted_data)
            if len(extracted_data) > 0:
                dt_exp = extracted_data[0].get(DataStrings.DATETIME).strftime("%Y%m%d-%H%M")
                symbol_list = [data.get(DataStrings.SYMBOL) for data in extracted_data]
                self.dbmanager.upsert_list_of_update_symbols(dt_exp=dt_exp, update_symbols=symbol_list)
        else:
            logger.error("cannot retrieve up-to-date data from tjgu.org. status code: {}".format(r.status_code))

    def _fill_symbol_queue_to_crawl(self, list_of_items):
        """
        Fill symbol_queue_to_crawl with given symbols and their info. add num_tries = 0 to each item
        :param list list_of_items:
        :return:
        """
        self.symbol_queue_to_crawl = janus.Queue().async_q  # reinitialise
        for item in list_of_items:
            item.update({"num_tries": 0})
            self.symbol_queue_to_crawl.put_nowait(item)

    def _add_or_update_symbols(self, list_of_symbol_info):
        """
        Update collection of symbols and their info
        :param list list_of_symbol_info: list of dict
        :return:
        """
        for info in list_of_symbol_info:
            self.dbmanager.upsert_symbol_info(info)

    def _generate_prefixes(self):
        """Generate a list of 2-letter length prefix from persian alphabet"""
        prefixes = []
        for i in PERSIAN_ALPHABET:
            for j in PERSIAN_ALPHABET:
                prefixes.append(i + j)
        return prefixes + self.dbmanager.get_symbol_names()

    def _run_a_crawl_job(self, crawl_func, **kwargs):
        """
        Create an event loop and run a crawl job until it completes
        :param func crawl_func: function to use in crawling
        :return:
        """
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        loop = asyncio.get_event_loop()
        self._initiate_async_db_manager(event_loop=loop)
        loop.run_until_complete(crawl_func(**kwargs))

    def initial_run(self):
        """Crawl data, Initiate db, and put crawled data into the db for the first time"""
        list_of_symbol_and_code = self._crawl_symbol_and_code()

        self.dbmanager.initiate_db()
        self._add_or_update_symbols(list_of_symbol_info=list_of_symbol_and_code)
        self._fill_symbol_queue_to_crawl(list_of_items=[{"prefix": x} for x in self._generate_prefixes()])
        self._run_a_crawl_job(self._crawl_title_and_tsetmc_code)
        self._fill_symbol_queue_to_crawl(list_of_items=self.dbmanager.get_symbols())
        self._run_a_crawl_job(self._crawl_symbol_historical_data)
        self.dbmanager.remove_duplicates_in_historical_data()
        self.dbmanager.add_configuration({"name": "crawler_is_initiated", "value": True})

    def update_data(self):
        """Update the data in the db"""
        self._crawl_symbols_recent_data()
        self.dbmanager.remove_duplicates_in_historical_data()

    def run(self):
        """Run the crawler"""
        update_interval = 600   # seconds
        checking_interval = 10  # seconds

        crawler_is_initiated = self.dbmanager.get_configuration("crawler_is_initiated")
        if not crawler_is_initiated:
            logger.info("initial run ...")
            self.initial_run()
        while True:
            if self._is_market_open():
                logger.info("updating the data ...")
                self.update_data()
                time.sleep(update_interval-checking_interval)
            else:
                logger.info("market is closed")
                time.sleep(checking_interval)


if __name__ == "__main__":
    crawler = CrawlSymbols()
    crawler.run()
