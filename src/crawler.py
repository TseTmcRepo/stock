import csv
import time
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
from bs4 import BeautifulSoup
from src.settings import PERSIAN_ALPHABET
from resources.data_resources import DataResource
from src.db_manager import MongodbManager, AsyncMongodbManager

# create logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(filename)s:%(lineno)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_symbol_name_code(symbol):
    return {
        "symbol": persian.convert_ar_characters(symbol.get("LVal18AFC", "").strip()),
        "symbol_raw": symbol.get("LVal18AFC", "").strip(),
        "code": symbol.get("InstrumentID").strip()
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

    def _extrac_titles_and_tsetmc_codes(self, response):
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
            extracted_info = self._extrac_titles_and_tsetmc_codes(response)
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
                logger.error("request try exceeded limit for symbol: {}".format(symbol))

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
                # sleep to prevent our IP from being blocked. sleep more whe remaining symbols is fewer than 2*workers
                time.sleep(max(2 * workers - self.symbol_queue_to_crawl.qsize(), 1))

    def _get_start_and_end_strings(self, days):
        """
        Return date strings form start and end of a last #days period
        :param int days:
        :return:
        """
        now = datetime.datetime.now()
        end_date = now.strftime("%Y%m%d")
        start_date = (now - datetime.timedelta(days=days)).strftime("%Y%m%d")
        return start_date, end_date

    def _extract_daily_trade_info(self, csv_row, symbol_info):
        """
        Extract trade information of a day, given a row of data. this is based on fipiran.com data format
        :param dict csv_row:
        :param dict symbol_info:
        :return:
        :rtype: dict
        """
        return {
            "symbol": symbol_info.get("symbol"),
            "datetime": datetime.datetime.strptime(csv_row["<DTYYYYMMDD>"].strip(), "%Y%m%d"),
            "close": int(float(csv_row["<LAST>"])),
            "average": int(float(csv_row["<CLOSE>"])),
            "low": int(float(csv_row["<LOW>"])),
            "high": int(float(csv_row["<HIGH>"])),
            "first": int(float(csv_row["<FIRST>"])),
            "last_day": int(float(csv_row["<OPEN>"])),
            "num_trade": int(csv_row["<OPENINT>"]),
            "vol": int(csv_row["<VOL>"]),
            "value": int(csv_row["<VALUE>"]),
        }

    def _parse_historical_data_table(self, csv_text, symbol_info):
        """
        Get a CSV table and return a list of dictionaries, each containing a row of daily data
        :param str csv_text:
        :param dict symbol_info
        :return:
        :rtype: list of dict
        """
        extracted_data_list = []
        file = StringIO(csv_text)
        csv_reader = csv.DictReader(file, delimiter=",")
        for row in csv_reader:
            extracted_data_list.append(self._extract_daily_trade_info(row, symbol_info))

        return extracted_data_list

    def _prepare_historical_request_params(self, symbol_info, start_date, end_date):
        """
        Prepare params of a GET request to get historical data
        :param dict symbol_info:
        :param str start_date: start date
        :param str end_date: end date
        :return:
        :rtype: dict
        """
        return {
            "a": "1",
            "t": "i",
            "i": symbol_info.get("tsetmc_code"),
            "b": "0"
        }

        # return {
        #     "a": "InsTrade",
        #     "InsCode": symbol_info.get("tsetmc_code"),
        #     "DateFrom": start_date,
        #     "DateTo": end_date,
        #     "b": "0"
        # }

    async def _gather_historical_data(self, session, url, start_date, end_date):
        """
        Send post request to the given url, fetch response and extract daily symbol data between start_date
        and end_date, finally stores the extracted data

        :param aiohttp.ClientSession session:
        :param str url:
        :param str start_date:
        :param str end_date:
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
        # TODO: Currently tsetmc.com only return all history of a symbol. start_date and end_date does not work
        request_params = self._prepare_historical_request_params(symbol_info, start_date, end_date)
        try:
            csv_table = await self._fetch_url(session=session, url=url, params=request_params, method="GET")
            extracted_data_list = self._parse_historical_data_table(csv_table, symbol_info)
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
        Schedule and execute getting historical data of last #days timespan for symbols present
        in self.symbol_code_queue

        :param int days:
        :return
        """
        resource = DataResource.TSETMC_HISTORY
        headers = resource.get("headers")
        url = resource.get("url")
        days = kwargs.get("days", 1)
        workers = 10  # number of concurrent requests to the server before sleep
        start_date, end_date = self._get_start_and_end_strings(days=days)

        logger.info("crawl historical data from {0} to {1} for {2} symbols with {3} workers".format(
            start_date, end_date, self.symbol_queue_to_crawl.qsize(), workers))

        async with aiohttp.ClientSession(headers=headers, read_timeout=30) as session:
            while self.symbol_queue_to_crawl.qsize() > 0:
                tasks = []
                for _ in range(workers):
                    tasks.append(
                        asyncio.ensure_future(self._gather_historical_data(session, url, start_date, end_date)))

                await asyncio.gather(*tasks)

                # sleep to prevent our IP from being blocked
                time.sleep(random.randint(1, 2))

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
        """Generate a list 2-letter length prefix from persian alphabet"""
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
        self._run_a_crawl_job(self._crawl_symbol_historical_data, **{"days": 3650})
        self.dbmanager.remove_duplicates_in_historical_data()
        self.dbmanager.add_configuration({"name": "crawler_is_initiated", "value": True})

    def update_data(self):
        """Update the data in the db"""
        list_of_symbol_and_code = self._crawl_symbol_and_code()

        self._add_or_update_symbols(list_of_symbol_info=list_of_symbol_and_code)
        self._fill_symbol_queue_to_crawl(list_of_items=list_of_symbol_and_code)
        self._run_a_crawl_job(self._crawl_symbol_historical_data, **{"days": 0})
        self.dbmanager.remove_duplicates_in_historical_data()

    def run(self):
        """Run the crawler"""
        crawler_is_initiated = self.dbmanager.get_configuration("crawler_is_initiated")

        if crawler_is_initiated:
            self.update_data()
        else:
            self.initial_run()
