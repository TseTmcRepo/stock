import motor.motor_asyncio as amotor
import pymongo
import logging
from src.settings import mongodb_settings, SYMBOL_INFO_COLLECTION_NAME, HISTORICAL_DATA_COLLECTION_NAME, \
    CONFIGS_COLLECTION_NAME

# create logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(filename)s:%(lineno)s - %(message)s")
logger = logging.getLogger(__name__)


class MongodbManager:
    """Class to manage mongodb"""

    def __init__(self):
        db_host = mongodb_settings.get("host", "localhost")
        db_port = mongodb_settings.get("port")
        db_name = mongodb_settings.get("name")
        logger.info("connecting to mongodb: {0}@{1}:{2}".format(db_name, db_host, db_port))
        self.client = pymongo.MongoClient(host=db_host, port=db_port)
        self.db = self.client[db_name]
        self._create_indexes()

    def _create_indexes(self):
        """creates indexes for more expeditious queries"""

        index_list = [("datetime", -1), ("symbol", 1)]
        res = self.db[HISTORICAL_DATA_COLLECTION_NAME].create_index(index_list)
        if res:
            logger.info("created index on datetime and symbol fields")

    def _find_duplicates(self, collection, unique_field_names):
        """
        Find duplicates in a collection, which have similar value in their `unique_field_names`
        :param str collection: collection name ,
        :param list unique_field_names: list of field names which should be unique across the collection.
        :return: a list of lists containing duplicate _ids
        """
        duplicate_pipe = {
            "$group": {
                "_id": {str(field): "$" + str(field) for field in unique_field_names},
                "duplicates": {"$push": "$_id"},
                "count": {"$sum": 1}
            }}
        match_pipe = {"$match": {"count": {"$gt": 1}}}
        return [item.get("duplicates") for item in
                self.db[collection].aggregate([duplicate_pipe, match_pipe], allowDiskUse=True)]

    def add_configuration(self, config):
        """
        Add a configuration to the db
        :param dict config: {"name": "config_name", "value": WHATEVER}
        :return:
        """
        res = self.db[CONFIGS_COLLECTION_NAME].update_one({"name": config.get("name")}, {"$set": config}, upsert=True)
        logger.info("{0} config '{1}' with value: {2}".format(
            "added" if bool(res.upserted_id) else "updated", config.get("name"), config.get("value")))

    def get_configuration(self, config_name):
        """
        Get a configuration from the db
        :param str config_name:
        :return:
        """
        config = self.db[CONFIGS_COLLECTION_NAME].find_one({"name": config_name})
        if type(config) is dict:
            return config.get("value")

    def initiate_db(self):
        """Destroy collections"""
        self.db[SYMBOL_INFO_COLLECTION_NAME].drop()
        self.db[HISTORICAL_DATA_COLLECTION_NAME].drop()
        logger.info("All collections dropped")
        self._create_indexes()

    def get_symbols(self):
        """Get all available symbols and their info"""
        return self.db[SYMBOL_INFO_COLLECTION_NAME].find({})

    def get_symbol_names(self):
        """Get a list of all available symbol names in the db"""
        return self.db[SYMBOL_INFO_COLLECTION_NAME].find({}).distinct("symbol")

    def upsert_symbol_info(self, symbol_info):
        """
        Upsert symbol info into the db

        :param dict symbol_info: e.g. {"symbol": "واحیا", "code": "IRO7VHYP0001"}
        :return:
        """
        try:
            symbol = symbol_info.get("symbol")
            if not symbol:
                return
            r = self.db[SYMBOL_INFO_COLLECTION_NAME].update_one({"symbol": symbol}, {"$set": symbol_info}, upsert=True)
            logger.info("{0} info for {1}: {2}".format(
                "added" if bool(r.upserted_id) else "updated", symbol, symbol_info))
        except Exception as e:
            logger.error(str(e))

    def get_symbol_info(self, symbol, key):
        """
        Get the info of a symbol related to a key

        :param str symbol:
        :param str key: either "code", "title", "tsetmc_code"
        :return: info or None
        """

        query = self.db[SYMBOL_INFO_COLLECTION_NAME].find_one({"symbol": symbol})
        if bool(query):
            logger.info("{0} {1} is {2}".format(symbol, key, query.get(key)))
            return query.get(key)
        else:
            logger.error("Symbol not found: {}".format(symbol))

    def add_historical_data(self, list_of_data):
        """
        Add historical data.

        :param list list_of_data: a list of dictionaries containing symbol info. e.g.
        >>> [{
        ... "datetime": datetime.datetime(2018, 7, 29),
        ... "symbol": "شغدیر",
        ... "NumberTrade": 7877,
        ... "VolumeTrade": 25022973,
        ... "Value": 57351375979,
        ... "ClosePrice": 2292,
        ... "LastPrice": 2280,
        ... "PriceMin": 2193,
        ... "PriceMax": 2302,
        ... "PriceFirst": 2193,
        ... "PriceYesterday": 2089
        ... }]
        :return:
        """

        try:
            self.db[HISTORICAL_DATA_COLLECTION_NAME].insert(list_of_data)
            logger.info(
                "{0} data inserted into collection '{1}'".format(len(list_of_data), SYMBOL_INFO_COLLECTION_NAME))
        except Exception as e:
            logger.error(str(e))

    def get_last_n_historical_data(self, symbol, days=1):
        """
        Returns the data related to a symbol for the last number of `n` days

        :param str symbol:
        :param int days:
        :return: list of dict
        """

        logger.info("Getting last {0} data for {1}".format(days, symbol))
        return list(
            self.db[HISTORICAL_DATA_COLLECTION_NAME].find({"symbol": symbol}).sort("datetime", -1).limit(days))

    def remove_duplicates_in_historical_data(self):
        """
        Remove duplicates in historical data.
        Two data are duplicates when their `symbol` and `datetime` are the same.
        """
        duplicates = self._find_duplicates(HISTORICAL_DATA_COLLECTION_NAME, ["symbol", "datetime"])

        # remove all duplicates except one (first one)
        to_remove = []
        for dups in duplicates:
            to_remove += dups[1:]

        result = self.db[HISTORICAL_DATA_COLLECTION_NAME].remove({"_id": {"$in": to_remove}})
        logger.info("removed {0} duplicates from {1}".format(result.get("n"), HISTORICAL_DATA_COLLECTION_NAME))

    def get_symbols_with_at_least_n_historical_data(self, n):
        """
        Get a list of symbols which have at least `n` historical data
        :param int n:
        :return:
        :rtype: list
        """
        symbol_count_pipe = {"$group": {"_id": {"symbol": "$symbol"}, "count": {"$sum": 1}}}
        match_pipe = {"$match": {"count": {"$gte": n}}}
        result = list(
            self.db[HISTORICAL_DATA_COLLECTION_NAME].aggregate([symbol_count_pipe, match_pipe], allowDiskUse=True))
        logger.info("found {0} symbols containing at least {1} historical data".format(len(result), n))
        return [item.get("_id", {}).get("symbol") for item in result]


class AsyncMongodbManager:
    """Class to asynchronously manage mongodb for special operations"""

    def __init__(self, event_loop=None):
        db_host = mongodb_settings.get("host", "localhost")
        db_port = mongodb_settings.get("port")
        db_name = mongodb_settings.get("name")
        logger.info("ASYNC | connecting to mongodb: {0}@{1}:{2}".format(db_name, db_host, db_port))
        if event_loop:
            self.client = amotor.AsyncIOMotorClient(host=db_host, port=db_port, io_loop=event_loop)
        else:
            self.client = amotor.AsyncIOMotorClient(host=db_host, port=db_port)
        self.db = amotor.AsyncIOMotorDatabase(self.client, db_name)

    async def upsert_symbol_info(self, symbol_info):
        """
        Upsert symbol info into the db

        :param dict symbol_info: e.g. {"symbol": "واحیا", "code": "IRO7VHYP0001"}
        :return:
        """
        try:
            collection = amotor.AsyncIOMotorCollection(self.db, SYMBOL_INFO_COLLECTION_NAME)
            symbol = symbol_info.get("symbol")
            if not symbol:
                return
            r = await collection.update_one({"symbol": symbol}, {"$set": symbol_info}, upsert=True)
            logger.info("ASYNC | {0} info for {1}: {2}".format(
                "added" if bool(r.upserted_id) else "updated", symbol, symbol_info))
        except Exception as e:
            logger.error(str(e))

    async def add_historical_data(self, list_of_data):
        """
        Add historical data.

        :param list list_of_data: a list of dictionaries containing symbol info. e.g.
        >>> [{
        ... "datetime": datetime.datetime(2018, 7, 29),
        ... "symbol": "شغدیر",
        ... "NumberTrade": 7877,
        ... "VolumeTrade": 25022973,
        ... "Value": 57351375979,
        ... "ClosePrice": 2292,
        ... "LastPrice": 2280,
        ... "PriceMin": 2193,
        ... "PriceMax": 2302,
        ... "PriceFirst": 2193,
        ... "PriceYesterday": 2089
        ... }]
        :return:
        """

        try:
            collection = amotor.AsyncIOMotorCollection(self.db, HISTORICAL_DATA_COLLECTION_NAME)
            await collection.insert_many(list_of_data)
            logger.info(
                "ASYNC | {0} data inserted into collection '{1}'".format(len(list_of_data),
                                                                         SYMBOL_INFO_COLLECTION_NAME))
        except Exception as e:
            logger.error("ASYNC | " + str(e))

    async def get_last_n_historical_data(self, symbol, days=1):
        """
        Returns the data related to a symbol for the last number of `n` trading days

        :param str symbol:
        :param int days:
        :return: list of dict
        """

        logger.info("ASYNC | Getting last {0} data for {1}".format(days, symbol))
        collection = amotor.AsyncIOMotorCollection(self.db, HISTORICAL_DATA_COLLECTION_NAME)
        return await collection.find({"symbol": symbol}).sort("datetime", -1).limit(days).to_list(days)
