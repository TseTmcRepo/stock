import pytz

TIMEZONE = pytz.timezone("Asia/Tehran")

# MongoDB settings and constants
mongodb_settings = {
    "host": "localhost",
    "port": 27017,
    "name": "stock"
}

CONFIGS_COLLECTION = "configs"
SYMBOL_INFO_COLLECTION = "symbol_info"
HISTORICAL_DATA_COLLECTION = "historical_data"
UPDATED_SYMBOLS_COLLECTION = "updated_symbols"


class DataStrings:
    SYMBOL = "symbol"
    DATETIME = "datetime"
    PRICELAST = "Last"
    PRICEFIRST = "First"
    PRICEHIGH = "High"
    PRICELOW = "Low"
    PRICECLOSING = "Close"
    PRICELASTDAY = "Previous"
    TRANSNUMBER = "TransNum"
    TRANSVOLUME = "TransVol"
    TRANSVALUE = "TransVal"


# Commissions
COMMISSIONS = {
    "buy_bours": 0.00486,
    "sell_bours": 0.01029,
    "buy_fara_bours": 0.00474,
    "sell_fara_bours": 0.01011,
}

# Persian alphabet
PERSIAN_ALPHABET = ["ا", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ", "د", "ذ", "ر", "ز", "ژ", "س", "ش", "ص", "ض", "ط", "ظ",
                    "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن", "و", "ه", "ی"]
