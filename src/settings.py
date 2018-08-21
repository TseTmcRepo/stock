# MongoDB settings and constants
mongodb_settings = {
    "host": "localhost",
    "port": 27017,
    "name": "stock"
}

CONFIGS_COLLECTION_NAME = "config_collection"
SYMBOL_INFO_COLLECTION_NAME = "symbol_code_collection"
HISTORICAL_DATA_COLLECTION_NAME = "historical_data_collection"

# MACD Settings
macd_settings = {
    "slow_range": 26,
    "fast_range": 12,
    "signal_range": 9
}

# Pivot Point settings
pp_settings = {
    "mid_range": 30,
    "long_range": 90
}

# Persian alphabet
PERSIAN_ALPHABET = ["ا", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ", "د", "ذ", "ر", "ز", "ژ", "س", "ش", "ص", "ض", "ط", "ظ",
                    "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن", "و", "ه", "ی"]
