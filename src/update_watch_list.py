import json
from src.google_sheet_api import WatchListSheetAPI
from src.paths import INSTRUMENT_JSON_FILE

spreadsheet_id = "1e_nVGTGZc4Dd5_yhdVUdkyRj0JP_Br3S_rVwang6XYU"
api = WatchListSheetAPI(spreadsheet_id=spreadsheet_id)


def get_symbols_data():
    """Get symbols data from json file
    """
    try:
        with open(INSTRUMENT_JSON_FILE) as f:
            data = json.load(f)
    except Exception:
        data = dict()

    return data


def update_sheet():
    """Update excel watch list sheet"""
    symbol_val_dict = get_symbols_data()
    api.update_symbols_price(symbol_value_dict=symbol_val_dict)


if __name__ == "__main__":
    update_sheet()
