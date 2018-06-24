import json
from google_sheet_api import WatchListSheetAPI

spreadsheet_id = "1e_nVGTGZc4Dd5_yhdVUdkyRj0JP_Br3S_rVwang6XYU"
api = WatchListSheetAPI(spreadsheet_id=spreadsheet_id)


def get_instruments_data():
    """Get instruments data from json file
    """
    try:
        with open("ins_val.json") as f:
            data = json.load(f)
    except Exception:
        data = dict()

    return data


def update_sheet():
    """Update excel watch list sheet"""
    ins_val_dict = get_instruments_data()
    api.update_instruments_price(instrument_value_dict=ins_val_dict)


if __name__ == "__main__":
    update_sheet()