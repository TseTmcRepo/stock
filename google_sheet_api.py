from google.oauth2 import service_account
from googleapiclient.discovery import build
from google.auth.transport.requests import AuthorizedSession
import gspread


scopes = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = service_account.Credentials.from_service_account_file("client_secret.json", scopes=scopes)
client = gspread.Client(auth=credentials)
client.session = AuthorizedSession(credentials)
# service = build("sheets", version="v4", credentials=credentials)


class WatchListSheetAPI:
    def __init__(self, spreadsheet_id):
        self.spreadsheet = client.open_by_key(spreadsheet_id)

    def get_list_of_current_instrument(self):
        """
        return list of instruments currently present in the sheet
        :return:
        """

        try:
            return self.spreadsheet.sheet1.col_values(1)
        except Exception:
            return []

    def update_instruments_price(self, instrument_value_dict):
        """
        update the last value of a list of instruments
        :param dict instrument_value_dict:
        :return:
        """
        ins_list = self.get_list_of_current_instrument()
        for ins, val in instrument_value_dict:
            if ins in ins_list:
                ins_index = ins_list.index(ins)
                self.spreadsheet.sheet1.update_cell(ins_index, 2, val)
