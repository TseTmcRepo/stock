from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
import gspread
from gspread.models import Cell


scopes = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = service_account.Credentials.from_service_account_file("client_secret.json", scopes=scopes)
client = gspread.Client(auth=credentials)
client.session = AuthorizedSession(credentials)


class WatchListSheetAPI:
    def __init__(self, spreadsheet_id):
        self.spreadsheet = client.open_by_key(spreadsheet_id)

    def get_list_of_current_symbol(self):
        """
        return a dictionary of symbols currently present in the sheet, and their values (in string format)
        :return:
        """

        try:
            return {
                "symbols": list(map(str.strip, self.spreadsheet.sheet1.col_values(1))),
                "values": list(map(str.strip, self.spreadsheet.sheet1.col_values(2)))
            }
        except Exception as e:
            return {
                "symbols": [],
                "values": []
            }

    def update_symbols_price(self, symbol_value_dict):
        """
        update the last value of a list of symbols
        :param dict symbol_value_dict:
        :return:
        """
        value_cell_list = []
        status_cell_list = []
        symbol_list = self.get_list_of_current_symbol().get("symbols", [])
        value_list = self.get_list_of_current_symbol().get("values", [])
        for i, symbol in enumerate(symbol_list):
            # skip header
            if i == 0:
                continue

            symbol_row = i + 1
            val = symbol_value_dict.get(symbol)
            if type(val) is int:    # Skip stopped symbols
                val = val / 10  # Convert ot Toman
                status_cell_list.append(Cell(symbol_row, 9, "مجاز"))
            else:
                val = value_list[i]
                status_cell_list.append(Cell(symbol_row, 9, "متوقف"))

            value_cell_list.append(Cell(symbol_row, 2, val))

        if value_cell_list:
            self.spreadsheet.sheet1.update_cells(value_cell_list)
        if status_cell_list:
            self.spreadsheet.sheet1.update_cells(status_cell_list)

    def simulate_update_symbols_price(self, symbol_value_dict):
        """
        update the last value of a list of symbols
        :param dict symbol_value_dict:
        :return:
        """
        cell_list = []
        symbol_list = self.get_list_of_current_symbol()
        for i, symbol in enumerate(symbol_list):
            # skip header
            if i == 0:
                continue

            symbol_row = i + 1
            val = symbol_value_dict.get(symbol, 0) / 10    # Convert to Toman
            print(symbol_row, ":", symbol, val)
            cell_list.append(Cell(symbol_row, 2, val))

        return cell_list
