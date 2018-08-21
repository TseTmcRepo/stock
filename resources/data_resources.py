class DataResource:
    FIPIRAN_SYMBOL_INFO = {
        "url": "http://www.fipiran.com/DataService/AutoCompletesymbol",
        "headers": {
            "Host": "www.fipiran.com",
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.7,fa;q=0.3",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "http://www.fipiran.com/DataService/TradeIndex",
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": "112",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
    }

    FIPIRAN_HISTORY = {
        "url": "http://www.fipiran.com/DataService/Exportsymbol",
        "headers": {
            "Host": "www.fipiran.com",
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.7,fa;q=0.3",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "http://www.fipiran.com/DataService/TradeIndex",
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": "112",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
    }

    TSETMC_SYMBOL_INFO = {
        "url": "http://www.tsetmc.com/tsev2/data/search.aspx",
        "headers": {
            "Host": "tsetmc.com",
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.7,fa;q=0.3",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "http://www.tsetmc.com/Loader.aspx?ParTree=15",
            "X-Requested-With": "XMLHttpRequest",
            "Cookie": "ASP.NET_SessionId=f00as1we1hsprkt1wppkaimm",
            "Connection": "keep-alive",
        }
    }

    TSETMC_HISTORY = {
        "url": "http://www.tsetmc.com/tsev2/data/Export-txt.aspx",
        "headers": {
            "Host": "tsetmc.com",
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.7,fa;q=0.3",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "http://tsetmc.com/Loader.aspx?ParTree=121C10",
            "Cookie": "ASP.NET_SessionId=cjc50wziezjjkpu4v2hbge55",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "0",
        }
    }
