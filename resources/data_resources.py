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

    TGJU_SCREENER = {
        "url": "https://www.sahamyab.com/dideban",
        "header": {
            "Host": "www.sahamyab.com",
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.7,fa;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.sahamyab.com/widget/dideban/-/EXT_BOURSE_GRID_INSTANCE_Gfrp",
            "X-Requested-With": "XMLHttpRequest",
            "Cookie": "GUEST_LANGUAGE_ID=fa_IR; COOKIE_SUPPORT=true; __utma=37739170.1283524191.1529729053.1534646975.1534795045.4; __utmz=37739170.1534795045.4.4.utmcsr=google.com|utmccn=(referral)|utmcmd=referral|utmcct=/; JSESSIONID=CCBABAA7ACDB3FD63966E351572CD0C2; anonymousId=2686196295921335300; __utmc=37739170",
            "Connection": "keep-alive",
        },
        "params": {
            "p_p_id": "EXT_BOURSE_GRID_INSTANCE_Gfrp",
            "p_p_lifecycle": "2",
            "p_p_state": "maximized",
            "p_p_mode": "view",
            "p_p_cacheability": "cacheLevelPage",
            "_EXT_BOURSE_GRID_INSTANCE_Gfrp_struts_action": "/ext/bourse/view",
            "_EXT_BOURSE_GRID_INSTANCE_Gfrp_industry": "",
            "_EXT_BOURSE_GRID_INSTANCE_Gfrp_flow": "",
            "_EXT_BOURSE_GRID_INSTANCE_Gfrp_chartRangeValue": "30",
            "_search": "false",
            "nd": "1536223207914",
            "rows": "10000",
            "page": "1",
            "sidx": "QTotCap",
            "sord": "desc",
            "filters": "{'groupOp':'AND','rules':[]}",
        }
    }
