import scrapy
from scrapy.signals import engine_stopped
import persian
import json


class InstrumentSpider(scrapy.Spider):
    name = "instrument_spider"

    def __init__(self, *args, **kwargs):
        self.ins_val_dict = dict()
        super(InstrumentSpider, self).__init__()

    start_urls = [
        "http://www.fipiran.com/Market/LupBourse",
        "http://www.fipiran.com/Market/LupBourseH",
        "http://www.fipiran.com/Market/LupBourseO",
        "http://www.fipiran.com/Market/LupOTC",
        "http://www.fipiran.com/Market/LupOTCH",
        "http://www.fipiran.com/Market/LupOTCO",
        "http://www.fipiran.com/Market/LupBasePTC",
    ]

    def parse(self, response):
        for row in response.css("table").css("tbody").css("tr"):
            yield {
                persian.convert_ar_characters(row.css("td")[0].css("a::text").extract_first()).strip():
                int(str(row.css("td::text")[1].extract()).strip().replace(",", "").replace("،", ""))
            }