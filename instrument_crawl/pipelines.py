# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import json
import os


class InstrumentCrawlPipeline(object):
    def process_item(self, item, spider):
        return item


class JsonPipeline(object):
    def __init__(self):
        self.items = dict()
        self.filename = "ins_val.json"
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def close_spider(self, spider):
        with open(self.filename, "w") as f:
            json.dump(self.items, f, ensure_ascii=False)
        f.close()

    def process_item(self, item, spider):
        self.items.update(item)
        return item
