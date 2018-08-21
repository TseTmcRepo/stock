from src.crawler import CrawlSymbols


if __name__ == "__main__":
    c = CrawlSymbols()
    # c.initial_run()
    # c.dbmanager.initiate_db()
    # a = c._crawl_symbol_and_code()
    # c._add_or_update_symbols(a)
    # c._fill_symbol_queue_to_crawl(list_of_items=[{"prefix": x} for x in c._generate_prefixes()])
    # c._run_a_crawl_job(c._crawl_title_and_tsetmc_code)
    #
    c._fill_symbol_queue_to_crawl(list_of_items=c.dbmanager.get_symbols())
    c._run_a_crawl_job(c._crawl_symbol_historical_data, **{"days": 3650})
