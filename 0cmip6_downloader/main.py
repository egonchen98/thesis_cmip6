import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import tools.colab as colab


# Press the green button in the gutter to run the script.:
if __name__ == '__main__':
    config = json.loads(Path('./resources/config.json').read_text())
    database = config['download_database']

    while True:

        print('Starting downloading 10 files at a time: ....')
        records = [colab.get_1_record() for i in range(100)]
        if 'Finished' in records:
            break
        records = list({v['url']: v for v in records}.values())

        pool = ThreadPoolExecutor(max_workers=10)
        # Add downloading threads
        for index, res in enumerate(records):
            pool.submit(colab.run, res, database)
            time.sleep(1)
        pool.shutdown()



