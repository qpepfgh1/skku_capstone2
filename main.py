from multiprocessing import Process, Queue
import multiprocessing
import logging
from webviewController import webView_start
from flaskController import flask_start

import time

# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# logs 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# logs 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
# file_handler = logging.handlers.TimedRotatingFileHandler(
#   filename='logs/log_main', when='midnight', interval=1, delay=True, encoding='utf-8'
#   )
# file_handler.suffix = '%Y%m%d.txt' # 파일명 끝에 붙여줌; ex. logs-20190811
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    state = True
    while state:
        th1 = Process(target=webView_start, args=())  # web view
        th2 = Process(target=flask_start, args=())  # flask

        th2.start()
        th1.start()

        th1.join()
        th2.terminate()
        state = False