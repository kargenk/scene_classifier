#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import shutil
import base64
import urllib
import argparse
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                     AppleWebKit/537.36 (KHTML, like Gecko) \
                     Chrome/74.0.3729.169 Safari/537.36'}


def img_download(srcs, save_dir, maximum):
    i = 0
    if maximum >= len(srcs):
        maximum = len(srcs)

    print("Downloading...")
    for idx, src in enumerate(srcs[:maximum]):
        if idx % 50 == 0 or idx == len(srcs)-1:
            print("|" + ("■" * (20 * idx // (len(srcs)-1))) + (" -" * (20 - 20 * idx //
                                                                       (len(srcs)-1))) + "|", f"{100*idx//(len(srcs)-1)}%")  # ダウンロードの進捗示すやつ
        file_path = os.path.join(save_dir, 'img_' + str(idx) + '.jpg')
        src = src.get_attribute("src")
        if src != None:
            # 画像に変換--
            if "base64," in src:
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(src.split(",")[1]))
            else:
                res = requests.get(src, stream=True)
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(res.raw, f)
            i += 1
    print(f"Download is complete. {i} images are downloaded.")


def img_search(query, driver):
    url = 'https://www.google.co.jp/search' + \
        '?q=' + urllib.parse.quote(query) + '&source=lnms&tbm=isch'

    driver.get(url)

    for t in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1.5)
    try:
        driver.find_element_by_class_name(
            "mye4qd").click()  # 「検索結果をもっと表示」ってボタンを押してる
    except:
        pass
    for t in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1.5)

    srcs = driver.find_elements(By.XPATH, '//img[@class="rg_i Q4LuWd"]')

    return srcs


def main():
    parser = argparse.ArgumentParser(description='Options for downloading')
    parser.add_argument('-q', '--query', default='ミリシタ',
                        type=str, help='search word')
    parser.add_argument('-n', '--num_max', default=10,
                        type=int, help='num of download images')
    parser.add_argument('-d', '--directory', default='./data',
                        type=str, help='save directory')
    args = parser.parse_args()

    query = '+'.join(args.query.split())  # 複数のキーワードを'+'で繋げる
    num_max = args.num_max
    save_dir = os.path.join(args.directory, query)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    driver = webdriver.Chrome('./data/chromedriver.exe')

    img_srcs = img_search(query, driver)
    img_download(img_srcs, save_dir, num_max)
    driver.quit()  # ウィンドウを閉じる


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    sys.exit()
