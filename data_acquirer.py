"""
author: Leo Aokma

System Environment
OS: Microsoft Windows 10 Professional x64, WSL with Ubuntu 16.0.4 LTS or MacOS Big Sur 11.0 above
(No requirements of necessity)

Python Environment
python==3.8
"""

from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
from io import StringIO


def convert2table(html_table):
    """
    Convert html table to dictionary
    :param html_table:
    :return:
    """
    table = pd.read_html(html_table)
    return table


def climb(url):
    isData = True
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
                 " (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    headers = {'User-Agent': user_agent}
    response = requests.get(url, headers=headers, verify=False)
    if '404 Error' in response.text:
        isData = False
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all("table")
    return tables, isData


def get_data():
    """
    Get database from the website
    :return: Return acquired data in specific form
    """
    i = 1
    while True:
        url = 'https://darkreactions.haverford.edu/database.html?reactions_only=1&page={}'.format(i)
        content, stat = climb(url)
        if not stat:
            break
        for _ in content:
            convert2table(StringIO(str(_)))
        time.sleep(3)
        i += 1


def test():
    content, stat = climb('https://darkreactions.haverford.edu/database.html?reactions_only=1&page=1')
    tables = []
    for _ in content:
        table = convert2table(StringIO(str(_)))
        tables.append(table)
