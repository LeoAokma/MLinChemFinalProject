"""
author: Leo Aokma

System Environment
OS: Microsoft Windows 10 Professional x64, WSL with Ubuntu 16.0.4 LTS or MacOS Big Sur 11.0 above
(No requirments of necessity)

Python Environment
python==3.8
"""

from bs4 import BeautifulSoup
import requests
import time


def get_data():
    """
    Get database from the website
    :return: Return acquired data in specific form
    """
    i = 1
    while True:
        try:
            url = 'https://darkreactions.haverford.edu/database.html?reactions_only=1&page={}'.format(i + 1)
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
                         " (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
            headers = {'User-Agent': user_agent}
            response = requests.get(url, headers=headers, verify=False)
            # print(response.text)
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find_all("tr", class_="quantities")
            print(content)
            time.sleep(3)
            i += 1
        except Exception:
            break
