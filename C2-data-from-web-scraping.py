import bs4 as bs
import re
import time
from urllib.request import urlopen

ROOT_URL = "https://news.sky.com/story/energy-crisis-plan-for-three-hour-power-blackouts-to-prioritise-heating-in-event-of-gas-shortages-as-eu-agrees-to-cut-electricity-use-12713253"
p = re.compile(r'((<p[^>]*>(.(?!</p>))*.</p>\s*){3,})', re.DOTALL)

def get_url_content(url):
    with urlopen(url) as url:
        raw_html = url.read().decode('utf-8')
        # clean up, extract HTML and save to database
        for match in p.finditer(raw_html):
            paragraph = match.group(1)

            soup = bs.BeautifulSoup(paragraph,'lxml')
            for link in soup.findAll('a'):
                new_url = (link.get('href'))
                # add a delay between each scrape
                time.sleep(1)
                get_url_content(new_url)

raw_html = get_url_content(ROOT_URL)