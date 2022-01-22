import requests
from bs4 import BeautifulSoup
import pandas as pd

import settings


if __name__ == '__main__':

    df = pd.DataFrame(columns=['title', 'content', 'category', 'year', 'month', 'day'])

    start_url = 'https://www.thehindu.com/archive/web/'
    index_response = BeautifulSoup(requests.get(start_url).text, 'html.parser')
    month_urls = [link.get('href') for link in index_response.select('div.archiveBorder ul.archiveMonthList li a')]

    for month_url in month_urls:
        # Iterate over month urls from Aug 2009 to Jan 2022
        month_response = BeautifulSoup(requests.get(month_url).text, 'html.parser')
        day_urls = [link.get('href') for link in month_response.select('table.archiveTable td a.ui-state-default')]

        for day_url in day_urls:
            # Iterate over available day urls
            day_response = BeautifulSoup(requests.get(day_url).text, 'html.parser')
            sections = day_response.select('div.tpaper-container section')
            year, month, day = day_url.strip('/').split('/')[-3:]

            for section in sections:
                # Iterate over sections (news categories)
                section_header = section.select('div.section-header a.section-list-heading')[0].text.strip()
                news_urls = [link.get('href') for link in section.select('div.section-container a')]

                for news_url in news_urls:
                    # Iterate over news urls and scrape the text inside them
                    news_response = BeautifulSoup(requests.get(news_url).text, 'html.parser')
                    news_title = news_response.select('div h1.title')[0].text.strip()
                    news_content = '\n'.join([p.text.strip() for p in news_response.select('div.article div div p')])
                    print(f'Scraping: {news_title} - {section_header} - {year}.{month}.{day}')

                    df = df.append({
                        'title': news_title,
                        'content': news_content,
                        'category': section_header,
                        'year': year,
                        'month': month,
                        'day': day
                    }, ignore_index=True)

    df.to_csv(settings.DATA / 'news.csv', index=False)
