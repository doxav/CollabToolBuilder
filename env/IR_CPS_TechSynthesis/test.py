import logging
import random
import time
import requests
from bs4 import BeautifulSoup
from http_request_randomizer.requests.proxy.requestProxy import RequestProxy

import json

import os
import ssl

class DefaultSelf:
    def __init__(self):
        self.google_api_key = 'AIzaSyCgxffd1LhS-TcpdDkWeFfmoIcOyllxm2Y'
        self.google_cse_id = 'f2360a04975674cbb'

    # A function which can web search either via Google API, either via Google scraping
    def search_google(self, query, output_format='json', pdf_only=False, max_results=20, try_direct_scraping_first=False, specific_site=None, result_pretty=True, use_proxy=False, try_selenium_if_1st_scraping_failed=False):
        """Search for patents on Google website.

        Args:
            query (str): Search query.
            output_format (str): Output format. Can be 'json' or 'html'.
            max_results (int): Maximum number of results to return.
            scraping (bool): If True, web scraping is used. Otherwise, Google API is used.

        Returns:
            list: List of patents.
        """
        if try_direct_scraping_first:
            # Google scraping, pdf type is optional
            url = 'https://www.google.com/search'

            if use_proxy:
                # pip install fake-useragent
                from fake_useragent import UserAgent
                ua = UserAgent()

                # get a random proxy through this command 'https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=us,fr&ssl=all&anonymity=all'
                proxies = requests.get('https://api.proxyscrape.com/v2/?request=displayproxies&protocol=https&timeout=5000&country=all&ssl=yes&anonymity=elite').text.split('\r\n')
            else:
                # Default headers for Google scraping
                #headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' }
                from fake_useragent import UserAgent
                ua = UserAgent()
                headers = { 'User-Agent': ua.edge }

            params = { 
                'q': ('site:' + specific_site + ' ' if specific_site else '') + query, 
                'num': max_results, 
                'fileType': 'pdf' if pdf_only else None
                }
            
            with requests.Session() as s:
                if use_proxy:
                    # Try a proxy until s.get() works
                    while True and len(proxies)>0:
                        # pick a random proxy from the list using random in the number of proxies
                        random_proxy = random.randint(0, len(proxies)-1)
                        proxy = proxies[random_proxy] # return a string with the format 'https://ip:port'
                        # remove the proxy from the list
                        proxies.pop(random_proxy)
                        print(proxy)
                        s.proxies = {"https": proxy}
                        print(url)
                        try:
                            # set timeout to 1 second
                            headers = { 'User-Agent': ua.edge }
                            r = s.get(url, params=params, headers=headers, timeout=5)
                            break
                        except:
                            print("Proxy error - remaining proxies: " + str(len(proxies)))
                            continue
                    print("SUCCESS Proxy used: " + proxy)
                else:
                    r = s.get(url, params=params, headers=headers)
                # Generate a proxied request
                #r = req_proxy.generate_proxied_request(url, params=params, headers=headers, method="GET")

                # if r HTTP status code is success, get the html content
                if r.status_code == 200:
                    # print complete url query with parameters
                    soup = BeautifulSoup(r.text, 'html.parser')
                else:
                    print(f'Google direct scraping failed with error code: {r.status_code}')
                    if try_selenium_if_1st_scraping_failed:
                        from selenium import webdriver
                        from selenium.webdriver.common.keys import Keys
                        from selenium.webdriver.common.by import By
                        from selenium.webdriver.support.ui import WebDriverWait
                        from selenium.webdriver.support import expected_conditions as EC
                        from fake_useragent import UserAgent
                        import random

                        # Set up selenium options
                        options = webdriver.ChromeOptions()

                        # Use headless mode (no GUI)
                        options.add_argument("--headless")

                        if use_proxy:
                            # Get a proxy
                            proxies = requests.get('https://api.proxyscrape.com/v2/?request=displayproxies&protocol=https&timeout=5000&country=all&ssl=yes&anonymity=elite').text.split('\r\n')
                            proxy = random.choice(proxies)  # Get a random proxy
                            options.add_argument(f"--proxy-server={proxy}")

                        # Create a browser instance
                        driver = webdriver.Chrome(options=options)

                        # URL and parameters for Google scraping
                        url = 'https://www.google.com/search?q=' + ('site:' + specific_site + ' ' if specific_site else '') + query + ('&fileType=pdf' if pdf_only else '') + '&num=' + str(max_results)
                        driver.get(url)

                        # Wait until the search results are loaded
                        try:
                            element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "search")))
                            soup = BeautifulSoup(driver.page_source, 'html.parser')
                        except Exception as e:
                            # print error
                            error = str(e.message) if hasattr(e, 'message') else str(e)
                            print(f'Google SELENIUM scraping failed with error code: {error}')
                            soup = None
                        driver.quit()

            # Get all results
            results = []
            if soup is not None:
                for g in soup.find_all('div', class_='g'):
                    anchors = g.find_all('a')
                    if anchors:
                        link = anchors[0]['href']
                        title = g.find('h3').text
                        spans = g.find_all('span')
                        description = ""
                        for span in spans:
                            # test if span contains <em> element or if span.text is a year date with 4 digits
                            if not span.find('em') and not ((span.text.isdigit() and len(span.text) == 4)):
                                continue
                            if span.text.isdigit() and len(span.text) == 4:
                                description += "[" + span.text + "] "    
                            else:
                                description += span.text + "\n"
                        # remove last \n if exists
                        if description.endswith("\n"):
                            description = description[:-1]
                        results.append({'link': link, 'title': title, 'description': description})
                    
            print('Scraping search results: ' + str(len(results)))

        if results is None or len(results) == 0: # Google API search
            if not (hasattr(self, 'google_api_key') or self.google_api_key) or not (hasattr(self, 'google_cse_id') or self.google_cse_id):
                raise ValueError('Google API key and/or Google CSE ID not found.')
            # Google Search API Parameters (URL: https://www.googleapis.com/customsearch/v1' + params)):
            # - c2coff: Enables/Disables search in both simplified and traditional Chinese. (Default: 0 [Enabled], 1: Disabled)
            # - cr: Restricts results to documents from a specific country, supports boolean operators.
            # - cx: Programmable Search Engine ID for the query.
            # - dateRestrict: Limits results based on date (e.g., d[number], w[number], m[number], y[number]).
            # - exactTerms: Specifies a phrase all result documents must contain.
            # - excludeTerms: Specifies a word/phrase that should not be in any result document.
            # - fileType: Limits results to files of a specific extension.
            # - filter: Enables/Disables duplicate content filter (0: Disabled, 1: Enabled [Default]).
            # - gl: End-user's geolocation (two-letter country code).
            # - googlehost: Deprecated. Use 'gl' instead. Specifies local Google domain (e.g., google.com, google.fr).
            # - highRange: Specifies end value for a range search.
            # - hl: Sets user interface language.
            # - hq: Appends specified query terms with an AND operator.
            # - imgColorType: Returns images of specific color types (e.g., "color", "gray", "mono", "trans").
            # - imgDominantColor: Returns images of specific dominant color (e.g., "blue", "green").
            # - imgSize: Returns images of specified size (e.g., "medium", "large").
            # - imgType: Returns images of a type (e.g., "clipart", "photo").
            # - linkSite: Requires all results to link to a specific URL.
            # - lowRange: Specifies start value for a range search.
            # - lr: Limits search to documents in a specific language (e.g., "lang_en" for English).
            # - num: Number of search results to return (1-10).
            # - orTerms: Provides additional search terms; each result must contain at least one.
            # - q: Query string.
            # - relatedSite: Requires all results to be pages related to a specific URL.
            # - rights: Filters based on licenses (e.g., cc_publicdomain, cc_attribute).
            # - safe: Search safety level ("active" for SafeSearch on, "off" [Default] for off).
            # - searchType: Specifies search type (e.g., "image" for image search).
            # - siteSearch: Specifies a site to always include/exclude from results.
            # - siteSearchFilter: Whether to include or exclude results from site in 'siteSearch' ("e": Exclude, "i": Include).
            # - sort: Sort expression for results (e.g., sort=date).
            # - start: Index of the first result to return (max of start + num <= 100, max num = 10).

            url = 'https://www.googleapis.com/customsearch/v1'
            current_results_count, results = 0, []
            while current_results_count < max_results:
                num_results = min(10, max_results-current_results_count)
                params = {
                    'key':  self.google_api_key,
                    'cx': self.google_cse_id,
                    'q': query,
                    'num': num_results,
                    'siteSearch': specific_site if specific_site else None,
                    'fileType': 'pdf' if pdf_only else None,
                    'start': current_results_count+1,
                }
                r = requests.get(url, params=params)

                if output_format == 'json':
                    items = r.json().get('items', [])
                    # filter html results to get only title, link and snippet for each result
                    if result_pretty:
                        items = [{'title': item['title'], 'link': item['link'], 'description': item['snippet']} for item in items]
                    results.extend(items)
                elif output_format == 'html':
                    results.append(r.text)
                else:
                    raise ValueError('Invalid output format.')
                current_results_count += num_results

            print('Google API search results: ' + str(len(results)))

        return results


    def search_wikipedia(self, query, output_format='json', max_results=10):
        base_url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': max_results
        }
        response = requests.get(base_url, params=params)
        search_results = response.json().get('query', {}).get('search', [])
        
        if output_format == 'json':
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    'title': result.get('title', ''),
                    'link': 'https://en.wikipedia.org/wiki/' + result.get('title', '').replace(' ', '_'),
                    'description': result.get('snippet', '')
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
        else:
            return {'error': 'Output format not supported'}

    def search_pubmed(self, query, output_format="json", max_results=10, pdf_only=False, scraping=True):
        if pdf_only:
            print("Pubmed does not support pdf only search")
        return self.search_google(query=query, output_format=output_format, pdf_only=False, max_results=max_results, try_direct_scraping_first=scraping, specific_site="pubmed.ncbi.nlm.nih.gov", result_pretty=True)

    def search_core(self, query, output_format='json', max_results=10, pdf_only=False, scraping=True):
        return self.search_google(query=query, output_format=output_format, pdf_only=pdf_only, max_results=max_results, try_direct_scraping_first=scraping, specific_site="core.ac.uk", result_pretty=True)

    def search_uspto(self, query, output_format='json', max_results=10, pdf_only=False, scraping=True):
        return self.search_google(query=query, output_format=output_format, pdf_only=pdf_only, max_results=max_results, try_direct_scraping_first=scraping, specific_site="uspto.gov", result_pretty=True)

    def search_semantic_scholar(self, query, output_format='json', max_results=10):
        base_url = 'https://api.semanticscholar.org/graph/v1/paper/search'
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'paperId,title,abstract'
        }
        
        response = requests.get(base_url, params=params)
        response_data = response.json()
        
        # Process response data
        results = []
        for paper in response_data.get('data', []):
            paper_id = paper.get('paperId', '')
            title = paper.get('title', '')
            # Construct the URL using paperId
            link = f'https://www.semanticscholar.org/paper/{paper_id}'
            description = paper.get('abstract', '')  # Assuming abstract can be treated as a description
            
            results.append({ 'title': title, 'link': link, 'description': description})
        
        if output_format == 'json':
            return results
        else:
            return {'error': 'Output format not supported'}

        
    def search_arxiv(self, query, output_format='json', max_results=10):
        """
        Search arXiv for articles relating to `query`.
        Returns a list of dictionaries containing article information.
        """

        arxiv_url = 'http://export.arxiv.org/api/query?search_query={}&start=0&max_results={}&sortBy=relevance&sortOrder=descending'.format(
            query, max_results)

        response = requests.get(arxiv_url)

        if response.status_code != 200:
            return []

        feed = response.text

        if output_format == 'json':
            import feedparser
            feed = feedparser.parse(feed)
            
            # Parse and transform the results
            results = []
            for entry in feed['entries']:
                pdf_link = [link['href'] for link in entry.get('links', []) if link.get('type') == 'application/pdf']
                results.append({
                    'title': entry.get('title', ''),
                    'link': pdf_link[0] if pdf_link else '',
                    'description': entry.get('summary', '')
                })
            return results

        else:
            return feed

    def search_epo(self, query, output_format='json', max_results=20, use_EPO_API=False):
        """
        Search for patents in EPO database
        :param query: query string
        :param output_format: output format (json or xml)
        :param max_results: maximum number of results returned
        :return: response
        """
        if not use_EPO_API:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from bs4 import BeautifulSoup
            import requests
            import random

            # Set up selenium options
            options = webdriver.ChromeOptions()

            # add header to avoid bot detection
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ")

            # Use headless mode (no GUI)
            options.add_argument("--headless")

            # Create a browser instance
            driver = webdriver.Chrome(options=options)

            # URL for EPO scraping
            # encode query string
            url = 'https://worldwide.espacenet.com/patent/search?q=' + query.replace(' ', '%20')
            print(url)
            driver.get(url)

            # Wait until the search results are loaded
            try:
                #element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
                element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "item--wSceB4di")))
                # sleep for 1 second
                time.sleep(1)
                #WebDriverWait(driver, 10).until(lambda d: d.execute_script('return document.readyState') == 'complete')
                soup = BeautifulSoup(driver.page_source, 'html.parser')
            except Exception as e:
                # print error
                error = str(e.message) if hasattr(e, 'message') else str(e)
                print(f'EPO SELENIUM scraping failed with error code: {error}')
                soup = None
            driver.quit()

            # Get all results
            results = []
            if soup is not None:
                # Find the container div for the results first
                container_div = soup.find('div', class_='publications-list--9wu4rcWN')

                # If container_div is found, extract individual results
                if container_div:
                    for article in container_div.find_all('article'):
                        try:
                            title_tag = article.find('span', class_='item__content--title--dYTuyzV6')
                            if title_tag:
                                title = title_tag.text.strip()
                            else:
                                title = "N/A"

                            patent_num_tag = article.find('div', class_='item__content--subtitle--mFxM6gqw')
                            if patent_num_tag:
                                patent_num = patent_num_tag.find('span').text
                                link = f"https://worldwide.espacenet.com/patent/search?q=pn%3D{patent_num}"
                            else:
                                link = "N/A"

                            description_tag = article.find('div', class_='copy-text--uk738M73')
                            if description_tag:
                                description = description_tag.text.strip()
                            else:
                                description = "N/A"

                            results.append({'title': title, 'link': link, 'description': description})

                        except Exception as e:
                            print(f"Error while extracting data for one article: {e}")
                else:
                    print("Couldn't find the results container.")


            if output_format == 'json':
                return json.dumps(results)
            else:
                return '\n'.join(['{} - {}: {}'.format(result["title"], result["link"], result["description"]) for result in results])
        else:
            raise NotImplementedError("EPO API not implemented yet")
            # TODO: implement EPO API using https://developers.epo.org/ops-v3-2/apis
            url = 'https://worldwide.espacenet.com/3.2/rest-services/search'
            # https://worldwide.espacenet.com/3.2/rest-services/search?lang=en%2Cde%2Cfr&q=cancer%20language%20models%20information&qlang=cql&
            params = {
                'q': query,
                'Range': '1-{}'.format(max_results),
            }
            # set header Authorization Bearer 0p8ydddwdAiGBJERyBAMGVCYJAtw
            
            response = requests.get(url, params=params)
            if output_format == 'json':
                import xmltodict
                response = xmltodict.parse(response.text)
                return response
            else:
                return response.text


TestDoc = DefaultSelf()
# EPO
print(TestDoc.search_epo('Cancer language models', max_results=8, output_format='json'))

# Scrapping Google
# print(TestDoc.search_google('Large language models agents solving complex problems', output_format='json', max_results=10, specific_site='arxiv.org', pdf_only=False, try_direct_scraping_first=True, try_selenium_if_1st_scraping_failed=True))

# Google API
# print(TestDoc.search_google('Liquid networks', output_format='json', max_results=10, specific_site='arxiv.org', pdf_only=False, scraping=False))
# More than 10 test
# print(TestDoc.search_google('Liquid networks', output_format='json', max_results=23, specific_site='arxiv.org', pdf_only=False, scraping=False))

# Arxiv
#print(TestDoc.search_arxiv('Large language models agents solving complex problems', output_format='json', max_results=10))

# PubMed
# Basic with scraping
# print(TestDoc.search_pubmed('Cancer language models', max_results=8, pdf_only=False))
# No scraping
# print(TestDoc.search_pubmed('Cancer language models', max_results=8, pdf_only=False, scraping=False))
# PDF
# print(TestDoc.search_pubmed('Cancer language models', max_results=8, pdf_only=True, scraping=False))

# CORE
# Basic with scraping
#print(TestDoc.search_core('Cancer language models', max_results=8, pdf_only=False))
# No scraping
# print(TestDoc.search_core('Cancer language models', max_results=8, pdf_only=False, scraping=False))

# Wikipedia
# print(TestDoc.search_wikipedia('language models', max_results=8))

# USPTO - NOT WORKING PROPERLY
# No scraping
# print(TestDoc.search_uspto('Cancer language models', max_results=8, pdf_only=False, scraping=False))
# PDF
# print(TestDoc.search_uspto('Motorcycle', max_results=8, pdf_only=True, scraping=False))

# Semantic Scholar - Could be even enriched
# print(TestDoc.search_semantic_scholar('Cancer language models', max_results=8))
# High limit of papers !
#print(TestDoc.search_semantic_scholar('Cancer language models', max_results=80))

