import datetime
import os.path
#import warnings
import copy
import time
import uuid
from dataclasses import dataclass, field
import re
from typing import List, SupportsFloat, Any, Tuple, Dict
from dataclasses import asdict
from langchain_core.messages import SystemMessage, HumanMessage
import json
import inspect, os
from typing import Any, Dict, Callable, List
from langchain_community.tools import ShellTool, tool
from pydantic import BaseModel
import io
shell = ShellTool()

from config import *
#from attr import dataclass, field
#import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pdb

import traceback

import json
import requests
import hashlib
from datasets import load_dataset
import contextlib

import argparse
import ast
import json
import os
import subprocess
import uuid

import pandas as pd
from tqdm import tqdm

from learn import Environment

# import a function from langchain which could embed a text into a vector using OpenAI ada-002 or HuggingFace
import langchain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from utils.file_utils import save_to_pickle, load_from_pickle
from utils.llm_utils import UnifiedVectorDB
#from langchain_community.cache import InMemoryCache, SQLiteCache
#langchain.llm_cache = SQLiteCache(database_path="sqlite/langchain_cache.db")

from bs4 import BeautifulSoup

##############################################################################################################
# Placeholder classes for the technical synthesis environment
##############################################################################################################

def method_call_counter(method):
    def wrapper(*args, **kwargs):
        if args and hasattr(args[0], '__dict__'):
            self = args[0]
            if not hasattr(self, '_method_counts'):
                self._method_counts = {}
            self._method_counts[method.__name__] = self._method_counts.get(method.__name__, 0) + 1
        return method(*args, **kwargs)
    return wrapper

@dataclass
class Section:
    section_id: int
    parent_id: int = 0
    title: str = ""
    title_embedding: List[float] = field(default_factory=list)
    content: str = ""
    content_embedding: List[float] = field(default_factory=list)
    title_validation_status: int = 0
    content_progress_validation_status: int = 0
    local_feedback_to_process: List[str] = field(default_factory=list) # could be citation to integrate, critics to process...
    local_feedback_processed: List[str] = field(default_factory=list) # same as above

@dataclass
class Document:
    title: str = "" # e.g. should be the title of the document if the goal is to write a SOTA survey paper, the title of a Wikipedia article if the goal is to write a Wikipedia article, the title of a patent if the goal is to write a patent, ...
    title_embedding: List[float] = field(default_factory=list) # TODO: check if we need to store the embedding of the title because, differently to sections, it is not used in comparison to target because it is an input
    context: str = "" # e.g. should be the abstract content of the document if the goal is to write a SOTA survey paper, the introduction of a Wikipedia article if the goal is to write a Wikipedia article, the introduction of a patent if the goal is to write a patent, ...
    context_embedding: List[float] = field(default_factory=list)  # TODO: check if we need to store the embedding of the context because, differently to sections content, it is not used in comparison to target because it is an input
    sections_list: List[Any] = field(default_factory=list)  
    sections_list_embedding: List[float] = field(default_factory=list)
    embedding_model_name: str = "intfloat/e5-base-v2" # e.g. "nomic-embed-text:latest" for OpenAI ada-002, "intfloat/e5-base-v2" for HuggingFace e5-base-v2, ...

class DocumentStructure:
    embedding_model_cls = None
    def __init__(self,
                 synthesis_type: str,
                 initial_goal: str,
                 refined_goals: List[str] = None,
                 embedding_model_name: str = "intfloat/e5-base-v2",  # nomic-embed-text:latest, intfloat/e5-base-v2
                 embedding_model_query_prefix: str = '',
                 # e.g. "query: " for intfloat/e5-base-v2 should improve for QA but we are in estimating straight semantic similarity
                 title: str = None,
                 context: str = None,
                 abstract: str = None,
                 ):
        if abstract is not None: context = abstract if context is None else context + "\n" + abstract
        self.abstract = abstract
        self.embedding_model_query_prefix = embedding_model_query_prefix
        self.embedding_model_name = embedding_model_name
        if not DocumentStructure.embedding_model_cls:
            if embedding_model_name == "text-embedding-ada-002":
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError("OpenAI API key is required for OpenAI ada-002 model.")
                self.embedding_model = OpenAIEmbeddings(
                    model=embedding_model_name)  # , openAIApiKey=os.getenv("OPENAI_API_KEY")
            else:
                self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                                             encode_kwargs={"normalize_embeddings": True},
                                                             model_kwargs={
                                                                 "trust_remote_code": True})  # , openAIApiKey=os.getenv("OPENAI_API_KEY"
            DocumentStructure.embedding_model_cls = self.embedding_model
        else:
            self.embedding_model = DocumentStructure.embedding_model_cls

        self.synthesis_type = synthesis_type
        self.initial_goal = initial_goal
        self.refined_goals = [initial_goal] if refined_goals is None else refined_goals
        self.document_content = Document()  # self.document_content = copy.deepcopy(DOCUMENT_SCHEMA)
        # self.document_content.sections_list = []
        self.title = title
        if title and title != "":
            self.set_plan_field_with_embedding('title', title)
        self.context = context
        if context and context != "":
            self.set_plan_field_with_embedding('context', context)
        self.dumb_embedding = self.embedding_model.embed_query(".")  # Used to compute min_cosine_similarity
        self.embedding_size = len(self.dumb_embedding)

        self.global_feedback_to_process = []
        self.global_feedback_processed = []
        self.resources = []
        def filename_friendly_collection_string(s):
            # Constraint 1: Truncate or pad the string to ensure it's between 3-63 characters
            s = s[:63].ljust(3, 'a')
            # Constraint 2: Ensure it starts and ends with an alphanumeric character
            if not s[0].isalnum():
                s = 'a' + s[1:]
            if not s[-1].isalnum():
                s = s[:-1] + 'a'
            # Constraint 3: Replace invalid characters with underscores
            s = re.sub(r'[^a-zA-Z0-9_-]', '_', s)
            # Constraint 4: Replace two consecutive periods with underscores
            s = s.replace('..', '__')
            # Constraint 5: Ensure it's not a valid IPv4 address
            if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', s):
                s = 'a' + s[1:]
            return s
        friendly_collection = filename_friendly_collection_string(self.synthesis_type)+"__"+filename_friendly_collection_string(self.title)
        self.resources_vectordb = UnifiedVectorDB(
            collection_name=friendly_collection[1:63],
            embedding_function=self.embedding_model,
            persist_directory=f"ckpt/doc/{friendly_collection}",
        )
        self.events = []

        self.get_state(save=True)

    def get_embedding(self, text: str) -> List[float]:
        # if not text:  # or if text == ""
        #     return np.zeros(self.embedding_size)  # Assuming you know the size of your embeddings
        # else:
        return self.embedding_model.embed_query(self.embedding_model_query_prefix + text)

    def update_sections_embeddings(self, section_ids: List[int] = None, force_update: bool = False, batch_update: bool = False):
    # compute and set embeddings of any empty content_embedding or title_embedding when not set, and if content or title is not empty (nor None, nor '')
    # if section_id is provided, only update the section with this id
        if batch_update:
            texts_to_embed = []
            sections_to_update = []
            sections_label_to_update = []

        # Collect texts that need to be updated
        for section in self.document_content.sections_list:
            if section_ids is not None and section.section_id not in section_ids:
                continue
            if (not section.content_embedding and section.content) or force_update:
                if batch_update:
                    texts_to_embed.append(section.content)
                    sections_to_update.append(section)
                    sections_label_to_update.append('content_embedding')
                else:
                    section.content_embedding = self.get_embedding(section.content)
            if (not section.title_embedding and section.title) or force_update:
                if batch_update:
                    texts_to_embed.append(section.title)
                    sections_to_update.append(section)
                    sections_label_to_update.append('title_embedding')
                else:
                    section.title_embedding = self.get_embedding(section.title)
        
        if batch_update:
            # Get embeddings in one batch call
            embeddings = self.get_embedding(texts_to_embed)
            
            for section, emb, label in zip(sections_to_update, embeddings, sections_label_to_update):
                setattr(section, label, emb)

        self.update_plan_embedding()

    def update_plan_embedding(self):
        """ update plan embedding by computing mean of all section embeddings and the title embedding (if any) of the synthesis plan """
        title_embeddings = []  # List to hold title embeddings
        content_embeddings = []  # List to hold content embeddings
        # get mode embeddings length 
        
        try:
            # Populate the title_embeddings and content_embeddings lists
            # x.title_embedding should be added only if x.title_embedding is not empty (nor None, nor '')
            title_embeddings = [x.title_embedding for x in self.document_content.sections_list if x.title_embedding]
            content_embeddings = [x.content_embedding for x in self.document_content.sections_list if x.content_embedding]

            # convert self.dumb_embedding to a list to be able to use it in np.mean
            # Compute mean embedding for titles
            title_mean = np.mean(title_embeddings, axis=0).tolist() if title_embeddings else self.dumb_embedding
            # if content_embeddings is empty, set content_mean to 0
            content_mean = np.mean(content_embeddings, axis=0).tolist() if content_embeddings else self.dumb_embedding
            
            # Combine title and content embeddings and compute their mean
            all_embeddings = title_embeddings + content_embeddings
            total_mean = np.mean(all_embeddings, axis=0).tolist() if all_embeddings else self.dumb_embedding

        except KeyError:
            raise KeyError("Could not find title_embedding or content_embedding in every section. Please check that every section has both of these keys.")
        
        self.document_content.sections_list_title_embedding = title_mean  # Store the mean title embedding
        self.document_content.sections_list_content_embedding = content_mean  # Store the mean content embedding
        self.document_content.sections_list_embedding = total_mean  # Store the combined mean embedding

    def set_plan_field_with_embedding(self, field: str, value: str, event: str = None, section_id: int = None):
        # can be chained with other set_plan_field calls (e.g. set_plan_field('title', 'my title').set_plan_field('context', 'my context').set_plan_field('title', 'my new title', section_id=1))
        embedding = self.get_embedding(value)

        if section_id:
            target = next((s for s in self.document_content.sections_list if s.section_id == section_id), None)
            if not target:
                raise ValueError(f"Section with id {section_id} not found.")
            setattr(target, field, value)  # Using setattr because target is a class instance
            setattr(target, f"{field}_embedding", embedding)
        else:
            setattr(self.document_content, field, value)  # because self.document_content is a class instance
            setattr(self.document_content, f"{field}_embedding", embedding)

        if event:
            self.events.append(('observe', {'action': event, 'section_id': section_id}))
        return self

    def get_state(self, save: bool=False) -> Dict[str, Any]:
        state = {
            'synthesis_type': self.synthesis_type,
            'initial_goal': self.initial_goal,
            'refined_goals': self.refined_goals,
            'document_content': self.document_content,
            'global_feedback_to_process': self.global_feedback_to_process,
            'global_feedback_processed': self.global_feedback_processed,
            'resources': self.resources,
            'events': self.events,
            'embedding_model_name': self.embedding_model_name,
        }
        if save:
            self.last_state = copy.deepcopy(state)
        return state

    def restore_state(self, state: Dict[str, Any]=None):
        """ if state is empty, restore last saved state, else restore the provided state"""
        if state is None:
            state = self.last_state
        self.synthesis_type = state['synthesis_type']
        self.initial_goal = state['initial_goal']
        self.refined_goals = state['refined_goals']
        self.document_content = state['document_content']
        self.global_feedback_to_process = state['global_feedback_to_process']
        self.global_feedback_processed = state['global_feedback_processed']
        self.resources = state['resources']
        self.events = state['events']
        return self.get_state()
      
    def reset(self):
    # TODO: check if refined_goals should be reset or not
        # this code might be redundant but ensure sections_list embeddings memory are cleared (might be removed later)
        for section in self.document_content.sections_list:
            section.content_embedding = []
            section.title_embedding = []
        self.document_content.sections_list_embedding = []
        self.document_content = Document() # self.document_content = copy.deepcopy(DOCUMENT_SCHEMA)

        self.resources.clear()
        self.events.clear()
        self.global_feedback_to_process.clear()
        self.global_feedback_processed.clear()
        self.get_state(save=True)
        return self

    def add_event(self, event='observe', status: Dict[str, Any]=None):
        return self.events.append((event, status))
    
    def get_events(self):
        return self.events

class SynthesisManager:
    def __init__(self, document: DocumentStructure, target_file_path: str = None):
        self.document = document
        self.title = self.document.title
        self.abstract = self.document.context
        self.min_cosine_similarity = cosine_similarity([self.document.embedding_model.embed_query(".")], [self.document.embedding_model.embed_query("If you can keep your head when all about you are losing theirs and blaming it on you, If you can trust yourself when all men doubt you, But make allowance for their doubting too ; If you can wait and not be tired by waiting, Or being lied about, don’t deal in lies, Or being hated, don’t give way to hating, And yet don’t look too good, nor talk too wise")])[0][0]
        if target_file_path:
            self.target_file_path = target_file_path

    @staticmethod
    @method_call_counter
    def validate_section_format(section: Dict[str, Any]) -> bool:
        try:
            # This will try to instantiate a Section. If there's a problem with the data, an exception will be raised (e.g., a type error).
            Section(**section)
            return True
        except TypeError as e:
            # print(e)
            return False

    @method_call_counter
    def chat(self, message: str):
     print(f"SynthesisManager Chat message:")

    @staticmethod
    @method_call_counter
    @load_from_pickle
    @save_to_pickle
    def search_generic(query, search_type, output_format='json', max_results=10):
            """
            Searches for resources based on the given query and search type.

            Args:
                query (str): The search query.
                search_type (str): The type of search to perform. Supported types are:
                    'patents_google', 'patent_epo', 'patent_uspto', 'paper_arxiv', 'paper_pubmed', 'websearch_google', 'core', 'websearch_wikipedia', 'paper_semantic_scholar'.
                output_format (str, optional): The format of the search results. Defaults to 'json'.
                max_results (int, optional): The maximum number of results to return. Defaults to 10.

            Returns:
                list: The search results. If an error occurs, a dictionary with an 'error' key is returned.
            """
            search_functions = { # ['paper_arxiv','paper_pubmed','websearch_google','websearch_wikipedia','paper_semantic_scholar']
                'patents_google': SynthesisManager.search_google_patents, #
                'patent_epo': SynthesisManager.search_epo, #
                'patent_uspto': SynthesisManager.search_uspto, #
                'paper_arxiv': SynthesisManager.search_arxiv, #
                'paper_pubmed': SynthesisManager.search_pubmed,
                #'websearch_bing': SynthesisManager.search_bing,
                'websearch_google': SynthesisManager.search_google, #
                'core': SynthesisManager.search_core, #
                'websearch_wikipedia': SynthesisManager.search_wikipedia, #
                'paper_semantic_scholar': SynthesisManager.search_semantic_scholar, #
                'fiche_anomalie': SynthesisManager.search_fiche_anomalie #
            }
            search_function = search_functions.get(search_type)
            if search_function:
                results = search_function(query, output_format=output_format, max_results=max_results)
                return results
            else:
                raise ValueError(f"Unsupported search type: {search_type}")

    # OK, basic implementation via scraping (short description and link to paper which should be scraped also to get full description) - waiting for a key for a proper implementation
    @staticmethod
    @method_call_counter
    def search_epo(query, output_format='json', max_results=20, use_EPO_API=False):
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
            # print(url)
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
        
    # OK, based on Arxiv API 
    @staticmethod
    @method_call_counter
    def search_arxiv(query, output_format='json', max_results=10):
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

    # OK, based on semantic scholar API
    # Could be enriched: https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_get_paper_search
    @staticmethod
    @method_call_counter
    def search_semantic_scholar(query, output_format='json', max_results=10):
        #  """
        #  Function currently used basic title, URL from semantic scholar, abstract/description but could be enriched with other graph informations
         
        # Semantic Scholar API Parameters:

        # # General Paper Parameters:
        # - paperId: Unique paper identifier (string).
        # - corpusId: Secondary unique paper identifier (numeric).
        # - url: URL on Semantic Scholar.
        # - title: Paper title. Included if no fields specified.
        # - venue: Normalized venue name.
        # - publicationVenue: Publication venue meta-data.
        # - year: Publication year.
        # - authors: Up to 500 returned, includes authorId & name.
        # - externalIds: IDs from sources like ArXiv, MAG, ACL, PubMed, etc.
        # - abstract: Paper's abstract (might be missing due to legal reasons).
        # - referenceCount: Number of papers referenced.
        # - citationCount: Number of citations found for paper.
        # - influentialCitationCount: See documentation.
        # - isOpenAccess: See documentation.
        # - openAccessPdf: Link to paper if open access.
        # - fieldsOfStudy: High-level academic categories from external sources.
        # - s2FieldsOfStudy: Academic categories, sourced externally or from internal classifier.
        # - publicationTypes: E.g., Journal Article, Conference, Review.
        # - publicationDate: Format YYYY-MM-DD.
        # - journal: Journal name, volume, and pages.
        # - citationStyles: Bibliographical citation (e.g., BibTeX).

        # # Author Parameters:
        # - authors: Up to 500 returned.
        # - authorId: Unique S2 ID.
        # - externalIds: IDs like ORCID/DBLP.
        # - url: URL on Semantic Scholar.
        # - name: Author's name.
        # - aliases: Other names used by author (caution: might contain deadnames).
        # - affiliations: Author's affiliations.
        # - homepage: Author's homepage.
        # - paperCount: Total publication count.
        # - citationCount: Author's total citation count.
        # - hIndex: See S2 FAQ on h-index.

        # # Additional Endpoints:
        # - /author/{author_id}/papers: Detailed info about an author's papers.
        # - /paper/{paper_id}/authors: Detailed info about a paper's authors.
        # - /paper/{paper_id}/citations: Detailed info about a paper's citations.
        # - /paper/{paper_id}/references: Detailed info about a paper's references.
        # """
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

    # OK: based on search_google
    @staticmethod
    @method_call_counter
    def search_google_patents(query, output_format="json", max_results=10, pdf_only=False, scraping=True):
        if pdf_only:
            print("Google patents does not support pdf only search")
        return SynthesisManager.search_google(query=query, output_format=output_format, pdf_only=False, max_results=max_results, try_direct_scraping_first=scraping, specific_site="patents.google.com", result_pretty=True)
    
    # NOK: based on search_google but information are not well formatted and poor
    @staticmethod
    @method_call_counter
    def search_uspto(query, output_format='json', max_results=10, pdf_only=False, scraping=True):
        return SynthesisManager.search_google(query=query, output_format=output_format, pdf_only=pdf_only, max_results=max_results, try_direct_scraping_first=scraping, specific_site="uspto.gov", result_pretty=True)

    # A function which can web search either via Google API, either via Google scraping
    @staticmethod
    @method_call_counter
    @load_from_pickle
    @save_to_pickle
    def search_google(query, output_format='json', pdf_only=False, max_results=20, try_direct_scraping_first=True, specific_site=None, result_pretty=True, use_proxy=False, try_selenium_if_1st_scraping_failed=True):
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
            # Check google_api_key and google_cse_id are set as a global variable in the script
            from config import google_api_key, google_cse_id
            if not google_api_key or not google_cse_id:
                raise ValueError('Please set google_api_key and google_cse_id as global variables in the script.')

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
                    'key':  google_api_key,
                    'cx': google_cse_id,
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

    @staticmethod
    @method_call_counter
    def extract_text_from_pdf(pdf_path):
        """
        Extract text from a PDF file
        :param pdf_path: path to the PDF file
        :return: extracted text as a string
        """
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text

    @staticmethod
    @method_call_counter
    def search_fiche_anomalie(query, directory_path, output_format='json', max_results=20):
        """
        Search for anomalies in PDF files within a directory
        :param query: query string
        :param directory_path: path to the directory containing PDF files
        :param output_format: output format (json or text)
        :param max_results: maximum number of results returned
        :return: response
        """
        
        results = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(directory_path, filename)
                text = extract_text_from_pdf(pdf_path)
                if query.lower() in text.lower():
                    results.append({
                        'filename': filename,
                        'content': text
                    })
                if len(results) >= max_results:
                    break
        
        if output_format == 'json':
            return json.dumps(results, indent=2)
        else:
            return '\n'.join(['Filename: {}\nContent: {}\n'.format(result['filename'], result['content']) for result in results])

    # OK: based on search_google
    @staticmethod
    @method_call_counter
    def search_pubmed(query, output_format="json", max_results=10, pdf_only=False, scraping=True):
        if pdf_only:
            print("Pubmed does not support pdf only search")
        return SynthesisManager.search_google(query=query, output_format=output_format, pdf_only=False, max_results=max_results, try_direct_scraping_first=scraping, specific_site="pubmed.ncbi.nlm.nih.gov", result_pretty=True)

    # OK: based on search_google
    @staticmethod
    @method_call_counter
    def search_core(query, output_format='json', max_results=10, pdf_only=False, scraping=True):
        return SynthesisManager.search_google(query=query, output_format=output_format, pdf_only=pdf_only, max_results=max_results, try_direct_scraping_first=scraping, specific_site="core.ac.uk", result_pretty=True)

    # OK: based on core Wikipedia search
    @staticmethod
    @method_call_counter
    def search_wikipedia(query, output_format='json', max_results=10):
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

    # add event using the document object add_event method add_event
    @method_call_counter
    def add_event(self, event: str, data: Dict[str, Any]=None):
        self.document.add_event(event, data)

    def normalized_cosine_similarity(self, a: List[float], b: List[float], min_cs: float = None) -> float:
        if min_cs is None:
            min_cs = self.min_cosine_similarity
        return (cosine_similarity([a], [b])[0][0] - min_cs) / (1 - min_cs)

    @method_call_counter
    def add_section(self, section: Section):
        if self.validate_section_format(asdict(section)):  # Convert dataclass to dict for validation
            self.document.document_content.sections_list.append(section)
            self.document.update_sections_embeddings([section.section_id])
            self.document.add_event({'action': 'add_section', 'section_id': section.section_id})
        else:
            print('Invalid section format.')
        return self
    
    @method_call_counter
    def create_and_add_section_then_return_id(self, title: str, content: str, section_id: int = None, parent_id: int = None) -> int:
        if not section_id:
            # Generate section_id by using max section_id + 1
            section_id = (max([s.section_id for s in self.document.document_content.sections_list]) + 1) if len(self.document.document_content.sections_list) > 0 else 1

        self.add_section(Section(section_id=section_id, parent_id=parent_id, title=title, content=content))
        return section_id

    @method_call_counter
    def get_sections(self, ids: List[int]) -> List[Section]:
        return [s for s in self.document.document_content.sections_list if s.section_id in ids]
    
    @method_call_counter
    def get_all_sections(self) -> List[Section]:
        return self.document.document_content.sections_list

    @method_call_counter
    def remove_section(self, section_id: int) -> bool:
        section = next((s for s in self.document.document_content.sections_list if s.section_id == section_id), None)
        if section:
            self.document.document_content.sections_list = [s for s in self.document.document_content.sections_list if s.section_id != section_id]
            self.document.update_plan_embedding()
            self.document.add_event({'action': 'remove_section','section_id': section_id})
            return True
        else:
            return False

    @method_call_counter
    def edit_section(self, section_id: int, new_content: str = None, new_title: str = None, new_parent_id: int = None) -> bool:
        #section = next((s for s in self.document.document_content if s['id'] == section_id), None)
        section = next((s for s in self.document.document_content.sections_list if s.section_id == section_id), None)
        if section:
            action_event = {'action': 'edit_section','section_id': section_id}
            update_embeddings = False
            if new_content:
                section.content = new_content
                update_embeddings = True
                action_event['new_content'] = new_content
            if new_title:
                section.title = new_title
                update_embeddings = True
                action_event['new_title'] = new_title
            if new_parent_id:
                section.parent_id = new_parent_id
                action_event['new_parent_id'] = new_parent_id
            self.document.add_event('observation', action_event)
            if update_embeddings:
                self.document.update_sections_embeddings([section_id])
            return True
        else:
            return False
    
    @method_call_counter
    def swap_sections(self, section_id_1: int, section_id_2: int) -> bool:
        section_1 = next((s for s in self.document.document_content.sections_list if s.section_id == section_id_1), None)
        section_2 = next((s for s in self.document.document_content.sections_list if s.section_id == section_id_2), None)
        if section_1 and section_2:
            section_1_index = self.document.document_content.sections_list.index(section_1)
            section_2_index = self.document.document_content.sections_list.index(section_2)
            self.document.document_content.sections_list[section_1_index], self.document.document_content.sections_list[section_2_index] = self.document.document_content.sections_list[section_2_index], self.document.document_content.sections_list[section_1_index]
            self.document.add_event('observation', {'action': 'swap_sections','section_id_1': section_id_1, 'section_id_2': section_id_2})
            return True
        else:
            return False

    # search into resources stored in self.document.resources_vectordb and self.document.resources, return a list of resources
    @method_call_counter
    def semantic_search_resources(self, query_embeddings = None, query_texts = None, n_results = 10, where = None, where_document = None, include = ["metadatas", "documents", "distances"]):
        result = self.document.resources_vectordb.similarity_search_with_score( query_embeddings, k=n_results)

    @method_call_counter
    def get_all_resources(self) -> List[Dict[str, Any]]:
        return self.document.resources

    @method_call_counter
    def add_or_update_results_in_resources(self, results, metadatas_to_add: dict = {}, store_linked_document_content: bool = False):
        if isinstance(results, dict): results = [results]
        for result in results:
            content = {'description': result['description']} if isinstance(result['description'], str) else result['description']
            self.add_or_update_result_in_resources(metadatas=metadatas_to_add, name=result['title'], link=result['link'], content=content, store_linked_document_content=store_linked_document_content)
        return self

    @method_call_counter
    def add_or_update_result_in_resources(self, metadatas: dict, name: str=None, content: dict = None, link: str = None, store_linked_document_content: bool = False, chaining: bool = True):
        # Move metadatas to content if content data were provided into metadatas
        if metadatas.get('title') and not name:
            name = metadatas.get('title')
            metadatas.pop('title')
        if metadatas.get('link') and not link:
            link = metadatas.get('link')
            metadatas.pop('link')
        if metadatas.get('description') and not content:
            content = {'description': metadatas.get('description')}
            metadatas.pop('description')

        # assert name or link are provided to identify the resource
        if not name and not link:
            raise ValueError("Either name or link must be provided")

        # Generate id using max
        id = max([r['id'] for r in self.document.resources]) + 1 if len(self.document.resources) > 0 else 1
        document = {'name': name, 'link': link, 'content': {'description': content} if isinstance(content, str) else content} # Convert content to dict if it's a string
        
        # Check for existing document
        existing_doc = next((doc for doc in self.document.resources if (doc['document']['name'] == name or (link and doc['document']['link'] == link))), None)
        
        if existing_doc:
            # Update the existing document
            updated_fields = []
            for key, value in document.items():
                if value and existing_doc['document'].get(key) != value:
                    existing_doc['document'][key] = value
                    updated_fields.append(key)
            
            # Log the event
            if updated_fields:
                # Update resources_vectordb
                self.document.resources_vectordb.add_texts([str(document)], metadatas=[metadatas], ids=[str(existing_doc['id'])])
                self.document.add_event('observation', {'action': 'modify_resource', 'document_name': name, 'updated_fields': updated_fields})
                #print(f"Resource updated fields: ", updated_fields)
            #else:
                #self.document.add_event('observation', {'action': 'modify_resource', 'document_name': name, 'message': 'No fields were updated'})
                #print(f"Resource no fields updated")
        else:
            # Add new document
            self.document.resources.append({
                'id': id,
                'metadatas': metadatas,
                'document': document,
            })
            if store_linked_document_content:
                childs_ids_list = self.get_and_store_link_content(link=link, parent_id=id, chaining=False)
                metadatas['childs_ids_list'] = childs_ids_list
            self.document.resources_vectordb.add_texts([str(document)], metadatas=[metadatas], ids=[str(id)])
            self.document.add_event('observation', {'action': 'add_resource', 'document_name': name})
            #print(f"Resource added - VectorDB collection count: {self.document.resources_vectordb.count()}")

        return self if chaining else (existing_doc if existing_doc else self.document.resources[-1])

    @method_call_counter
    def get_and_store_link_content(self, link: str = None, parent_id = None, chaining: bool = True):
            """
            Downloads an online document from the given link and stores it in the resources database.
            
            Args:
            link (str): The URL of the online document to download.
            parent_id: The ID of the parent document, if any.
            chaining (bool): Whether to return the current object or the IDs of the stored documents.
            
            Returns:
            If chaining is True, returns the current object. Otherwise, returns the IDs of the stored documents.
            """
            from langchain.document_loaders import WebBaseLoader
            if link is None:
                raise ValueError("Please provide a link to download the document from")
            loader = WebBaseLoader(link)
            data = loader.load()
            if parent_id is not None:
                for doc in data:
                    doc.metadata.extend([{'parent_id': parent_id}])
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter()
            all_splits = splitter.split_documents(data)
            splits_ids = self.document.resources_vectordb.db.add_documents(all_splits)
            if chaining:
                return self
            else:
                return splits_ids

    @method_call_counter
    def remove_resource(self, resource_id):
        # resource_id can be array or single int
        if isinstance(resource_id, list):
            self.document.resources = [r for r in self.document.resources if r['id'] not in resource_id]
        elif isinstance(resource_id, int):
            self.document.resources = [r for r in self.document.resources if r['id'] != resource_id]
        self.document.add_event('observation', {'action': 'remove_resources','resource_id': str(resource_id)})
        return self
    
    @method_call_counter
    def remove_resources(self, resource_ids: List[int]):
        return self.remove_resource(resource_ids)

    def restore_last_state(self):
        return self.document.restore_state()

    @method_call_counter
    def list_all_previous_document_events(self) -> List[Any]:
        return self.document.events

    def set_targetJSON_comparison(self, file_path: str, target_section_title_embedding_label: str = "section_embedding_2", target_section_content_embedding_label: str = "content_embedding_2", target_plan_embedding_label: str = "plan_embedding_2", normalize_embeddings: bool = True, min_cosine_similarity: float = None):
        self.target_file_path = file_path

        with open(file_path, 'r') as f:
            self.target_data = json.load(f)
        output_check = ''
        for section in self.target_data['plan']:
            output_check += section['section'] + " /"
        print(output_check)
        # Compute the total length for the target data (similar to the test method)
        self.target_total_content_length = sum(len(section['content']) for section in self.target_data['plan'])
        self.target_total_sections_count = len(self.target_data["plan"])

        self.target_plan_titles_embedding = np.mean([section[target_section_title_embedding_label] for section in self.target_data["plan"]], axis=0)
        self.target_plan_contents_embedding = np.mean([section[target_section_content_embedding_label] for section in self.target_data["plan"]], axis=0)
        self.target_plan_embedding = self.target_data[target_plan_embedding_label]

        if normalize_embeddings:
            if min_cosine_similarity is None:
                dumb_embedding = self.document.dumb_embedding
                self.min_plan_titles_cosine_similarity = cosine_similarity([dumb_embedding], [self.target_plan_titles_embedding])[0][0]
                self.min_plan_contents_cosine_similarity = cosine_similarity([dumb_embedding], [self.target_plan_contents_embedding])[0][0]
                self.min_plan_cosine_similarity = cosine_similarity([dumb_embedding], [self.target_plan_embedding])[0][0]
            else:
                self.min_plan_titles_cosine_similarity = self.min_plan_contents_cosine_similarity = self.min_plan_cosine_similarity = min_cosine_similarity
        else:
            self.min_plan_titles_cosine_similarity = self.min_plan_contents_cosine_similarity = self.min_plan_cosine_similarity = 0 

    def get_distance_to_targetJSON(self, target_section_title_embedding_label: str = "section_embedding_2", target_section_content_embedding_label: str = "content_embedding_2", target_plan_embedding_label: str = "plan_embedding_2", get_progress: bool = False):
        # if self does not have target_file_path
        if not hasattr(self, 'target_file_path'):
            raise ValueError("Please set target_file_path using set_targetJSON_comparison method")

        if not hasattr(self, 'target_data'):
            section_embedding_key, content_embedding_key, plan_embedding_key = "content_embedding_2", "section_embedding_2", "plan_embedding_2"
            self.set_targetJSON_comparison(self.target_file_path,
                                           target_section_title_embedding_label=section_embedding_key,
                                           target_section_content_embedding_label=content_embedding_key,
                                           target_plan_embedding_label=plan_embedding_key)
            self.document.update_plan_embedding()
        elif not hasattr(self.document.document_content, 'sections_list_title_embedding'):
            self.document.update_plan_embedding()

        # Compute current document metrics
        current_sections_count = len(self.document.document_content.sections_list)
        current_plan_non_empty_sections_content_count = sum(1 for section in self.document.document_content.sections_list if section.content and len(section.content) > 1)
        current_plan_non_empty_sections_title_count = sum(1 for section in self.document.document_content.sections_list if section.title and len(section.title) > 1)
        current_content_length = sum(len(getattr(section, 'content', '')) for section in self.document.document_content.sections_list)

        # Get embeddings
        plan_embedding = self.document.document_content.sections_list_embedding
        plan_titles_embedding = self.document.document_content.sections_list_title_embedding
        plan_contents_embedding = self.document.document_content.sections_list_content_embedding

        # Compute embedding similarity
        plan_embedding_similarity = self.normalized_cosine_similarity(plan_embedding, self.target_plan_embedding, self.min_plan_cosine_similarity)
        plan_titles_embedding_similarity = self.normalized_cosine_similarity(plan_titles_embedding, self.target_plan_titles_embedding, self.min_plan_titles_cosine_similarity)
        plan_contents_embedding_similarity = self.normalized_cosine_similarity(plan_contents_embedding, self.target_plan_contents_embedding, self.min_plan_contents_cosine_similarity)

        # Refined ratio calculations
        content_length_ratio_to_target = round(
            min(current_content_length / (self.target_total_content_length + 1e-5), 1.5), 4)
        sections_count_ratio_to_target = round(
            min(current_sections_count / (self.target_total_sections_count + 1e-5), 1.5), 4)
        sections_content_non_empty_count_ratio_to_target = round(
            min(current_plan_non_empty_sections_content_count / (self.target_total_sections_count + 1e-5), 1.5), 4)
        sections_title_non_empty_count_ratio_to_target = round(
            min(current_plan_non_empty_sections_title_count / (self.target_total_sections_count + 1e-5), 1.5), 4)

        # Build the result dictionary
        distance_to_targetJSON = {
            "plan_embedding_similarity": round(plan_embedding_similarity, 6),
            "plan_titles_embedding_similarity": round(plan_titles_embedding_similarity, 6),
            "plan_contents_embedding_similarity": round(plan_contents_embedding_similarity, 6),

            "current_sections_count": current_sections_count,
            "sections_count_ratio_to_target": sections_count_ratio_to_target,

            "title_non_empty_count_ratio_to_target": sections_title_non_empty_count_ratio_to_target,

            "current_content_length": current_content_length,
            "content_length_ratio_to_target": content_length_ratio_to_target,

            "content_non_empty_count_ratio_to_target": sections_content_non_empty_count_ratio_to_target,
        }

        # Progress comparison (optional)
        if get_progress:
            def get_ratio(previous_value, current_value):
                return round((previous_value - current_value) / (previous_value + 0.0000001) * 100, 2) if previous_value else 0
            if hasattr(self, 'distance_to_targetJSON'):
                distance_to_targetJSON['plan_embedding_similarity_progress'] = get_ratio(plan_embedding_similarity, self.distance_to_targetJSON['plan_embedding_similarity'])
                distance_to_targetJSON['plan_titles_embedding_similarity_progress'] = get_ratio(plan_titles_embedding_similarity, self.distance_to_targetJSON['plan_titles_embedding_similarity'])
                distance_to_targetJSON['plan_contents_embedding_similarity_progress'] = get_ratio(plan_contents_embedding_similarity, self.distance_to_targetJSON['plan_contents_embedding_similarity'])
                distance_to_targetJSON['sections_count_ratio_to_target_progress'] = get_ratio(sections_count_ratio_to_target, self.distance_to_targetJSON['sections_count_ratio_to_target'])
                distance_to_targetJSON['title_non_empty_count_ratio_to_target_progress'] = get_ratio(sections_title_non_empty_count_ratio_to_target, self.distance_to_targetJSON['title_non_empty_count_ratio_to_target'])
                distance_to_targetJSON['content_length_ratio_to_target_progress'] = get_ratio(content_length_ratio_to_target, self.distance_to_targetJSON['content_length_ratio_to_target'])
                distance_to_targetJSON['content_non_empty_count_ratio_to_target_progress'] = get_ratio(sections_content_non_empty_count_ratio_to_target, self.distance_to_targetJSON['content_non_empty_count_ratio_to_target'])

        # Save the results
        self.distance_to_targetJSON = distance_to_targetJSON

        return self.distance_to_targetJSON

    # return the list of current sections with title, length of content, validation status, and feedback
    def get_plan_status(self, compact_string_format: bool = False, keys = ["section_id", "title", "content_length", "content_preview[:100]"]):
        #keys = ["section_id", "title", "content_length", "validation_status", "feedback_to_process", "feedback_processed"]
        plan_status = []
        for section in self.document.document_content.sections_list:
            status_data_full = [
                section.section_id,
                section.title,
                len(section.content) if section.content is not None else 0,
                section.content[:100] if section.content is not None else 0,
                #round(section.content_progress_validation_status, 1),
                #section.local_feedback_to_process,
                #section.local_feedback_processed,
            ]
            status_data = [data for key, data in zip(keys, status_data_full)]

            if compact_string_format:
                plan_status.append("|".join(map(str, status_data)))
            else:
                plan_status.append(dict(zip(keys, status_data)))

        if len(plan_status) == 0: return []
        if compact_string_format:
            header = "|".join(keys)
        else:
            header = ["section_id", "title", "content_length", "content_preview[:100]"] #, "validation_status", "feedback_to_process", "feedback_processed"]
        plan_status.insert(0, header)
        
        return plan_status

    def get_resources_status(self, compact_string_format: bool = False):
        resources_status = []
        content_info = {}

        for resource in self.document.resources:
            # Vérifier que resource est un dict et contient la clé 'document'
            if isinstance(resource, dict) and 'document' in resource and isinstance(resource['document'], dict):
                # Vérifier que le document contient 'content' et que c'est un dict
                if 'content' in resource['document'] and isinstance(resource['document']['content'], dict):
                    for key, value in resource['document']['content'].items():
                        content_info[f"len(content['{key}'])"] = len(str(value))

                # Vérification des autres éléments
                status_data = [
                    resource.get('id', 'unknown'),  # Utiliser 'unknown' si id n'est pas disponible
                    resource.get('metadatas', {}).get('search', 'unknown'),  # Vérifier que 'metadatas' est un dict
                    resource['document'].get('name', 'unknown'),  # Utiliser 'unknown' si name n'est pas disponible
                    len(resource['document'].get('link', [])) if isinstance(resource['document'].get('link'), (list, tuple, np.ndarray)) else 0,  # Vérifier le type de 'link'
                    *content_info.values()  # Ajouter les longueurs de contenu
                ]

                # Formater la sortie selon compact_string_format
                if compact_string_format:
                    resources_status.append("|".join(map(str, status_data)))
                else:
                    keys = ["id", "metadatas", "document_name", "document_link_length"] + list(content_info.keys())
                    resources_status.append(dict(zip(keys, status_data)))
        
        if compact_string_format:
            # if content_info is empty, it means that there is no resource in the document
            if len(content_info) == 0:
                return []
            header = "id|metadatas|document_name|document_link_length|" + "|".join(content_info.keys())
            resources_status.insert(0, header)
        
        return resources_status

    def get_count_method_calls(self):
        return self._method_counts if hasattr(self, '_method_counts') else {}

    def reset_method_calls_counters(self):
        if hasattr(self, '_method_counts'):
            self._method_counts = {}

class VoyagerEnvIR_CPS_TechSynthesis(Environment):
    def __init__(self,
                 synthesis_type: str = "",
                 goal: str = "",
                 refined_goals: [str] = None,
                 #server_host='http://127.0.0.1', server_port=3000, request_timeout=600,
                 log_path='./logs',
                 CPS_env_type="techsynthesis",
                 title: str = "",
                 context: str = None,
                 embedding_model_name: str = "intfloat/e5-base-v2", # nomic-embed-text:latest, intfloat/e5-base-v2
                 openai_api_key: str = None,
                 target_file_path: str = None,
                 llm = None,
                 id: str = None,
                 ):
        super().__init__()
        self.id = str(uuid.uuid4()) if id is None else id
        if CPS_env_type != "techsynthesis":
            raise ValueError("problem_type must be techsynthesis")
        self.target_file_path = target_file_path
        self.synthesis_type = synthesis_type
        self.initial_goal = goal
        self.refined_goals = [goal] if refined_goals is None else refined_goals
        self.title = title
        self.context = context
        VoyagerEnvIR_CPS_TechSynthesis.llm_model = llm
        self.document = DocumentStructure(synthesis_type=synthesis_type, initial_goal=goal, refined_goals=self.refined_goals, embedding_model_name=embedding_model_name, title=title, context=context)  # Initialize your document structure
        self.synthesis_manager = SynthesisManager(document=self.document, target_file_path=target_file_path)  # Initialize your Synthesis Manager
        #self.server = f"{server_host}:{server_port}" # TODO: voir si on a besoin d'un serveur type TGI pour les inférences
        #self.request_timeout = request_timeout
        self.log_path = log_path
        self.has_reset_once = False
        self.state = None  # Placeholder: initial state of the environment

    # llm static method
    @staticmethod
    def llm(prompt: str):
        return VoyagerEnvIR_CPS_TechSynthesis.llm_model.invoke([SystemMessage(content=""), HumanMessage(content=prompt)]).content

    def get_score(self):
        distance = self.synthesis_manager.get_distance_to_targetJSON()
        return {'plan/titles similarity (top:1, worst:0)': distance['plan_titles_embedding_similarity'],
                'sections contents similarity (top:1, worst:0)': distance['plan_contents_embedding_similarity'],
                'sections count (top:1, <1:too short, >1:too long)': distance['sections_count_ratio_to_target'],
                'titles count (top:1, <1:too short, >1:too long)': distance['title_non_empty_count_ratio_to_target'],
                'sections contents length (top:1, <1:too short, >1:too long)': distance[
                    'content_length_ratio_to_target'],
                'sections contents non-empty (top:1, <1:too short, >1:too long)': distance[
                    'content_non_empty_count_ratio_to_target']}

    def reset(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.has_reset_once = True
        super().reset()
        self.document.reset()  # Reset document structure
        # TODO: could set different levels of reset or at least using reset to last saved state rather than full reset
        return self.document.get_state()  # Return initial state

    def close(self):
        with open(self.log_path, 'a') as log_file:
            print(f"Environment closed at {datetime.now()}\n") # TODO: log it
        self.document.reset()
        self.has_reset_once = False
        self.state = None
        print("Environment internal states reset and closed.") # TODO: log it

    def step(
        self,
        code: str = "",
        programs: str = "",
        reset_step: bool = False,
        completed_tasks: int = 0,
    ):
        self.synthesis_manager.reset_method_calls_counters() # we want to count method calls for a step only
        if not self.has_reset_once:
            print("Environment has not been reset yet - resetting now !")
            self.reset()
        return super().step(action_code=code, context={'problem': self.synthesis_manager, 'bot': self.synthesis_manager, 'results': None, 'SynthesisManager': SynthesisManager, 'DocumentStructure': DocumentStructure, 'Section': Section, 'Document': Document, 'llm':VoyagerEnvIR_CPS_TechSynthesis.llm})

    def get_state(self, extended: bool = False):
        #TODO: move to self.document.get_state() ?
        table_of_content_list = self.synthesis_manager.get_plan_status(compact_string_format=False if extended else True)
        # Convert the list of sections to a string with \n separator
        table_of_content = "\n".join([str(section) for section in table_of_content_list])
        resources_observation = self.synthesis_manager.get_resources_status(compact_string_format=True)
        document_state = f"<<< Document #{self.id} properties:\n"
        if extended: document_state += f"> Title: {self.title}\n"
        if extended: document_state += f"> Abstract (first 100 characters): {self.context[:100]}\n"
        document_state += f"> Current table of content:\n{table_of_content if len(table_of_content) > 0 else 'Empty'}\n"
        document_state += f"> Current resources: {resources_observation if len(resources_observation) > 0 else 'Empty'}\n"
        if extended:
            distance_to_targetJSON = self.synthesis_manager.get_distance_to_targetJSON()
            events_action_counts = self.synthesis_manager.get_count_method_calls()
            document_state += f"1. sections titles progress: {distance_to_targetJSON['plan_titles_embedding_similarity']}\n"
            document_state += f"2. sections content progress: {distance_to_targetJSON['plan_contents_embedding_similarity']}\n"
            document_state += f"3. sections count ratio progress: {distance_to_targetJSON['sections_count_ratio_to_target']}\n"
            document_state += f"4. title non-empty count ratio progress: {distance_to_targetJSON['title_non_empty_count_ratio_to_target']}\n"
            document_state += f"5. content length ratio progress: {distance_to_targetJSON['content_length_ratio_to_target']}\n"
            document_state += f"6. content non-empty count ratio progress: {distance_to_targetJSON['content_non_empty_count_ratio_to_target']}\n"
            document_state += f"7. events counted: {events_action_counts if len(events_action_counts) > 0 else 'Empty'}\n"

        document_state += ">>>"
        return document_state
    
    
class SWEBenchEnvironment(Environment):
    def __init__(self,
                 synthesis_type: str = "",
                 refined_goals: [str] = None,
                 goal: str = "",
                 title: str = "",
                 openai_api_key: str = None,
                 target_repo_commit: str = None,
                 target_repo_id: str = None,
                 llm = None,
                 log_path='./logs',
                 id: str = None,
                 CPS_env_type="sweSynthesis",
                 context: str = None,
                 directory_path="../../swe_repo", 
                 problem_id: str = None,                 
                 ):
        super().__init__()
        

        # dataset = load_dataset(path="ahsanirfan961/swe-bech-lite-bm25-13k-take3", split='train'
        self.files_having_bugs=[],
        self.id = str(uuid.uuid4()) if id is None else id
        self.generated_patch_code="",
        self.patch_applied=False,
        self.unit_tests=[],
        self.unit_test_passed=3,
        self.unit_test_failed=0,
        self.has_reset_once = False,
        SWEBenchEnvironment.llm_model = llm
        # os.chdir('swe_repo')
        self.directory_path=str(directory_path),
        self.synthesis_manager = SweSynthesisManager(directory_path=self.directory_path) # Initialize your Synthesis Manager
    
    @staticmethod
    def llm(prompt: str):
        return SWEBenchEnvironment.llm_model.invoke([SystemMessage(content=""), HumanMessage(content=prompt)]).content
           
    def get_properties_as_string(self):
        properties = {
            'files_having_bugs': self.files_having_bugs,
            'generated_patch_code': self.generated_patch_code,
            'patch_applied': self.patch_applied,
            'unit_tests': self.unit_tests,
            'unit_test_passed': self.unit_test_passed,
            'unit_test_failed': self.unit_test_failed
        }
        # Generate string representation of each property and its value
        result = "\n".join([f"{key}: {value}" for key, value in properties.items()])
        return result
    
          
    def get_row_by_instance_id(self,instance_id):
     for row in self.dataset:
        if row['instance_id'] == instance_id:
            return row
     return None
    
        
    def clone_repo(self,instance_id,target_folder=None):
     try:
        repo=self.get_row_by_instance_id(instance_id)
         
        print(
            f"Cloning repository from https://github.com/{repo['repo']}.git to Repositories Folder"
        )
        current_dir = os.path.dirname(os.path.realpath(__file__))
        if(target_folder is None):
         target_folder = os.path.join(current_dir, '..', '..','swe_repo')
        os.chdir(target_folder)
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo['repo']}.git",
            ],
            check=True,
        )
        print("Repository cloned successfully.")
     except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
     except Exception as e:
        print(f"An unexpected error occurred: {e}")  
            
    def checkout_commit(repo_path, commit_id):
     """Checkout the specified commit in the given local git repository.
     :param repo_path: Path to the local git repository
     :param commit_id: Commit ID to checkout
     :return: None
     """
     try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Checking out commit {commit_id} in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
     except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
     except Exception as e:
        print(f"An unexpected error occurred: {e}")      
    
    
    def get_project_structure_from_scratch(self,repo_name, commit_id, instance_id):
        
     self.clone_repo(instance_id)
     self.checkout_commit("../../swe_repo/{repo_name}", commit_id)
     structure = self.create_structure(f"{repo_playground}/{repo_to_top_folder[repo_name]}")
 
 
 
     d = {
        "repo": repo_name,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
     }
     return d
        
    
    def create_structure(self,directory_path):
     """Create the structure of the repository directory by parsing Python files.
         :param directory_path: Path to the repository directory.
         :return: A dictionary representing the structure.
     """
     structure = {}

     for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = self.parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    
    
     return structure   
 
    
    def parse_python_file(self,file_path, file_content=None):
     """Parse a Python file to extract class and function definitions with their line numbers.
     :param file_path: Path to the Python file.
     :return: Class names, function names, and file contents
     """
     if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            # print(f"Error in file {file_path}: {e}")
            return [], [], ""
     else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            # print(f"Error in file {file_path}: {e}")
            return [], [], ""

     class_info = []
     function_names = []
     class_methods = set()

     for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

     return class_info, function_names, file_content.splitlines()    

    
    def get_state(self,unique_id=None, extended_comparison=False, extended=False):
        return self.get_properties_as_string()
    
    
    def step(
        self,
        code: str = "",
        programs: str = "",
        reset_step: bool = False,
        completed_tasks: int = 0,
    ):
        self.synthesis_manager.reset_method_calls_counters() # we want to count method calls for a step only
        if not self.has_reset_once:
            print("Environment has not been reset yet - resetting now !")
            self.reset()
        return super().step(action_code=code, context={'problem': self.synthesis_manager, 'bot': self.synthesis_manager, 'results': None, 'SynthesisManager': SweSynthesisManager, 'llm':SWEBenchEnvironment.llm})
        


class SweSynthesisManager:
    def __init__(self, target_file_path: str = None,directory_path: str = None):   
        if target_file_path:
            self.target_file_path = target_file_path            
        self.directory_path = directory_path        
    
    
    @method_call_counter
    @tool
    def ls(self) -> str:
     """
     This function lists the files in the current directory.
     :function: ls
     :return: a string containing the path of current directory and the files present in the current directory
     """
     ls_out = shell.run({"commands": [f"cd test-repos{self.directory_path}", 'ls']})
     return f"""
     Current Directory: {self.directory_path}
     Files: 
     {ls_out}
     """


   
    @method_call_counter
    @tool
    def goto_directory(self,path: str) -> str:
     """
     This function changes the current directory to the specified directory. This function must be used to goto a directory not to open a file.
     :function: goto_dir
     :param str path: path of the new directory you want to change relative to the current directory e.g 'matplotlib/doc'
     :return: output of command 'fail' or 'success'
     """

     if '..' in path:
        return 'use goto_previous_dir tool instead to go to previous directory'
     out = shell.run({"commands": [f"cd test-repos{self.directory_path}", f'cd {path}']})
     if out == "":
        self.directory_path = f"{self.directory_path}{path}/"
        return 'successfully entered ' + self.directory_path
     else:
        return out
    
    
    

    
    @method_call_counter
    @tool
    def goto_previous_dir(self) -> str:
     """
     This function takes the user to the previous directory.
     :function: goto_previous_dir
     :return: output 
     """
     if self.directory_path == '/':
         return "Already in top most directory. Can't go back anymore"
     else:
        paths = self.directory_path.split('/')[1:-1]
        paths.pop()
        self.directory_path = '/'
        for dir in paths:
           self.directory_path += dir + '/'
        return f"Current Directory: {self.directory_path}"

    
    @method_call_counter
    @tool
    def get_current_dir(self) -> str:
     """
     This function returns the path of the currently opened directory.
     :function: get_current_dir
     :return: the current directory
     """
     return self.directory_path


    @method_call_counter
    @tool
    def get_abs_current_dir(self) -> str:
     """
    Constructs and returns the absolute path to the current directory.

    The path is generated by joining the base project directory (`self.abs_project_dir`)
    with the subdirectory 'test-repos' and a modified version of `self.directory_path` 
    (with the first and last characters removed). 

    Returns:
        str: The absolute path to the current directory.
       """
     return os.path.join(self.abs_project_dir, 'test-repos', self.directory_path[1:-1])


    
    @method_call_counter
    @tool
    def number_of_lines(self,path: str) -> str:
     """
     This function takes a file path as input and returns the number of lines in the file.
    :function: number_of_lines
     :param path: The relative path to the file (e.g., 'lib/matplotlib/axis.py').
     
     :return: The number of lines in the file.
     """

     abs_file_path = os.path.join(self,self.get_abs_current_dir(), path)
     if os.path.exists(abs_file_path):
        with open(abs_file_path, 'r') as file:
            return f"Number of lines in {self.directory_path+path}: {sum(1 for line in file)}"
     else:
        return f"File {self.directory_path+path} not found"


    
    @method_call_counter
    @tool
    def open_file(self,path: str, line_number: int = 1, max_lines: int = 100) -> str:
     """   
     This function takes a file path, a line number, and a maximum number of lines as input and returns the contents of the file starting from the specified line number, limited to the maximum number of lines.
     :function: open_file
    :param path: The relative path to the file (e.g., 'lib/matplotlib/axis.py').
     :param line_number: The line number from which to start reading the file. Defaults to 1.
    :param max_lines: The maximum number of lines to return from the starting line. Defaults to 100.
     :return: A string containing the file contents from the specified starting line, limited to max_lines.
     """

     if line_number < 1:
        return "Error: line number cannot be zero or negative"
     if max_lines < 1:
        return "Error: max_lines cannot be zero or negative"

     abs_file_path = os.path.join(self.get_abs_current_dir(), path)
     if os.path.exists(abs_file_path):
        with open(abs_file_path, 'r') as file:
            with open(abs_file_path, 'r') as temp_file:
                num_lines = sum(1 for line in temp_file)
            if line_number > num_lines:
                return f"Can't access {line_number} line. This file only contains {num_lines} lines"
            
            out = f"Showing contents of File: {self.directory_path+path} starting from {line_number}\n\n"
            for n, line in enumerate(file, 1):
                if n >= line_number:
                    out += f"{n}: {line}\n"
                    if n == line_number + max_lines - 1:
                        break
            return out
     else:
        return path + " doesn't exist"

    
    
    @method_call_counter
    @tool
    def find_files(self,file_name: str) -> str:
     """
     Searches the current directory and its subdirectories for the files that have name containing the specified file_name.
     :function: find_files
     :param file_name: The file_name to search for in file names.
     :return: paths of the files that contain the specified file_name in their names.
     """
     matched_files = []
     
     # Walk through the current directory and all subdirectories
     for root, dirs, files in os.walk("test-repos" + self.directory_path):
         for file in files:
             if file_name in file:
                 matched_files.append(os.path.join(self.directory_path, file))
     
     return "Files found:\n" + "\n".join(matched_files)

    
    
    @method_call_counter
    @tool
    def search_file(self,path: str, search_term: str) -> str:
     """
     This function takes a file path and a search term as input and returns the lines in the file that contain the search term. 
     :param str path: The relative path to the file (e.g., 'lib/matplotlib/axis.py').
     :param str search_term: The term to search for in the file.
     :return: A string containing the lines in the file that contain the search term.
     """
 
     abs_file_path = os.path.join(self.get_abs_current_dir(), path)
     if os.path.exists(abs_file_path):
         with open(abs_file_path, 'r') as file:
             out = f"Searching for '{search_term}' in {self.directory_path+path}\n\n"
             for n, line in enumerate(file, 1):
                 if search_term in line:
                     out += f"{n}: {line}\n"
             return out
     else:
        return f"File {self.directory_path+path} not found"
    


    
    @method_call_counter
    @tool
    def search_dir(self,path: str, search_term: str) -> str:
     """
     Searches for files in the specified directory that contain the search term. It returns the file names and line numbers where the search term is found.
 
     :param path: The relative path to the directory (e.g., 'lib/matplotlib').
     :param search_term: The term to search for in the files in the directory.
     :return: A string containing the files in the directory that contain the search term.
     """
     abs_dir_path = os.path.join(self.get_abs_current_dir(), path)
     
     if os.path.exists(abs_dir_path):
         matched = ""
         for root, dirs, files in os.walk(abs_dir_path):
             for file in files:
                 file_path = os.path.join(root, file)
                 try:
                     with open(file_path, 'r', encoding='utf-8') as file_a:
                         for n, line in enumerate(file_a, 1):
                             if search_term in line:
                                 matched += f"File: {file_path}, Line: {n}\n"
                                 break
                 except (UnicodeDecodeError, IOError):  # Handle decoding errors and file I/O errors
                     continue
         
         if matched:
             return "Files found:\n" + matched
         else:
             return "No files containing the search term were found."
     else:
        return f"Directory {abs_dir_path} not found."

   
   
    @method_call_counter
    @tool
    def get_files_content(self,paths: List[str], line_numbers: List[int]) -> str:
     """
     This function takes a list of file paths and a list of line numbers as input and returns the contents of the files starting from the specified line numbers.
     :param paths: A list of relative paths to the files (e.g., ['lib/matplotlib/axis.py', 'lib/matplotlib/figure.py']).
     :param line_numbers: A list of line numbers from which to start reading the files.
     :return: A string containing the contents of the files from the specified starting line numbers.
     """
     out = ""
     for path, line_number in zip(paths, line_numbers):
         out += self.open_file.invoke({'path': path, "directory.cwd":self.directory_path , 'line_number': line_number, 'max_lines': 50}) + '\n\n'
     return out

   
    @method_call_counter
    def getSearchTools(self):
     return [self.ls, self.goto_directory, self.goto_previous_dir, self.get_current_dir, self.number_of_lines, self.open_file, self.find_files, self.search_file, self.search_dir]


    @method_call_counter
    def getContentViewingTools(self):
     return [self.get_files_content]
    
    @method_call_counter
    def reset_method_calls_counters(self):
        if hasattr(self, '_method_counts'):
            self._method_counts = {}

        
    
            