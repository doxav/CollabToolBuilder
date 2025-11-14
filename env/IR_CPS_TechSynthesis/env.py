from datetime import datetime
import os.path

# import warnings
import copy
import time
import uuid
from dataclasses import dataclass, field
import re
from typing import List, SupportsFloat, Any, Tuple, Dict
from dataclasses import asdict
from langchain_core.messages import SystemMessage, HumanMessage

from config import *

# from attr import dataclass, field
# import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pdb

import traceback

import json
import requests

from env.env import Environment

# import a function from langchain which could embed a text into a vector using OpenAI ada-002 or HuggingFace
import langchain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from utils.file_utils import save_to_pickle, load_from_pickle
from utils.llm_utils import UnifiedVectorDB, UnifiedVectorDBConfig

# from langchain_community.cache import InMemoryCache, SQLiteCache
# langchain.llm_cache = SQLiteCache(database_path="sqlite/langchain_cache.db")

from bs4 import BeautifulSoup

##############################################################################################################
# Placeholder classes for the technical synthesis environment
##############################################################################################################


def method_call_counter(method):
    def wrapper(*args, **kwargs):
        if args and hasattr(args[0], "__dict__"):
            self = args[0]
            if not hasattr(self, "_method_counts"):
                self._method_counts = {}
            self._method_counts[method.__name__] = (
                self._method_counts.get(method.__name__, 0) + 1
            )
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
    resource: str = ""  # temporary support
    resources: List[str] = None
    title_validation_status: int = 0
    content_progress_validation_status: int = 0
    local_feedback_to_process: List[str] = field(
        default_factory=list
    )  # could be citation to integrate, critics to process...
    local_feedback_processed: List[str] = field(default_factory=list)  # same as above


@dataclass
class Document:
    title: str = (
        ""  # e.g. should be the title of the document if the goal is to write a SOTA survey paper, the title of a Wikipedia article if the goal is to write a Wikipedia article, the title of a patent if the goal is to write a patent, ...
    )
    title_embedding: List[float] = field(
        default_factory=list
    )  # TODO: check if we need to store the embedding of the title because, differently to sections, it is not used in comparison to target because it is an input
    context: str = (
        ""  # e.g. should be the abstract content of the document if the goal is to write a SOTA survey paper, the introduction of a Wikipedia article if the goal is to write a Wikipedia article, the introduction of a patent if the goal is to write a patent, ...
    )
    context_embedding: List[float] = field(
        default_factory=list
    )  # TODO: check if we need to store the embedding of the context because, differently to sections content, it is not used in comparison to target because it is an input
    # resource: str = ""
    resources_embedding: List[float] = field(default_factory=list)
    sections_list: List[Any] = field(default_factory=list)
    sections_list_embedding: List[float] = field(default_factory=list)


class DocumentStructure:
    # declare a class variable to store the embedding model class instance to avoid reinitializing it for each DocumentStructure instance
    embedding_model_cls = None
    embedding_model_cls_name = None

    def __init__(
        self,
        synthesis_type: str,
        initial_goal: str,
        refined_goals: List[str] = None,
        embedding_model_name: str = "intfloat/e5-base-v2",  # nomic-embed-text:latest, intfloat/e5-base-v2
        embedding_model_query_prefix: str = "",
        # e.g. "query: " for intfloat/e5-base-v2 should improve for QA but we are in estimating straight semantic similarity
        title: str = None,
        context: str = None,
        resources: str = None,
        abstract: str = None,
    ):
        if abstract is not None:
            context = abstract if context is None else context + "\n" + abstract
        self.abstract = abstract
        self.embedding_model_query_prefix = embedding_model_query_prefix
        self.embedding_model_name = embedding_model_name
        if not DocumentStructure.embedding_model_cls:
            if (
                DocumentStructure.embedding_model_cls_name == embedding_model_name
                and DocumentStructure.embedding_model_cls
            ):
                self.embedding_model = DocumentStructure.embedding_model_cls
            elif embedding_model_name == "text-embedding-ada-002":
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError(
                        "OpenAI API key is required for OpenAI ada-002 model."
                    )
                self.embedding_model = OpenAIEmbeddings(
                    model=embedding_model_name
                )  # , openAIApiKey=os.getenv("OPENAI_API_KEY")
            else:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    encode_kwargs={"normalize_embeddings": True},
                    model_kwargs={"trust_remote_code": True},
                )  # , openAIApiKey=os.getenv("OPENAI_API_KEY"
            (
                DocumentStructure.embedding_model_cls,
                DocumentStructure.embedding_model_cls_name,
            ) = (self.embedding_model, embedding_model_name)
        else:
            self.embedding_model = DocumentStructure.embedding_model_cls

        self.synthesis_type = synthesis_type
        self.initial_goal = initial_goal
        self.refined_goals = [initial_goal] if refined_goals is None else refined_goals
        self.document_content = (
            Document()
        )  # self.document_content = copy.deepcopy(DOCUMENT_SCHEMA)
        # self.document_content.sections_list = []
        self.title = title
        if title and title != "":
            self.set_plan_field_with_embedding("title", title)
        self.context = context
        if context and context != "":
            self.set_plan_field_with_embedding("context", context)
        if resources and resources != "":
            self.set_plan_field_with_embedding("resources", resources)
        self.dumb_embedding = self.embedding_model.embed_query(
            "."
        )  # Used to compute min_cosine_similarity
        self.embedding_size = len(self.dumb_embedding)

        self.global_feedback_to_process = []
        self.global_feedback_processed = []
        self.resources = []
        self.section_hashes = (
            {}
        )  # Used for tracking changes but only used for Latex conversion
        name = f"{self.synthesis_type}_{self.title}"
        self.resources_vectordb = UnifiedVectorDB(
            UnifiedVectorDBConfig(
                collection_name=name,
                embedding_function=self.embedding_model,
                persist_directory=f"ckpt/doc/{name}",
            )
        )
        self.events = []

        self.get_state(save=True)

    def get_embedding(self, text: str) -> List[float]:
        # if not text:  # or if text == ""
        #     return np.zeros(self.embedding_size)  # Assuming you know the size of your embeddings
        # else:
        return self.embedding_model.embed_query(
            self.embedding_model_query_prefix + text
        )

    def update_sections_embeddings(
        self,
        section_ids: List[int] = None,
        force_update: bool = False,
        batch_update: bool = False,
        skip_plan_embeddings_update: bool = False,
    ):
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
                    sections_label_to_update.append("content_embedding")
                else:
                    section.content_embedding = self.get_embedding(section.content)
            if (not section.title_embedding and section.title) or force_update:
                if batch_update:
                    texts_to_embed.append(section.title)
                    sections_to_update.append(section)
                    sections_label_to_update.append("title_embedding")
                else:
                    section.title_embedding = self.get_embedding(section.title)

        if batch_update:
            # Get embeddings in one batch call
            embeddings = self.get_embedding(texts_to_embed)

            for section, emb, label in zip(
                sections_to_update, embeddings, sections_label_to_update
            ):
                setattr(section, label, emb)

        if not skip_plan_embeddings_update:
            self.update_plan_embedding()

    def update_plan_embedding(self):
        """update plan embedding by computing mean of all section embeddings and the title embedding (if any) of the synthesis plan"""
        title_embeddings = []  # List to hold title embeddings
        content_embeddings = []  # List to hold content embeddings
        resource_embeddings = []
        embedding_label = "resource_embedding_" + (
            "1" if self.embedding_model_name == "text-embedding-ada-002" else "2"
        )  # get mode embeddings length
        # get mode embeddings length

        try:
            # Populate the title_embeddings and content_embeddings lists
            # x.title_embedding should be added only if x.title_embedding is not empty (nor None, nor '')
            title_embeddings = [
                x.title_embedding
                for x in self.document_content.sections_list
                if x.title_embedding
            ]
            content_embeddings = [
                x.content_embedding
                for x in self.document_content.sections_list
                if x.content_embedding
            ]
            # resource_embeddings = [x.resource_embedding for x in self.document_content.sections_list if x.resource_embedding]
            # resource_embeddings = [resource.get(embedding_label, None) for resource in self.resources]
            for resource in self.resources:
                if embedding_label not in resource and (
                    "name" in resource or "description" in resource
                ):
                    # resource_description = str(resource['description'])
                    # + (("\n" + str(resource['content'])) if 'content' in resource else None)
                    # + ("\n" + str(resource['name']) if 'name' in resource else None)
                    # + ("\n" + str(resource['link']) if 'link' in resource else None)
                    resource_description = str(
                        resource.get("name", resource.get("description", ""))
                    )
                    resource[embedding_label] = self.get_embedding(resource_description)
                resource_embeddings.append(resource[embedding_label])

            # convert self.dumb_embedding to a list to be able to use it in np.mean
            # Compute mean embedding for titles
            title_mean = (
                np.mean(title_embeddings, axis=0).tolist()
                if title_embeddings
                else self.dumb_embedding
            )
            # if content_embeddings is empty, set content_mean to 0
            content_mean = (
                np.mean(content_embeddings, axis=0).tolist()
                if content_embeddings
                else self.dumb_embedding
            )
            resource_mean = (
                np.mean(resource_embeddings, axis=0).tolist()
                if resource_embeddings
                else self.dumb_embedding
            )
            # Combine title and content embeddings and compute their mean
            all_embeddings = (
                title_embeddings + content_embeddings
            )  # + resource_embeddings
            total_mean = (
                np.mean(all_embeddings, axis=0).tolist()
                if all_embeddings
                else self.dumb_embedding
            )

        except KeyError:
            raise KeyError(
                "Could not find title_embedding or content_embedding in every section. Please check that every section has both of these keys."
            )

        self.document_content.sections_list_title_embedding = (
            title_mean  # Store the mean title embedding
        )
        self.document_content.sections_list_content_embedding = (
            content_mean  # Store the mean content embedding
        )
        self.document_content.resources_embedding = resource_mean
        self.document_content.sections_list_embedding = (
            total_mean  # Store the combined mean embedding
        )

    def set_plan_field_with_embedding(
        self, field: str, value: str, event: str = None, section_id: int = None
    ):
        # can be chained with other set_plan_field calls (e.g. set_plan_field('title', 'my title').set_plan_field('context', 'my context').set_plan_field('title', 'my new title', section_id=1))
        embedding = self.get_embedding(value)

        if section_id:
            target = next(
                (
                    s
                    for s in self.document_content.sections_list
                    if s.section_id == section_id
                ),
                None,
            )
            if not target:
                raise ValueError(f"Section with id {section_id} not found.")
            setattr(
                target, field, value
            )  # Using setattr because target is a class instance
            setattr(target, f"{field}_embedding", embedding)
        else:
            setattr(
                self.document_content, field, value
            )  # because self.document_content is a class instance
            setattr(self.document_content, f"{field}_embedding", embedding)

        if event:
            self.events.append(("observe", {"action": event, "section_id": section_id}))
        return self

    def get_state(self, save: bool = False) -> Dict[str, Any]:
        state = {
            "synthesis_type": self.synthesis_type,
            "initial_goal": self.initial_goal,
            "refined_goals": self.refined_goals,
            "document_content": self.document_content,
            "global_feedback_to_process": self.global_feedback_to_process,
            "global_feedback_processed": self.global_feedback_processed,
            "resources": self.resources,
            "events": self.events,
            "embedding_model_name": self.embedding_model_name,
        }
        if save:
            self.last_state = copy.deepcopy(state)
        return state

    def restore_state(self, state: Dict[str, Any] = None):
        """if state is empty, restore last saved state, else restore the provided state"""
        if state is None:
            state = self.last_state
        self.synthesis_type = state["synthesis_type"]
        self.initial_goal = state["initial_goal"]
        self.refined_goals = state["refined_goals"]
        self.document_content = state["document_content"]
        self.global_feedback_to_process = state["global_feedback_to_process"]
        self.global_feedback_processed = state["global_feedback_processed"]
        self.resources = state["resources"]
        self.events = state["events"]
        return self.get_state()

    def reset(self):
        # TODO: check if refined_goals should be reset or not
        # this code might be redundant but ensure sections_list embeddings memory are cleared (might be removed later)
        for section in self.document_content.sections_list:
            section.content_embedding = []
            section.title_embedding = []
            section.resources = []
        self.document_content.sections_list_embedding = []
        self.document_content = (
            Document()
        )  # self.document_content = copy.deepcopy(DOCUMENT_SCHEMA)

        self.resources.clear()
        self.events.clear()
        self.global_feedback_to_process.clear()
        self.global_feedback_processed.clear()
        self.get_state(save=True)
        return self

    def add_event(self, event="observe", status: Dict[str, Any] = None):
        return self.events.append((event, status))

    def get_events(self):
        return self.events


class SynthesisManager:
    def __init__(self, document: DocumentStructure, target_file_path: str = None):
        self.plan_embedding_update_required = False
        self.document = document
        self.title = self.document.title
        self.abstract = self.document.context
        self.min_cosine_similarity = cosine_similarity(
            [self.document.embedding_model.embed_query(".")],
            [
                self.document.embedding_model.embed_query(
                    "If you can keep your head when all about you are losing theirs and blaming it on you, If you can trust yourself when all men doubt you, But make allowance for their doubting too ; If you can wait and not be tired by waiting, Or being lied about, don't deal in lies, Or being hated, don't give way to hating, And yet don't look too good, nor talk too wise"
                )
            ],
        )[0][0]
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
            print(e)
            return False

    @method_call_counter
    def chat(self, message: str):
        print(f"SynthesisManager Chat message: {message}")

    @staticmethod
    @method_call_counter
    @load_from_pickle
    @save_to_pickle
    def search_generic(query, search_type, output_format="json", max_results=10):
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
        search_functions = {  # ['paper_arxiv','paper_pubmed','websearch_google','websearch_wikipedia','paper_semantic_scholar']
            "patents_google": SynthesisManager.search_google_patents,  #
            "patent_epo": SynthesisManager.search_epo,  #
            "patent_uspto": SynthesisManager.search_uspto,  #
            "paper_arxiv": SynthesisManager.search_arxiv,  #
            "paper_pubmed": SynthesisManager.search_pubmed,
            #'websearch_bing': SynthesisManager.search_bing,
            "websearch_google": SynthesisManager.search_google,  #
            "core": SynthesisManager.search_core,  #
            "websearch_wikipedia": SynthesisManager.search_wikipedia,  #
            "paper_semantic_scholar": SynthesisManager.search_semantic_scholar,  #
            "fiche_anomalie": SynthesisManager.search_fiche_anomalie,  #
        }
        search_function = search_functions.get(search_type)
        if search_function:
            results = search_function(
                query, output_format=output_format, max_results=max_results
            )
            return results
        else:
            raise ValueError(f"Unsupported search type: {search_type}")

    # OK, basic implementation via scraping (short description and link to paper which should be scraped also to get full description) - waiting for a key for a proper implementation
    @staticmethod
    @method_call_counter
    def search_epo(query, output_format="json", max_results=20, use_EPO_API=False):
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
            options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            )

            # Use headless mode (no GUI)
            options.add_argument("--headless")

            # Create a browser instance
            driver = webdriver.Chrome(options=options)

            # URL for EPO scraping
            # encode query string
            url = "https://worldwide.espacenet.com/patent/search?q=" + query.replace(
                " ", "%20"
            )
            print(url)
            driver.get(url)

            # Wait until the search results are loaded
            try:
                # element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
                element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "item--wSceB4di"))
                )
                # sleep for 1 second
                time.sleep(1)
                # WebDriverWait(driver, 10).until(lambda d: d.execute_script('return document.readyState') == 'complete')
                soup = BeautifulSoup(driver.page_source, "html.parser")
            except Exception as e:
                # print error
                error = str(e.message) if hasattr(e, "message") else str(e)
                print(f"EPO SELENIUM scraping failed with error code: {error}")
                soup = None
            driver.quit()

            # Get all results
            results = []
            if soup is not None:
                # Find the container div for the results first
                container_div = soup.find("div", class_="publications-list--9wu4rcWN")

                # If container_div is found, extract individual results
                if container_div:
                    for article in container_div.find_all("article"):
                        try:
                            title_tag = article.find(
                                "span", class_="item__content--title--dYTuyzV6"
                            )
                            if title_tag:
                                title = title_tag.text.strip()
                            else:
                                title = "N/A"

                            patent_num_tag = article.find(
                                "div", class_="item__content--subtitle--mFxM6gqw"
                            )
                            if patent_num_tag:
                                patent_num = patent_num_tag.find("span").text
                                link = f"https://worldwide.espacenet.com/patent/search?q=pn%3D{patent_num}"
                            else:
                                link = "N/A"

                            description_tag = article.find(
                                "div", class_="copy-text--uk738M73"
                            )
                            if description_tag:
                                description = description_tag.text.strip()
                            else:
                                description = "N/A"

                            results.append(
                                {
                                    "title": title,
                                    "link": link,
                                    "description": description,
                                }
                            )

                        except Exception as e:
                            print(f"Error while extracting data for one article: {e}")
                else:
                    print("Couldn't find the results container.")

            if output_format == "json":
                return json.dumps(results)
            else:
                return "\n".join(
                    [
                        "{} - {}: {}".format(
                            result["title"], result["link"], result["description"]
                        )
                        for result in results
                    ]
                )
        else:
            raise NotImplementedError("EPO API not implemented yet")
            # TODO: implement EPO API using https://developers.epo.org/ops-v3-2/apis
            url = "https://worldwide.espacenet.com/3.2/rest-services/search"
            # https://worldwide.espacenet.com/3.2/rest-services/search?lang=en%2Cde%2Cfr&q=cancer%20language%20models%20information&qlang=cql&
            params = {
                "q": query,
                "Range": "1-{}".format(max_results),
            }
            # set header Authorization Bearer 0p8ydddwdAiGBJERyBAMGVCYJAtw

            response = requests.get(url, params=params)
            if output_format == "json":
                import xmltodict

                response = xmltodict.parse(response.text)
                return response
            else:
                return response.text

    # OK, based on Arxiv API
    @staticmethod
    @method_call_counter
    def search_arxiv(query, output_format="json", max_results=10):
        """
        Search arXiv for articles relating to `query`.
        Returns a list of dictionaries containing article information.
        """

        arxiv_url = "http://export.arxiv.org/api/query?search_query={}&start=0&max_results={}&sortBy=relevance&sortOrder=descending".format(
            query, max_results
        )

        response = requests.get(arxiv_url)

        if response.status_code != 200:
            return []

        feed = response.text

        if output_format == "json":
            import feedparser

            feed = feedparser.parse(feed)

            # Parse and transform the results
            results = []
            for entry in feed["entries"]:
                pdf_link = [
                    link["href"]
                    for link in entry.get("links", [])
                    if link.get("type") == "application/pdf"
                ]
                results.append(
                    {
                        "title": entry.get("title", ""),
                        "link": pdf_link[0] if pdf_link else "",
                        "description": entry.get("summary", ""),
                    }
                )
            return results

        else:
            return feed

    # OK, based on semantic scholar API
    # Could be enriched: https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_get_paper_search
    @staticmethod
    @method_call_counter
    def search_semantic_scholar(query, output_format="json", max_results=10):
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
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "paperId,title,abstract",
        }

        response = requests.get(base_url, params=params)
        response_data = response.json()

        # Process response data
        results = []
        for paper in response_data.get("data", []):
            paper_id = paper.get("paperId", "")
            title = paper.get("title", "")
            # Construct the URL using paperId
            link = f"https://www.semanticscholar.org/paper/{paper_id}"
            description = paper.get(
                "abstract", ""
            )  # Assuming abstract can be treated as a description

            results.append({"title": title, "link": link, "description": description})

        if output_format == "json":
            return results
        else:
            return {"error": "Output format not supported"}

    # OK: based on search_google
    @staticmethod
    @method_call_counter
    def search_google_patents(
        query, output_format="json", max_results=10, pdf_only=False, scraping=True
    ):
        if pdf_only:
            print("Google patents does not support pdf only search")
        return SynthesisManager.search_google(
            query=query,
            output_format=output_format,
            pdf_only=False,
            max_results=max_results,
            try_direct_scraping_first=scraping,
            specific_site="patents.google.com",
            result_pretty=True,
        )

    # NOK: based on search_google but information are not well formatted and poor
    @staticmethod
    @method_call_counter
    def search_uspto(
        query, output_format="json", max_results=10, pdf_only=False, scraping=True
    ):
        return SynthesisManager.search_google(
            query=query,
            output_format=output_format,
            pdf_only=pdf_only,
            max_results=max_results,
            try_direct_scraping_first=scraping,
            specific_site="uspto.gov",
            result_pretty=True,
        )

    # A function which can web search either via Google API, either via Google scraping
    @staticmethod
    @method_call_counter
    @load_from_pickle
    @save_to_pickle
    def search_google(
        query,
        output_format="json",
        pdf_only=False,
        max_results=20,
        try_direct_scraping_first=True,
        specific_site=None,
        result_pretty=True,
        use_proxy=False,
        try_selenium_if_1st_scraping_failed=True,
    ):
        if try_direct_scraping_first:
            # Google scraping, pdf type is optional
            url = "https://www.google.com/search"

            if use_proxy:
                # pip install fake-useragent
                from fake_useragent import UserAgent

                ua = UserAgent()

                # get a random proxy through this command 'https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=us,fr&ssl=all&anonymity=all'
                proxies = requests.get(
                    "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=https&timeout=5000&country=all&ssl=yes&anonymity=elite"
                ).text.split("\r\n")
            else:
                # Default headers for Google scraping
                # headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' }
                from fake_useragent import UserAgent

                ua = UserAgent()
                headers = {"User-Agent": ua.edge}

            params = {
                "q": ("site:" + specific_site + " " if specific_site else "") + query,
                "num": max_results,
                "fileType": "pdf" if pdf_only else None,
            }

            with requests.Session() as s:
                if use_proxy:
                    # Try a proxy until s.get() works
                    while True and len(proxies) > 0:
                        # pick a random proxy from the list using random in the number of proxies
                        random_proxy = random.randint(0, len(proxies) - 1)
                        proxy = proxies[
                            random_proxy
                        ]  # return a string with the format 'https://ip:port'
                        # remove the proxy from the list
                        proxies.pop(random_proxy)
                        print(proxy)
                        s.proxies = {"https": proxy}
                        print(url)
                        try:
                            # set timeout to 1 second
                            headers = {"User-Agent": ua.edge}
                            r = s.get(url, params=params, headers=headers, timeout=5)
                            break
                        except:
                            print(
                                "Proxy error - remaining proxies: " + str(len(proxies))
                            )
                            continue
                    print("SUCCESS Proxy used: " + proxy)
                else:
                    r = s.get(url, params=params, headers=headers)
                # Generate a proxied request
                # r = req_proxy.generate_proxied_request(url, params=params, headers=headers, method="GET")

                # if r HTTP status code is success, get the html content
                if r.status_code == 200:
                    # print complete url query with parameters
                    soup = BeautifulSoup(r.text, "html.parser")
                else:
                    print(
                        f"Google direct scraping failed with error code: {r.status_code}"
                    )
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
                            proxies = requests.get(
                                "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=https&timeout=5000&country=all&ssl=yes&anonymity=elite"
                            ).text.split("\r\n")
                            proxy = random.choice(proxies)  # Get a random proxy
                            options.add_argument(f"--proxy-server={proxy}")

                        # Create a browser instance
                        driver = webdriver.Chrome(options=options)

                        # URL and parameters for Google scraping
                        url = (
                            "https://www.google.com/search?q="
                            + ("site:" + specific_site + " " if specific_site else "")
                            + query
                            + ("&fileType=pdf" if pdf_only else "")
                            + "&num="
                            + str(max_results)
                        )
                        driver.get(url)

                        # Wait until the search results are loaded
                        try:
                            element = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.ID, "search"))
                            )
                            soup = BeautifulSoup(driver.page_source, "html.parser")
                        except Exception as e:
                            # print error
                            error = str(e.message) if hasattr(e, "message") else str(e)
                            print(
                                f"Google SELENIUM scraping failed with error code: {error}"
                            )
                            soup = None
                        driver.quit()

            # Get all results
            results = []
            if soup is not None:
                for g in soup.find_all("div", class_="g"):
                    anchors = g.find_all("a")
                    if anchors:
                        link = anchors[0]["href"]
                        title = g.find("h3").text
                        spans = g.find_all("span")
                        description = ""
                        for span in spans:
                            # test if span contains <em> element or if span.text is a year date with 4 digits
                            if not span.find("em") and not (
                                (span.text.isdigit() and len(span.text) == 4)
                            ):
                                continue
                            if span.text.isdigit() and len(span.text) == 4:
                                description += "[" + span.text + "] "
                            else:
                                description += span.text + "\n"
                        # remove last \n if exists
                        if description.endswith("\n"):
                            description = description[:-1]
                        results.append(
                            {"link": link, "title": title, "description": description}
                        )

            print("Scraping search results: " + str(len(results)))

        if results is None or len(results) == 0:  # Google API search
            # Check google_api_key and google_cse_id are set as a global variable in the script
            from config import google_api_key, google_cse_id

            if not google_api_key or not google_cse_id:
                raise ValueError(
                    "Please set google_api_key and google_cse_id as global variables in the script."
                )

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

            url = "https://www.googleapis.com/customsearch/v1"
            current_results_count, results = 0, []
            while current_results_count < max_results:
                num_results = min(10, max_results - current_results_count)
                params = {
                    "key": google_api_key,
                    "cx": google_cse_id,
                    "q": query,
                    "num": num_results,
                    "siteSearch": specific_site if specific_site else None,
                    "fileType": "pdf" if pdf_only else None,
                    "start": current_results_count + 1,
                }
                r = requests.get(url, params=params)

                if output_format == "json":
                    items = r.json().get("items", [])
                    # filter html results to get only title, link and snippet for each result
                    if result_pretty:
                        items = [
                            {
                                "title": item["title"],
                                "link": item["link"],
                                "description": item["snippet"],
                            }
                            for item in items
                        ]
                    results.extend(items)
                elif output_format == "html":
                    results.append(r.text)
                else:
                    raise ValueError("Invalid output format.")
                current_results_count += num_results

            print("Google API search results: " + str(len(results)))

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
                import PyPDF2

                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text

    @staticmethod
    @method_call_counter
    def search_fiche_anomalie(
        query, directory_path, output_format="json", max_results=20
    ):
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
                text = SynthesisManager.extract_text_from_pdf(pdf_path)
                if query.lower() in text.lower():
                    results.append({"filename": filename, "content": text})
                if len(results) >= max_results:
                    break

        if output_format == "json":
            return json.dumps(results, indent=2)
        else:
            return "\n".join(
                [
                    "Filename: {}\nContent: {}\n".format(
                        result["filename"], result["content"]
                    )
                    for result in results
                ]
            )

    # OK: based on search_google
    @staticmethod
    @method_call_counter
    def search_pubmed(
        query, output_format="json", max_results=10, pdf_only=False, scraping=True
    ):
        if pdf_only:
            print("Pubmed does not support pdf only search")
        return SynthesisManager.search_google(
            query=query,
            output_format=output_format,
            pdf_only=False,
            max_results=max_results,
            try_direct_scraping_first=scraping,
            specific_site="pubmed.ncbi.nlm.nih.gov",
            result_pretty=True,
        )

    # OK: based on search_google
    @staticmethod
    @method_call_counter
    def search_core(
        query, output_format="json", max_results=10, pdf_only=False, scraping=True
    ):
        return SynthesisManager.search_google(
            query=query,
            output_format=output_format,
            pdf_only=pdf_only,
            max_results=max_results,
            try_direct_scraping_first=scraping,
            specific_site="core.ac.uk",
            result_pretty=True,
        )

    # OK: based on core Wikipedia search
    @staticmethod
    @method_call_counter
    def search_wikipedia(query, output_format="json", max_results=10):
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
        }
        response = requests.get(base_url, params=params)
        search_results = response.json().get("query", {}).get("search", [])

        if output_format == "json":
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "title": result.get("title", ""),
                    "link": "https://en.wikipedia.org/wiki/"
                    + result.get("title", "").replace(" ", "_"),
                    "description": result.get("snippet", ""),
                }
                formatted_results.append(formatted_result)

            return formatted_results
        else:
            return {"error": "Output format not supported"}

    # add event using the document object add_event method add_event
    @method_call_counter
    def add_event(self, event: str, data: Dict[str, Any] = None):
        self.document.add_event(event, data)

    def normalized_cosine_similarity(
        self, a: List[float], b: List[float], min_cs: float = None
    ) -> float:
        if min_cs is None:
            min_cs = self.min_cosine_similarity
        return (cosine_similarity([a], [b])[0][0] - min_cs) / (1 - min_cs)

    @method_call_counter
    def add_section(self, section: Section):
        if self.validate_section_format(
            asdict(section)
        ):  # Convert dataclass to dict for validation
            self.document.document_content.sections_list.append(section)
            self.document.update_sections_embeddings(
                [section.section_id], skip_plan_embeddings_update=True
            )
            self.plan_embedding_update_required = True
            self.document.add_event(
                {"action": "add_section", "section_id": section.section_id}
            )
        else:
            print("Invalid section format.")
        return self

    @method_call_counter
    def create_and_add_section_then_return_id(
        self,
        title: str,
        content: str,
        section_id: int = None,
        parent_id: int = None,
        resources: list | str | int = None,
    ) -> int:
        if not section_id:
            # Generate section_id by using max section_id + 1
            section_id = (
                (
                    max(
                        [
                            s.section_id
                            for s in self.document.document_content.sections_list
                        ]
                    )
                    + 1
                )
                if len(self.document.document_content.sections_list) > 0
                else 1
            )

        self.add_section(
            Section(
                section_id=section_id,
                parent_id=parent_id,
                title=title,
                content=content,
                resources=resources,
            )
        )
        return section_id

    @method_call_counter
    def get_sections(self, ids: List[int]) -> List[Section]:
        return [
            s
            for s in self.document.document_content.sections_list
            if s.section_id in ids
        ]

    @method_call_counter
    def get_all_sections(self) -> List[Section]:
        return self.document.document_content.sections_list

    @method_call_counter
    def remove_section(self, section_id: int) -> bool:
        section = next(
            (
                s
                for s in self.document.document_content.sections_list
                if s.section_id == section_id
            ),
            None,
        )
        if section:
            self.document.document_content.sections_list = [
                s
                for s in self.document.document_content.sections_list
                if s.section_id != section_id
            ]
            self.requires_update_plan_embedding()
            self.document.add_event(
                {"action": "remove_section", "section_id": section_id}
            )
            return True
        else:
            return False

    @method_call_counter
    def edit_section(
        self,
        section_id: int,
        new_content: str = None,
        new_title: str = None,
        new_resources: list | str | int = None,
        new_parent_id: int = None,
    ) -> bool:
        section = next(
            (
                s
                for s in self.document.document_content.sections_list
                if s.section_id == section_id
            ),
            None,
        )
        if section:
            action_event = {"action": "edit_section", "section_id": section_id}
            update_embeddings = False
            if new_content:
                section.content = new_content
                update_embeddings = True
                action_event["new_content"] = new_content
            if new_title:
                section.title = new_title
                update_embeddings = True
                action_event["new_title"] = new_title
            if new_resources:
                if not isinstance(new_resources, list):
                    new_resources = [str(new_resources)]
                section.resources = [str(resource) for resource in new_resources]
                action_event["new_resources"] = new_resources
            if new_parent_id:
                section.parent_id = new_parent_id
                action_event["new_parent_id"] = new_parent_id
            self.document.add_event("observation", action_event)
            if update_embeddings:
                self.document.update_sections_embeddings(
                    [section_id], skip_plan_embeddings_update=True
                )
                self.plan_embedding_update_required = True
            return True
        else:
            return False

    @method_call_counter
    def swap_sections(self, section_id_1: int, section_id_2: int) -> bool:
        section_1 = next(
            (
                s
                for s in self.document.document_content.sections_list
                if s.section_id == section_id_1
            ),
            None,
        )
        section_2 = next(
            (
                s
                for s in self.document.document_content.sections_list
                if s.section_id == section_id_2
            ),
            None,
        )
        if section_1 and section_2:
            section_1_index = self.document.document_content.sections_list.index(
                section_1
            )
            section_2_index = self.document.document_content.sections_list.index(
                section_2
            )
            (
                self.document.document_content.sections_list[section_1_index],
                self.document.document_content.sections_list[section_2_index],
            ) = (
                self.document.document_content.sections_list[section_2_index],
                self.document.document_content.sections_list[section_1_index],
            )
            self.document.add_event(
                "observation",
                {
                    "action": "swap_sections",
                    "section_id_1": section_id_1,
                    "section_id_2": section_id_2,
                },
            )
            return True
        else:
            return False

    # search into resources stored in self.document.resources_vectordb and self.document.resources, return a list of resources
    @method_call_counter
    def semantic_search_resources(
        self,
        query_embeddings=None,
        query_texts=None,
        n_results=10,
        where=None,
        where_document=None,
        include=["metadatas", "documents", "distances"],
    ):
        result = self.document.resources_vectordb._similarity_search_with_score(
            query_embeddings, k=n_results
        )

    @method_call_counter
    def get_all_resources(self) -> List[Dict[str, Any]]:
        return self.document.resources

    @method_call_counter
    def add_or_update_results_in_resources(
        self, results, metadatas_to_add=None, store_linked_document_content=False
    ):
        results = [results] if isinstance(results, dict) else results or []

        for result in results:
            content = (
                {"description": str(result.get("description", result)).strip()}
                if isinstance(result, dict)
                else {"description": str(result)}
            )
            self.add_or_update_result_in_resources(
                metadatas=metadatas_to_add or {},
                name=(
                    str(result.get("title", "")).strip()
                    if isinstance(result, dict)
                    else ""
                ),
                link=(
                    str(result.get("link", "")).strip()
                    if isinstance(result, dict)
                    else ""
                ),
                content=content,
                store_linked_document_content=store_linked_document_content,
            )
        self.plan_embedding_update_required = True
        return self

    @method_call_counter
    def add_or_update_result_in_resources(
        self,
        metadatas: dict,
        name: str = None,
        content: dict = None,
        link: str = None,
        store_linked_document_content: bool = False,
        chaining: bool = True,
    ):
        # Move metadatas to content if content data were provided into metadatas
        if metadatas.get("title") and not name:
            name = metadatas.get("title")
            metadatas.pop("title")
        if (
            metadatas.get("link", metadatas.get("source", metadatas.get("url")))
            and not link
        ):
            link = metadatas.get("link")
            metadatas.pop("link")
        if metadatas.get("description") and not content:
            content = metadatas.get("description", "")
            metadatas.pop("description")
        resource_embedding_label = "resource_embedding_" + (
            "1"
            if self.document.embedding_model_name == "text-embedding-ada-002"
            else "2"
        )

        # assert name or link are provided to identify the resource
        if not name and not link:
            raise ValueError("Either name or link must be provided")

        # Generate id using max
        id = (
            max([r["id"] for r in self.document.resources]) + 1
            if len(self.document.resources) > 0
            else 1
        )
        document = {
            "name": name,
            "link": link,
            "content": str(content),
            "description": str(
                content.get("description", "") if isinstance(content, dict) else content
            ),
        }
        embedding = self.document.embedding_model.embed_query(
            str(content)
            + (("\n" + str(name)) if name else "")
            + (("\n" + link) if link else "")
        )

        # Check for existing document
        existing_doc = next(
            (
                doc
                for doc in self.document.resources
                if (
                    doc["document"]["name"] == name
                    or (link and doc["document"]["link"] == link)
                )
            ),
            None,
        )

        if existing_doc:
            # Update the existing document
            updated_fields = []
            for key, value in document.items():
                if value and existing_doc["document"].get(key) != value:
                    existing_doc["document"][key] = value
                    updated_fields.append(key)

            # Log the event
            if updated_fields:
                # Update resources_vectordb
                self.document.resources_vectordb._add_texts(
                    [str(document)],
                    metadatas=[metadatas],
                    ids=[str(existing_doc["id"])],
                )
                self.document.add_event(
                    "observation",
                    {
                        "action": "modify_resource",
                        "document_name": name,
                        "updated_fields": updated_fields,
                    },
                )
                # print(f"Resource updated fields: ", updated_fields)
            # else:
            # self.document.add_event('observation', {'action': 'modify_resource', 'document_name': name, 'message': 'No fields were updated'})
            # print(f"Resource no fields updated")
        else:
            # Add new document
            self.document.resources.append(
                {
                    "id": id,
                    "metadatas": metadatas,
                    "document": document,
                    resource_embedding_label: embedding,
                }
            )
            if store_linked_document_content:
                childs_ids_list = self.get_and_store_link_content(
                    link=link, parent_id=id, chaining=False
                )
                metadatas["childs_ids_list"] = childs_ids_list
            self.document.resources_vectordb._add_texts(
                [str(document)], metadatas=[metadatas], ids=[str(id)]
            )
            self.document.add_event(
                "observation", {"action": "add_resource", "document_name": name}
            )
            # print(f"Resource added - VectorDB collection count: {self.document.resources_vectordb.count()}")

        self.plan_embedding_update_required = True
        return (
            self
            if chaining
            else (existing_doc if existing_doc else self.document.resources[-1])
        )

    @method_call_counter
    def get_and_store_link_content(
        self, link: str = None, parent_id=None, chaining: bool = True
    ):
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
                doc.metadata.extend([{"parent_id": parent_id}])
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
            self.document.resources = [
                r for r in self.document.resources if r["id"] not in resource_id
            ]
        elif isinstance(resource_id, int):
            self.document.resources = [
                r for r in self.document.resources if r["id"] != resource_id
            ]
        self.document.add_event(
            "observation",
            {"action": "remove_resources", "resource_id": str(resource_id)},
        )
        self.plan_embedding_update_required = True
        return self

    @method_call_counter
    def remove_resources(self, resource_ids: List[int]):
        return self.remove_resource(resource_ids)

    def restore_last_state(self):
        return self.document.restore_state()

    @method_call_counter
    def list_all_previous_document_events(self) -> List[Any]:
        return self.document.events

    def set_targetJSON_comparison(
        self,
        file_path: str,
        target_section_title_embedding_label: str = "section_embedding_2",
        target_section_content_embedding_label: str = "content_embedding_2",
        target_plan_embedding_label: str = "plan_embedding_2",
        target_resource_embedding_label: str = "resource_embedding_2",
        normalize_embeddings: bool = True,
        min_cosine_similarity: float = None,
    ):
        self.target_file_path = file_path

        with open(file_path, "r") as f:
            self.target_data = json.load(f)
        output_check = ""
        for section in self.target_data["plan"]:
            output_check += section["section"] + " /"
        print(output_check)
        # Compute the total length for the target data (similar to the test method)
        self.target_total_content_length = sum(
            len(section["content"]) for section in self.target_data["plan"]
        )
        self.target_total_sections_count = len(self.target_data["plan"])

        self.target_plan_titles_embedding = np.mean(
            [
                section[target_section_title_embedding_label]
                for section in self.target_data["plan"]
            ],
            axis=0,
        )
        self.target_plan_contents_embedding = np.mean(
            [
                section[target_section_content_embedding_label]
                for section in self.target_data["plan"]
            ],
            axis=0,
        )
        self.target_plan_embedding = self.target_data[target_plan_embedding_label]

        embed_len = len(self.document.dumb_embedding)
        self.target_total_resources_count = len(self.target_data["resources"])
        self.target_plan_resources_embedding = (
            np.mean(
                [
                    r[target_resource_embedding_label]
                    for r in self.target_data["resources"]
                ],
                axis=0,
            )
            if len(self.target_data["resources"]) > 0
            else np.zeros((embed_len,))
        )

        if normalize_embeddings:
            if min_cosine_similarity is None:
                dumb_embedding = self.document.dumb_embedding
                self.min_plan_titles_cosine_similarity = cosine_similarity(
                    [dumb_embedding], [self.target_plan_titles_embedding]
                )[0][0]
                self.min_plan_contents_cosine_similarity = cosine_similarity(
                    [dumb_embedding], [self.target_plan_contents_embedding]
                )[0][0]
                self.min_plan_cosine_similarity = cosine_similarity(
                    [dumb_embedding], [self.target_plan_embedding]
                )[0][0]
                self.min_plan_resources_cosine_similarity = cosine_similarity(
                    [dumb_embedding], [self.target_plan_resources_embedding]
                )[0][0]
            else:
                self.min_plan_titles_cosine_similarity = (
                    self.min_plan_contents_cosine_similarity
                ) = self.min_plan_cosine_similarity = (
                    self.min_plan_resources_cosine_similarity
                ) = min_cosine_similarity
        else:
            self.min_plan_titles_cosine_similarity = (
                self.min_plan_contents_cosine_similarity
            ) = self.min_plan_cosine_similarity = (
                self.min_plan_resources_cosine_similarity
            ) = 0

    def get_distance_to_targetJSON(
        self,
        target_section_title_embedding_label: str = "section_embedding_2",
        target_section_content_embedding_label: str = "content_embedding_2",
        target_plan_embedding_label: str = "plan_embedding_2",
        target_resource_embedding_label: str = "resource_embedding_2",
        get_progress: bool = False,
    ):
        # if self does not have target_file_path
        if not hasattr(self, "target_file_path"):
            raise ValueError(
                "Please set target_file_path using set_targetJSON_comparison method"
            )

        if not hasattr(self, "target_data"):
            self.set_targetJSON_comparison(
                self.target_file_path,
                target_section_title_embedding_label=target_section_title_embedding_label,
                target_section_content_embedding_label=target_section_content_embedding_label,
                target_plan_embedding_label=target_plan_embedding_label,
                target_resource_embedding_label=target_resource_embedding_label,
            )
            self.document.update_plan_embedding()
        elif self.plan_embedding_update_required or not hasattr(
            self.document.document_content, "sections_list_title_embedding"
        ):
            self.document.update_plan_embedding()

        # Compute current document metrics
        current_sections_count = len(self.document.document_content.sections_list)
        current_plan_non_empty_sections_content_count = sum(
            1
            for section in self.document.document_content.sections_list
            if section.content and len(section.content) > 1
        )
        current_plan_non_empty_sections_title_count = sum(
            1
            for section in self.document.document_content.sections_list
            if section.title and len(section.title) > 1
        )
        current_content_length = sum(
            len(getattr(section, "content", ""))
            for section in self.document.document_content.sections_list
        )

        # Get embeddings
        plan_embedding = self.document.document_content.sections_list_embedding
        plan_titles_embedding = (
            self.document.document_content.sections_list_title_embedding
        )
        plan_contents_embedding = (
            self.document.document_content.sections_list_content_embedding
        )
        plan_resources_embedding = self.document.document_content.resources_embedding

        # Compute embedding similarity
        plan_embedding_similarity = self.normalized_cosine_similarity(
            plan_embedding, self.target_plan_embedding, self.min_plan_cosine_similarity
        )
        plan_titles_embedding_similarity = self.normalized_cosine_similarity(
            plan_titles_embedding,
            self.target_plan_titles_embedding,
            self.min_plan_titles_cosine_similarity,
        )
        plan_contents_embedding_similarity = self.normalized_cosine_similarity(
            plan_contents_embedding,
            self.target_plan_contents_embedding,
            self.min_plan_contents_cosine_similarity,
        )
        plan_resources_embedding_similarity = self.normalized_cosine_similarity(
            plan_resources_embedding,
            self.target_plan_resources_embedding,
            self.min_plan_resources_cosine_similarity,
        )

        # Refined ratio calculations
        content_length_ratio_to_target = round(
            min(
                current_content_length / (self.target_total_content_length + 1e-5), 1.5
            ),
            3,
        )
        sections_count_ratio_to_target = round(
            min(
                current_sections_count / (self.target_total_sections_count + 1e-5), 1.5
            ),
            3,
        )
        sections_content_non_empty_count_ratio_to_target = round(
            min(
                current_plan_non_empty_sections_content_count
                / (self.target_total_sections_count + 1e-5),
                1.5,
            ),
            3,
        )
        sections_title_non_empty_count_ratio_to_target = round(
            min(
                current_plan_non_empty_sections_title_count
                / (self.target_total_sections_count + 1e-5),
                1.5,
            ),
            3,
        )
        # count number of resources from self.document.resources list which are cited/resources_used in self.document.document_content.sections_list
        cited_resources_count = sum(
            1
            for resource in self.document.resources
            if any(
                resource["id"] in section.resources
                for section in self.document.document_content.sections_list
                if section.resources and len(section.resources) > 0
            )
        )
        resources_count_ratio_to_target = round(
            min(
                cited_resources_count / (self.target_total_resources_count + 1e-5), 1.5
            ),
            3,
        )

        sections_with_citations = sum(
            1
            for section in self.document.document_content.sections_list
            if section.resources and len(section.resources) > 0
        )
        # Count similarly the sum in target_data
        target_sections_with_citations = sum(
            1
            for section in self.target_data["plan"]
            if section["resources_used"] and len(section["resources_used"]) > 0
        )
        resources_citation_coverage_score = (
            sections_with_citations / target_sections_with_citations
            if target_sections_with_citations > 0
            else 0.0
        )

        # Build the result dictionary
        distance_to_targetJSON = {
            "plan_embedding_similarity": round(plan_embedding_similarity, 3),
            "plan_titles_embedding_similarity": round(
                plan_titles_embedding_similarity, 3
            ),
            "plan_contents_embedding_similarity": round(
                plan_contents_embedding_similarity, 3
            ),
            "plan_resources_embedding_similarity": round(
                plan_resources_embedding_similarity, 3
            ),
            "current_sections_count": current_sections_count,
            "sections_count_ratio_to_target": sections_count_ratio_to_target,
            "title_non_empty_count_ratio_to_target": sections_title_non_empty_count_ratio_to_target,
            # "current_content_length": current_content_length,
            "content_length_ratio_to_target": content_length_ratio_to_target,
            "content_non_empty_count_ratio_to_target": sections_content_non_empty_count_ratio_to_target,
            "resources_citation_coverage_score": round(
                resources_citation_coverage_score, 3
            ),
            "resources_count_ratio_to_target": resources_count_ratio_to_target,
            "resources_count": len(self.document.resources),
        }

        # Progress comparison (optional)
        if get_progress:

            def get_ratio(previous_value, current_value):
                return (
                    round(
                        (previous_value - current_value)
                        / (previous_value + 0.0000001)
                        * 100,
                        2,
                    )
                    if previous_value
                    else 0
                )

            if hasattr(self, "distance_to_targetJSON"):
                distance_to_targetJSON["plan_embedding_similarity_progress"] = (
                    get_ratio(
                        plan_embedding_similarity,
                        self.distance_to_targetJSON["plan_embedding_similarity"],
                    )
                )
                distance_to_targetJSON["plan_titles_embedding_similarity_progress"] = (
                    get_ratio(
                        plan_titles_embedding_similarity,
                        self.distance_to_targetJSON["plan_titles_embedding_similarity"],
                    )
                )
                distance_to_targetJSON[
                    "plan_contents_embedding_similarity_progress"
                ] = get_ratio(
                    plan_contents_embedding_similarity,
                    self.distance_to_targetJSON["plan_contents_embedding_similarity"],
                )
                distance_to_targetJSON[
                    "plan_resources_embedding_similarity_progress"
                ] = get_ratio(
                    plan_resources_embedding_similarity,
                    self.distance_to_targetJSON["plan_resources_embedding_similarity"],
                )
                distance_to_targetJSON["sections_count_ratio_to_target_progress"] = (
                    get_ratio(
                        sections_count_ratio_to_target,
                        self.distance_to_targetJSON["sections_count_ratio_to_target"],
                    )
                )
                distance_to_targetJSON[
                    "title_non_empty_count_ratio_to_target_progress"
                ] = get_ratio(
                    sections_title_non_empty_count_ratio_to_target,
                    self.distance_to_targetJSON[
                        "title_non_empty_count_ratio_to_target"
                    ],
                )
                distance_to_targetJSON["content_length_ratio_to_target_progress"] = (
                    get_ratio(
                        content_length_ratio_to_target,
                        self.distance_to_targetJSON["content_length_ratio_to_target"],
                    )
                )
                distance_to_targetJSON[
                    "content_non_empty_count_ratio_to_target_progress"
                ] = get_ratio(
                    sections_content_non_empty_count_ratio_to_target,
                    self.distance_to_targetJSON[
                        "content_non_empty_count_ratio_to_target"
                    ],
                )
                distance_to_targetJSON["resources_citation_coverage_score_progress"] = (
                    get_ratio(
                        resources_citation_coverage_score,
                        self.distance_to_targetJSON[
                            "resources_citation_coverage_score"
                        ],
                    )
                )
                distance_to_targetJSON["resources_count_ratio_to_target_progress"] = (
                    get_ratio(
                        resources_count_ratio_to_target,
                        self.distance_to_targetJSON["resources_count_ratio_to_target"],
                    )
                )

        # Save the results
        self.distance_to_targetJSON = distance_to_targetJSON

        return self.distance_to_targetJSON

    # return the list of current sections with title, length of content, validation status, and feedback
    def get_plan_status(
        self,
        compact_string_format: bool = False,
        keys=["section_id", "title", "content_length", "content_preview[:100]"],
    ):
        # keys = ["section_id", "title", "content_length", "validation_status", "feedback_to_process", "feedback_processed"]
        plan_status = []
        for section in self.document.document_content.sections_list:
            status_data_full = [
                section.section_id,
                section.title,
                len(section.content) if section.content is not None else 0,
                section.content[:100] if section.content is not None else 0,
                # round(section.content_progress_validation_status, 1),
                # section.local_feedback_to_process,
                # section.local_feedback_processed,
            ]
            status_data = [data for key, data in zip(keys, status_data_full)]

            if compact_string_format:
                plan_status.append("|".join(map(str, status_data)))
            else:
                plan_status.append(dict(zip(keys, status_data)))

        if len(plan_status) == 0:
            return []
        if compact_string_format:
            header = "|".join(keys)
        else:
            header = [
                "section_id",
                "title",
                "content_length",
                "content_preview[:100]",
            ]  # , "validation_status", "feedback_to_process", "feedback_processed"]
        plan_status.insert(0, header)

        return plan_status

    def get_resources_status(self, compact_string_format: bool = False):
        resources_status = []
        content_info = {}

        for resource in self.document.resources:
            # Vérifier que resource est un dict et contient la clé 'document'
            if (
                isinstance(resource, dict)
                and "document" in resource
                and isinstance(resource["document"], dict)
            ):
                # Vérifier que le document contient 'content' et que c'est un dict
                if "content" in resource["document"] and isinstance(
                    resource["document"]["content"], dict
                ):
                    for key, value in resource["document"]["content"].items():
                        content_info[f"len(content['{key}'])"] = len(str(value))

                # Vérification des autres éléments
                status_data = [
                    resource.get(
                        "id", "unknown"
                    ),  # Utiliser 'unknown' si id n'est pas disponible
                    resource.get("metadatas", {}).get(
                        "search", "unknown"
                    ),  # Vérifier que 'metadatas' est un dict
                    resource["document"].get(
                        "name", "unknown"
                    ),  # Utiliser 'unknown' si name n'est pas disponible
                    (
                        len(resource["document"].get("link", []))
                        if isinstance(
                            resource["document"].get("link"), (list, tuple, np.ndarray)
                        )
                        else 0
                    ),  # Vérifier le type de 'link'
                    *content_info.values(),  # Ajouter les longueurs de contenu
                ]

                # Formater la sortie selon compact_string_format
                if compact_string_format:
                    resources_status.append("|".join(map(str, status_data)))
                else:
                    keys = [
                        "id",
                        "metadatas",
                        "document_name",
                        "document_link_length",
                    ] + list(content_info.keys())
                    resources_status.append(dict(zip(keys, status_data)))

        if compact_string_format:
            # if content_info is empty, it means that there is no resource in the document
            if len(content_info) == 0:
                return []
            header = "id|metadatas|document_name|document_link_length|" + "|".join(
                content_info.keys()
            )
            resources_status.insert(0, header)

        return resources_status

    def get_count_method_calls(self):
        return self._method_counts if hasattr(self, "_method_counts") else {}

    def reset_method_calls_counters(self):
        if hasattr(self, "_method_counts"):
            self._method_counts = {}

    def GetFromLatex(
        self, latex_string: str, bib_file: str = None, debug: bool = False
    ) -> None:
        """
        Parses a complete LaTeX document from 'latex_string' using TexSoup and updates the
        technical synthesis document (self.document) with the extracted title, abstract,
        sections, and bibliography.

        If 'bib_file' is provided, the function extracts additional bibliography entries
        and merges them with self.document.resources.
        """
        from TexSoup import TexSoup
        import uuid
        from datetime import datetime
        import os
        import re

        try:
            import bibtexparser
        except ImportError:
            raise ImportError(
                "Please install bibtexparser with `pip install bibtexparser`."
            )

        if "```latex" in latex_string:
            m = re.search(r"```latex\s*(.*?)```", latex_string, re.I | re.S)
            if m and any(
                t in m.group(1) for t in ("\\documentclass", "\\begin{document}")
            ):
                latex_string = m.group(1).strip()

        # --- Helper to process both internal and external BibTeX entries ---
        def process_bib_entry(entry, default_key=None):
            bib_id = entry.get("ID", default_key or str(uuid.uuid4()))
            entry_title = entry.get("title", bib_id)
            if entry.get("author"):
                entry_title += f"; Author: {entry['author']}"
            if entry.get("year"):
                entry_title += f"; Year: {entry['year']}"
            link = entry.get("url", "").strip() or (
                f"https://doi.org/{entry.get('doi', '').strip()}"
                if "doi" in entry
                else ""
            )
            description = entry.get(
                "abstract", entry.get("note", entry.get("comment", ""))
            ).strip()
            db = bibtexparser.bibdatabase.BibDatabase()
            db.entries = [entry]
            entry_text = bibtexparser.dumps(db).strip()
            # entry_text = bibtexparser.dumps(bibtexparser.bibdatabase.BibDatabase(entries=[entry])).strip()
            content_hash = hash(entry_text)

            existing = next(
                (r for r in self.document.resources if r.get("id") == bib_id), None
            )
            if existing:
                print(
                    f"Updating of existing bib resource entry not implemented yet (bib_id: {bib_id})"
                )
            else:
                self.document.resources.append(
                    {
                        "id": bib_id,
                        "key": bib_id,
                        "name": entry_title,
                        "content": entry_text,
                        "link": link,
                        "description": description,
                    }
                )
                self.document.section_hashes[bib_id] = content_hash

        def latex_extract_citations(text, references):
            citations = re.findall(r"\\cite[t|p]*\{([^}]+)\}", text)
            all_keys = []
            for cite in citations:
                keys = [key.strip().lower() for key in cite.split(",")]
                all_keys.extend(keys)
            if references and len(references):
                return list(
                    {citation for citation in all_keys if citation in references}
                )
            else:
                return all_keys

        def safe_string(tag):
            try:
                return tag.string.strip()
            except (AssertionError, AttributeError):
                return " ".join(str(child).strip() for child in tag.contents).strip()

        def get_full_section_content(tag, section_tags, raw_latex):
            content = []
            # Attempt to get the next_elements attribute and default to [] if it is None.
            siblings = tag.next_elements
            if siblings is None:
                siblings = []
            for element in siblings:
                # If we encounter another section tag, stop processing.
                if hasattr(element, "name") and element.name in section_tags:
                    break
                # Only process text nodes.
                if isinstance(element, str):
                    stripped = element.strip()
                    if stripped:
                        content.append(stripped)
            extracted = " ".join(content).strip()
            if extracted:
                return extracted
            else:
                # Fallback: use regex on the raw LaTeX.
                from re import compile, escape, DOTALL

                title = safe_string(tag)
                title_esc = escape(title)
                # Build a regex that matches the section command with this title and captures subsequent content.
                pattern_str = (
                    r"\\(?:"
                    + "|".join(section_tags)
                    + r")\*?\{\s*"
                    + title_esc
                    + r"\s*\}(?P<content>.*?)(?=\\(?:"
                    + "|".join(section_tags)
                    + r")\*?\{|\\end\{document\})"
                )
                pattern = compile(pattern_str, DOTALL)
                match = pattern.search(raw_latex)
                if match:
                    return match.group("content").strip()
                else:
                    # Provide detailed debug information.
                    if debug:
                        print(
                            f"DEBUG: Regex fallback did not find content for section with title: '{title}'."
                        )
                        print(f"DEBUG: Regex pattern used: {pattern.pattern}")
                        if hasattr(tag, "position"):
                            pos = tag.position
                            snippet = raw_latex[max(0, pos - 50) : pos + 400]
                            print(
                                f"DEBUG: Raw LaTeX snippet around tag position: {snippet}"
                            )
                    return ""

        # --- Load LaTeX string ---
        if os.path.exists(latex_string):
            with open(latex_string, "r", encoding="utf-8") as f:
                latex_string = f.read()
        try:
            soup = TexSoup(latex_string)
        except Exception as e:
            raise ValueError(f"Failed to parse LaTeX string: {e}")

        # --- Process \bibitem as pseudo BibTeX ---
        for bib in soup.find_all("bibitem"):
            note = "".join(str(child).strip() for child in bib.contents).strip()
            # Check if bib.attrs exists before trying to access it
            if not hasattr(bib, "attrs") or bib.attrs is None:
                bib_attrs = {}
            else:
                bib_attrs = bib.attrs

            title = bib_attrs.get("title", f"Resource {bib_attrs.get('id', '')}")
            key = bib_attrs.get("id", str(uuid.uuid4()))
            fake_bibtex = (
                f"@misc{{{key},\n  title = {{{title}}},\n  note = {{{note}}}\n}}"
            )
            try:
                bib_entry = bibtexparser.loads(fake_bibtex).entries[0]
                process_bib_entry(bib_entry, default_key=key)
            except Exception:
                continue  # silently skip malformed \bibitem

        # --- Process External .bib File ---
        if bib_file:
            bib_text = bib_file
            if os.path.exists(bib_file):
                with open(bib_file, "r", encoding="utf-8") as f:
                    bib_text = f.read()
            try:
                entries = bibtexparser.loads(bib_text).entries
            except Exception as e:
                print(f"Failed to load bib: {e}\nbib_text:{bib_text}")
            for entry in entries:
                try:
                    process_bib_entry(entry)
                except Exception as e:
                    print(f"Failed to process bib entry: {e}\nbib_entry:{entry}")
                    continue

        all_ref_keys = {
            ref.get("key", "").lower()
            for ref in self.document.resources
            if "key" in ref
        }

        # --- Title & Abstract ---
        title_node = soup.find("title")
        if debug:
            print(f"DEBUG: Title node string: {title_node.string}")
        if title_node and title_node.string:
            self.document.title = title_node.string.strip()
            self.document.set_plan_field_with_embedding("title", self.document.title)

        try:
            abstract_node = soup.find("abstract") or soup.find(
                lambda t: getattr(t, "name", None) == "abstract"
                and getattr(t, "string", None)
            )
            abstract = (getattr(abstract_node, "string", "") or "").strip()
        except Exception as e:
            print(f"Failed to extract abstract: {e}")
            abstract, abstract_node = "", None
        if debug:
            print(f"DEBUG: abstract → {abstract!r}")
        if abstract:
            self.document.context = abstract
            self.document.set_plan_field_with_embedding("context", abstract)

        # --- Sections ---
        self.document.document_content.sections_list.clear()
        section_tags = ["section", "subsection", "subsubsection"]

        for tag in soup.find_all(section_tags):
            title = safe_string(tag) or "Untitled Section"
            try:
                sec_id = int(tag.attrs.get("id", uuid.uuid4().int >> 64))
            except Exception:
                sec_id = uuid.uuid4().int >> 64
            content = get_full_section_content(tag, section_tags, latex_string)
            if title == "Untitled Section" and content == "":
                print(
                    f"Warning: No title or content found for section. Skipping this section."
                )
                continue
            if not content:
                print(
                    f"Warning: No content found for section {title} using get_full_section_content. Fall-back to minimal safe_string."
                )
                content = safe_string(tag)

            citations = latex_extract_citations(content, all_ref_keys)

            section = Section(
                section_id=sec_id,
                parent_id=0,
                title=title,
                content=content,
                title_embedding=self.document.get_embedding(title),
                content_embedding=self.document.get_embedding(content),
                resources=citations,
            )
            self.add_section(section)
            if debug:
                print(
                    f"DEBUG: Added section with Title:<<<{title}>>>\nID:{sec_id}, Content:<<<{content}>>>"
                )

        used_keys = {
            key
            for section in self.document.document_content.sections_list
            for key in section.resources
        }
        self.document.resources = [
            r for r in self.document.resources if r.get("key", "").lower() in used_keys
        ]
        self.plan_embedding_update_required = True
        self.last_sync_time = datetime.now()

    def GetFromMarkdown(
        self,
        markdown_string: str,
        extract_references: bool = True,
        debug: bool = False,
        title: str = None,
        context: str = None,
        skip_sections=("contents", "references", "external links"),
    ) -> None:
        """
        Parses a markdown string to extract title, context, sections, and resources without HTML conversion.
          - title (from frontmatter or first '# ')
          - context (from frontmatter or first paragraph before any '##')
          - resources (inline and reference links)
          - sections (headings '##'–'######' with their content and used resources)
        """
        import re
        from uuid import uuid4

        def parse_references_section(lines, start_idx=0):
            """Parse the References section and return structured reference data"""
            references = {}
            ref_section_found = False

            for i, line in enumerate(lines[start_idx:], start_idx):
                # Detect References section
                if re.match(r"^#{1,6}\s*(references|external links|bibliography)\s*$", line, re.IGNORECASE):
                    ref_section_found = True
                    continue

                # Stop at next major section
                if (
                    ref_section_found
                    and re.match(r"^#{1,6}\s", line)
                    and not re.match(r"^#{1,6}\s*(references|external links|bibliography)\s*$", line, re.IGNORECASE)
                ):
                    break

                if ref_section_found and line.strip():
                    # Parse reference entries with multiple possible formats
                    patterns = [
                        r'^(\d+)\..+\["(.*)"\][^\(]*\(([^\)]*)\)\.[^\*]*\*+([^*]*)\*\.',
                        r'^(\d+)\..+\["(.*)"\][^\(]*\(([^\)]*)\)\.',
                        r'^(\d+)\..+\["(.*)"\](.+)$',
                        r'^(\d+)\.\s+\[(.*?)\]\((.*?)\)\s*-\s*(.+)$',
                        r'^\[(\d+)\]\s+\[(.*?)\]\((.*?)\)*$',
                        r'^\[(\d+)\]\s+\[(.*?)\]\((.*?)\)\s*$'
                    ]

                    for pattern in patterns:
                        match = re.match(pattern, line)
                        if match:
                            if len(match.groups()) == 4:  # Full reference with ID
                                ref_id, title, url, source = match.groups()
                                references[ref_id] = {
                                    "id": ref_id,
                                    "title": title.strip(),
                                    "url": url.strip(),
                                    "source": source.strip(),
                                    "line_number": i,
                                    "full_text": line,
                                }
                            elif len(match.groups()) == 3:  # Simple reference
                                ref_id, title, url_source = match.groups()
                                ref_id = str(len(references) + 1)
                                references[ref_id] = {
                                    "id": ref_id,
                                    "title": title.strip(),
                                    "url": (url_source.strip() if "http" in url_source else ""),
                                    "source": (url_source.strip() if "http" not in url_source else ""),
                                    "line_number": i,
                                    "full_text": line,
                                }
                            break

            return references

        def extract_citations_from_content(content, references):
            """Extract citation references from content and return matching reference IDs"""
            citations = []

            # Multiple citation patterns
            citation_patterns = [
                # r"\[\[(\d+)\]\]\(#cite[^\d]+\d+\)",  # [[..]](#cite_note-88)
                r"\[\[(\d+)\]\]",  # [[88]]
                # r"\(#cite[^\d]+(\d+)\)",  # (#cite_ref-88)
                # r"\[\^\]\(#cite[^\d]+(\d+)\)",  # [^](#cite_ref-88)
                r"\.\[(\d+)\]",  # .[88]
                r"\]\[(\d+)\]",  # ][88]
            ]

            for pattern in citation_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if match in references:
                        citations.append(match)

            return list(set(citations))  # Remove duplicates

        def add_citation_resource(ref_id, ref_data):
            """Add a citation reference as a resource"""
            if ref_id not in res_map:
                rid = str(uuid4())
                res_map[ref_id] = rid
                self.document.resources.append(
                    {
                        "id": rid,
                        "key": ref_id,
                        "name": ref_data["title"],
                        "link": ref_data["url"],
                        "content": "",
                        "description": f"{ref_data['title']} - {ref_data['source']}"
                    }
                )
            return res_map[ref_id]

        lines = markdown_string.splitlines()
        fm = {}
        idx = 0
        # --- YAML frontmatter ---
        if lines and lines[0].strip() == "---":
            for j, ln in enumerate(lines[1:], start=1):
                if ln.strip() == "---":
                    for fm_ln in lines[1:j]:
                        if ":" in fm_ln:
                            k, v = fm_ln.split(":", 1)
                            fm[k.strip()] = v.strip().strip("\"'")
                    idx = j + 1
                    break
        # --- Reference-style link definitions ---
        # ref_defs = {}
        # for ln in lines[idx:]:
        #     m = re.match(r"^\[([^]]+)\]:\s*(\S+)", ln)
        #     if m:
        #         ref_defs[m.group(1)] = m.group(2)
        # --- Parse References Section ---
        references = {}
        if extract_references:
            references = parse_references_section(lines, idx)
            if debug:
                print(f"Extracted {len(references)} references")
        # --- Resources initialization ---
        self.document.resources.clear()
        res_map = {}

        # --- Title ---
        title = title or fm.get("title", "")
        if not title:
            for ln in lines[idx:]:
                if ln.startswith("# "):
                    title = ln[2:].strip()
                    break
        if title:
            self.document.title = title
            self.document.set_plan_field_with_embedding("title", title)
        # --- Context ---
        context = context or fm.get("context", "")
        if not context:
            for ln in lines[idx:]:
                if re.match(r"^#{2,6}\s", ln):
                    break
                if ln.strip():
                    context += ln.strip() + " "
        if context:
            context = context.strip()
            self.document.context = context
            self.document.set_plan_field_with_embedding("context", context)
        # --- Section headings ---
        sec_heads = []  # list of (line_index, level, title)
        for i, ln in enumerate(lines[idx:], start=idx):
            m = re.match(r"^(#{2,6})\s*(.+)$", ln)
            if m:
                sec_heads.append((i, len(m.group(1)), m.group(2).strip()))
        # --- Extract sections ---
        for s_idx, (start, level, sec_title) in enumerate(sec_heads):
            if sec_title.lower() in skip_sections:
                continue
            end = sec_heads[s_idx + 1][0] if s_idx + 1 < len(sec_heads) else len(lines)
            content = " ".join(
                ln.strip() for ln in lines[start + 1 : end] if ln.strip()
            )
            used = []
            # Extract and process citations
            citations = extract_citations_from_content(content, references)
            for cite_id in citations:
                used.append(add_citation_resource(cite_id, references[cite_id]))

            # # unique
            used = list(dict.fromkeys(used))
            sec = Section(
                section_id=s_idx + 1,
                parent_id=0,
                title=sec_title,
                content=content,
                title_embedding=self.document.get_embedding(sec_title),
                content_embedding=self.document.get_embedding(content),
                resources=used,
            )
            self.add_section(sec)
        # --- Prune unused resources ---
        used_ids = {r_id for sec in self.document.document_content.sections_list for r_id in sec.resources}
        self.document.resources = [r for r in self.document.resources if r["id"] in used_ids]

        self.plan_embedding_update_required = True
        
        # dump document to temp file the document state
        if debug:
            print(f"Parsed title={title!r}, context={context!r}, sections={len(self.document.document_content.sections_list)}, resources={len(self.document.resources)}")
            self.document.update_plan_embedding()
            tmp = self.document
            # create a uuid with YmdHMS format
            uid = datetime.now().strftime("%Y%m%d%H%M%S")
            temp_state_file = f"/tmp/{uid}_document_state.json"
            with open(temp_state_file, "w", encoding="utf-8") as f: f.write(f"{self.document.get_state()}")
            print(f"Document state saved to {temp_state_file}")
            # dump markdown to temp file
            temp_markdown_file = f"/tmp/{uid}_markdown.md"
            with open(temp_markdown_file, "w", encoding="utf-8") as f: f.write(markdown_string)
            # copy solution to temp file
            temp_solution_file = f"/tmp/{uid}_solution.json"
            if self.target_file_path and os.path.exists(self.target_file_path):
                import shutil
                shutil.copy(self.target_file_path, temp_solution_file)
            print(f"Files saved: {temp_state_file}, {temp_markdown_file}, {temp_solution_file}")


class LLMResponse:
    def __init__(self, response):
        self.content = response.content  # Always access the .content property

    def __str__(self):
        return self.content  # Ensure it behaves like a string

    def __repr__(self):
        return f"LLMResponse(content={repr(self.content)})"

    def __getattr__(self, _):
        return self.content  # Any unexpected attribute access returns the content


class VoyagerEnvIR_CPS_TechSynthesis(Environment):
    def __init__(
        self,
        synthesis_type: str = "",
        goal: str = "",
        refined_goals: List[str] = None,
        # server_host='http://127.0.0.1', server_port=3000, request_timeout=600,
        log_path="./logs",
        CPS_env_type="techsynthesis",
        title: str = "",
        context: str = None,
        embedding_model_name: str = "intfloat/e5-base-v2",  # nomic-embed-text:latest, intfloat/e5-base-v2
        openai_api_key: str = None,
        target_file_path: str = None,
        llm=None,
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
        self.document = DocumentStructure(
            synthesis_type=synthesis_type,
            initial_goal=goal,
            refined_goals=self.refined_goals,
            embedding_model_name=embedding_model_name,
            title=title,
            context=context,
        )  # Initialize your document structure
        self.synthesis_manager = SynthesisManager(
            document=self.document, target_file_path=target_file_path
        )  # Initialize your Synthesis Manager
        # self.server = f"{server_host}:{server_port}" # TODO: voir si on a besoin d'un serveur type TGI pour les inférences
        # self.request_timeout = request_timeout
        self.log_path = log_path
        self.has_reset_once = False
        self.state = None  # Placeholder: initial state of the environment

    # llm static method
    @staticmethod
    def llm(prompt: str):
        return LLMResponse(
            VoyagerEnvIR_CPS_TechSynthesis.llm_model.invoke(
                [SystemMessage(content=""), HumanMessage(content=prompt)]
                if isinstance(prompt, str)
                else prompt
            )
        )

    def get_score(self):
        embed_id = (
            "1"
            if self.document.embedding_model_name == "text-embedding-ada-002"
            else "2"
        )
        distance = self.synthesis_manager.get_distance_to_targetJSON(
            target_section_title_embedding_label="section_embedding_" + embed_id,
            target_section_content_embedding_label="content_embedding_" + embed_id,
            target_plan_embedding_label="plan_embedding_" + embed_id,
            target_resource_embedding_label="resource_embedding_" + embed_id,
        )
    

        
        return {
            "plan/titles similarity (top:1, worst:0)": distance[
                "plan_titles_embedding_similarity"
            ],
            "sections contents similarity (top:1, worst:0)": distance[
                "plan_contents_embedding_similarity"
            ],
            "sections resources similarity (top:1, worst:0)": distance[
                "plan_resources_embedding_similarity"
            ],
            "sections count (top:1, <1:too short, >1:too long)": distance[
                "sections_count_ratio_to_target"
            ],
            "titles count (top:1, <1:too short, >1:too long)": distance[
                "title_non_empty_count_ratio_to_target"
            ],
            "sections contents length (top:1, <1:too short, >1:too long)": distance[
                "content_length_ratio_to_target"
            ],
            "sections contents non-empty (top:1, <1:too short, >1:too long)": distance[
                "content_non_empty_count_ratio_to_target"
            ],
            "resources citation coverage score (top:1, <1:too short, >1:too long)": distance[
                "resources_citation_coverage_score"
            ],
            "resources count (top:1, <1:too short, >1:too long)": distance[
                "resources_count_ratio_to_target"
            ],
        }

    def reset(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.has_reset_once = True
        super().reset()
        self.document.reset()  # Reset document structure
        # TODO: could set different levels of reset or at least using reset to last saved state rather than full reset
        return self.document.get_state()  # Return initial state

    def close(self):
        with open(self.log_path, "a") as log_file:
            print(f"Environment closed at {datetime.now()}\n")  # TODO: log it
        self.document.reset()
        self.has_reset_once = False
        self.state = None
        print("Environment internal states reset and closed.")  # TODO: log it

    def step(
        self,
        code: str = "",
        programs: str = "",
        reset_step: bool = False,
        completed_tasks: int = 0,
    ):
        self.synthesis_manager.reset_method_calls_counters()  # we want to count method calls for a step only
        if not self.has_reset_once:
            print("Environment has not been reset yet - resetting now !")
            self.reset()
        return super().step(
            action_code=code,
            context={
                "problem": self.synthesis_manager,
                "bot": self.synthesis_manager,
                "results": None,
                "SynthesisManager": SynthesisManager,
                "DocumentStructure": DocumentStructure,
                "Section": Section,
                "Document": Document,
                "llm": VoyagerEnvIR_CPS_TechSynthesis.llm_model,
            },
        )

    def get_state(self, extended: bool = False):
        # TODO: move to self.document.get_state() ?
        table_of_content_list = self.synthesis_manager.get_plan_status(
            compact_string_format=False if extended else True
        )
        # Convert the list of sections to a string with \n separator
        table_of_content = "\n".join(
            [str(section) for section in table_of_content_list]
        )
        resources_observation = self.synthesis_manager.get_resources_status(
            compact_string_format=True
        )
        document_state = f"<<< Document #{self.id} properties:\n"
        if extended:
            document_state += f"> Title: {self.title}\n"
        if extended:
            document_state += (
                f"> Abstract (first 100 characters): {self.context[:100]}\n"
            )
        document_state += f"> Current table of content:\n{table_of_content if len(table_of_content) > 0 else 'Empty'}\n"
        document_state += f"> Current resources: {resources_observation if len(resources_observation) > 0 else 'Empty'}\n"
        if extended:
            embed_id = (
                "1"
                if self.document.embedding_model_name == "text-embedding-ada-002"
                else "2"
            )
            distance_to_targetJSON = self.synthesis_manager.get_distance_to_targetJSON(
                target_section_title_embedding_label="section_embedding_" + embed_id,
                target_section_content_embedding_label="content_embedding_" + embed_id,
                target_plan_embedding_label="plan_embedding_" + embed_id,
                target_resource_embedding_label="resource_embedding_" + embed_id,
            )
            events_action_counts = self.synthesis_manager.get_count_method_calls()
            document_state += f"1. sections titles progress: {distance_to_targetJSON['plan_titles_embedding_similarity']}\n"
            document_state += f"2. sections content progress: {distance_to_targetJSON['plan_contents_embedding_similarity']}\n"
            document_state += f"3. sections count ratio progress: {distance_to_targetJSON['sections_count_ratio_to_target']}\n"
            document_state += f"4. title non-empty count ratio progress: {distance_to_targetJSON['title_non_empty_count_ratio_to_target']}\n"
            document_state += f"5. content length ratio progress: {distance_to_targetJSON['content_length_ratio_to_target']}\n"
            document_state += f"6. content non-empty count ratio progress: {distance_to_targetJSON['content_non_empty_count_ratio_to_target']}\n"
            document_state += f"7. resources citation coverage score: {distance_to_targetJSON['resources_citation_coverage_score']}\n"
            document_state += f"8. resources similarity progress: {distance_to_targetJSON['plan_resources_embedding_similarity']}\n"
            document_state += f"9. resources count: {distance_to_targetJSON['resources_count_ratio_to_target']}\n"
            document_state += (
                f"10. resources count: {distance_to_targetJSON['resources_count']}\n"
            )
            document_state += (
                f"11. events counted: {events_action_counts}\n"
                if len(events_action_counts) > 0
                else ""
            )

        document_state += ">>>"
        return document_state
