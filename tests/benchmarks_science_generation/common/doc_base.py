import re, os, json
import tiktoken
import numpy as np
from copy import copy
from uuid import uuid4
from logging import Logger
from pathlib import Path
from typing import Dict, Any, List
from common.utils import DocType
from common.config import (
    OPENAI_EMBEDDING_MODEL_NAME,
    HUGGINGFACE_EMBEDDING_MODEL_NAME,
    HUGGINGFACE_EMBEDDING_PREFIX,
    MAX_EMBEDDING_TOKEN_LENGTH,
    embed_OPENAI,
    embed_HF,
    ALLOW_parallel_gen_embed_section_content,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from openai import OpenAI


class Document:
    def __init__(
        self,
        source: str | Path,
        doc_type: DocType,
        logger: Logger,
        output_dir: str = None,
    ):
        self.__source = source
        self.__type = doc_type
        self.__logger = logger
        self.__output_dir = output_dir or f"output/{doc_type.value}"
        self.__title = None
        self.__references = []
        if not os.path.exists(self.__output_dir):
            os.makedirs(self.__output_dir)
        self.__output_file = None

    def setReferences(self, references: List):
        self.__references = references

    def getInput(self) -> str | Path:
        return self.__source

    def getType(self) -> DocType:
        return self.__type

    def getLogger(self) -> Logger:
        return self.__logger

    def generateOutputFile(self, **kargs):
        if self.__type == DocType.LATEX:
            without_embeddings = (
                bool(kargs["without_embeddings"])
                if "without_embeddings" in kargs
                else False
            )
            if without_embeddings:
                self.__output_file = os.path.join(
                    self.__output_dir, self.__source.replace(".tex", ".json")
                )
            else:
                self.__output_file = os.path.join(
                    self.__output_dir,
                    self.__source.name.replace(".tex", "_with_embeddings.json"),
                )
        else:
            self.__output_file = self.__output_dir / Path(
                f"{self.__generate_title()}.json"
            )

    def getOutputFile(self):
        return self.__output_file

    def __generate_title(self):
        """Extracts the title from the input based on the document type."""
        if self.__type == DocType.WIKI:
            title = self.__source.split("/")[-1].replace("_", " ")
        elif self.__type == DocType.ARXIV:
            title = Path(self.__source).stem.replace("_", " ")
        elif self.__type == DocType.PATENT:
            with open(self.__source, "r") as f:
                first_line = f.readline()
            title = first_line.split("\t")[-1].strip()
            title = "".join(i for i in title if i not in "\/:*?<>|")
        self.__title = title
        return title

    def getTitle(self):
        return self.__title

    def __gen_embed_plan(self, plan: List[dict], i: int) -> List[float]:
        """Calculate plan embedding by averaging the section embeddings and content embeddings
        sequentially.

        Parameters
        ----------
        plan : List[dict]
            List of dictionaries, each containing the section and content embeddings.
        i : int
            The index of the embedding to use.
        """
        try:
            s1_mean = np.mean([x[f"section_embedding_{i}"] for x in plan], axis=0)
            c1_mean = np.mean([x[f"content_embedding_{i}"] for x in plan], axis=0)
        except KeyError:
            raise KeyError(
                f"Could not find section_embedding_{i} or content_embedding_{i} in "
                f"every element in plan. Please check that every element has both of these "
                f"keys."
            )
        total_mean = np.mean([c1_mean, s1_mean], axis=0)
        total_mean = list(total_mean)
        self.__logger.info(f"Created plan embedding {i}")
        return total_mean

    def __gen_embed_section_content_batch(
        self,
        headings: List[str],
        contents: List[str],
        ids: List[int],
        total_sections: int,
    ) -> List[Dict[str, str | List[float]]]:
        """Given lists of headings and contents, returns a list of dictionaries
        with the headings, contents, and embeddings for each section.

        Parameters
        ----------
        headings : List[str]
            The headings of the sections.
        contents : List[str]
            The contents of the sections.
        ids : List[int]
            The ids of the sections.
        total_sections : int
            The total number of sections.
        """
        section_jsons = []

        heading_embeddings_1 = embed_OPENAI.embed_documents(headings)
        content_embeddings_1 = embed_OPENAI.embed_documents(contents)

        def HF_embed_split(contents, n):
            return sum(
                [
                    embed_HF.embed_documents(
                        [HUGGINGFACE_EMBEDDING_PREFIX + c for c in contents[i::n]]
                    )
                    for i in range(n)
                ],
                [],
            )

        n = 1
        while True:
            try:
                heading_embeddings_2 = HF_embed_split(headings, n)
                break
            except:
                n *= 2
                self.__logger.info(
                    f"Failed heading_embeddings_2 to HF embed content 1 batch of {len(headings)/(n/2)} trying with {n} splits"
                )
        n = 1
        while True:
            try:
                content_embeddings_2 = HF_embed_split(contents, n)
                break
            except:
                n *= 2
                self.__logger.info(
                    f"Failed content_embeddings_2 to HF embed content 1 batch of {len(contents)/(n/2)} trying with {n} splits"
                )

        for id, heading, content, emb_h1, emb_h2, emb_c1, emb_c2 in zip(
            ids,
            headings,
            contents,
            heading_embeddings_1,
            heading_embeddings_2,
            content_embeddings_1,
            content_embeddings_2,
        ):

            section_jsons.append(
                {
                    "section_id": id,
                    "section": heading,
                    "content": content,
                    "section_embedding_1": emb_h1,
                    "section_embedding_2": emb_h2,
                    "content_embedding_1": emb_c1,
                    "content_embedding_2": emb_c2,
                    "resources_used": self.__get_reference_used(content),
                }
            )

            self.__logger.info(
                f"{id}/{total_sections} - created section + content embeddings for {heading} - SECTION: {heading[:50]}... CONTENT: {content[:50]}..."
            )

        return section_jsons

    def __parallel_gen_embed_section_content(
        self,
        headings: List,
        contents: List,
        start_index: int,
        total_sections: List,
        initial_plan=None,
    ):
        # Collect data for batch processing
        headings_to_process = []
        contents_to_process = []
        indices = []

        for i, (heading, content) in enumerate(
            zip(headings[start_index:], contents[start_index:]), start=1
        ):
            headings_to_process.append(heading)
            contents_to_process.append(content)
            indices.append(i)

        # Process embeddings in batch
        plan = self.__gen_embed_section_content_batch(
            headings_to_process, contents_to_process, indices, total_sections
        )
        # Merge dict data for each element of plan if initial_plan is provided
        if initial_plan:
            for i, (p, ip) in enumerate(zip(plan, initial_plan), start=1):
                p.update(ip)

        return plan

    def __get_reference_used(self, content: str):
        # Get resource indexes used in content
        resources_ids = []
        if self.__type != DocType.PATENT:
            resources_ids = sorted(
                set([int(num) for num in re.findall(r"\[(\d+)\]", content)])
            )
        else:
            for reference in self.__references:
                refer_name = reference["resource_description"]
                if refer_name in content:
                    resources_ids.append(reference["resource_id"])
            resources_ids = sorted(list(set(resources_ids)))
        return resources_ids

    def __gen_embed_section_content(
        self, heading: str, content: str, id: int = 1, total_sections: int = 1
    ) -> Dict[str, str | list[float]]:
        """Given a heading and content, returns a dictionary with the heading, content,
        and embeddings of the heading and content.

        Parameters
        ----------
        heading : str
            The heading of the section.
        content : str
            The content of the section.
        id : int, optional
            The id of the section, by default 1
        total_sections : int, optional
            The total number of sections, by default 1
        """
        section_json = {
            "section_id": id,
            "section": heading,
            "content": content,
            "section_embedding_1": embed_OPENAI.embed_query(heading),
            "section_embedding_2": embed_HF.embed_query(
                HUGGINGFACE_EMBEDDING_PREFIX + heading
            ),
            "content_embedding_1": embed_OPENAI.embed_query(content),
            "content_embedding_2": embed_HF.embed_query(
                HUGGINGFACE_EMBEDDING_PREFIX + content
            ),
            "resources_used": self.__get_reference_used(content),
        }

        self.__logger.info(
            f"{id}/{total_sections} - created section + content embeddings for {heading} - SECTION: {heading[:50]}... CONTENT: {content[:50]}..."
        )

        return section_json

    def generate_plan_embeddings(
        self,
        paper_data: Dict,
        headings: List,
        content: List,
        title: str,
        abstract: str,
        start_index: int,
        total_sections: List,
        initial_plan=None,
        allow_parallel: bool = True,
    ):

        self.__logger.info("Title: " + title)
        self.__logger.info("Abstract: " + abstract)

        if allow_parallel:
            plan = self.__parallel_gen_embed_section_content(
                headings, content, start_index, total_sections, initial_plan
            )
        else:
            plan = [
                self.__gen_embed_section_content(
                    heading, content, id=i, total_sections=total_sections
                )
                for i, (heading, content) in enumerate(
                    zip(headings[start_index:], content[start_index:]), start=1
                )
            ]

        # Get Resources
        self.__logger.info(f"Creating resource embedding")
        # resources = paper_data.get("references", [])
        resources = self.__references
        for i in range(len(resources)):
            resources[i]["resource_embedding_1"] = embed_OPENAI.embed_query(
                resources[i]["resource_description"]
            )
            resources[i]["resource_embedding_2"] = embed_HF.embed_query(
                HUGGINGFACE_EMBEDDING_PREFIX + resources[i]["resource_description"]
            )

        plan_embed_1 = self.__gen_embed_plan(plan, 1)
        plan_embed_2 = self.__gen_embed_plan(plan, 2)
        try:
            plan_json = {
                "id": str(uuid4()) if "id" not in paper_data else paper_data["id"],
                "title": title,
                "abstract": abstract,
                "title_embedding_1": embed_OPENAI.embed_query(title),
                "title_embedding_2": embed_HF.embed_query(
                    HUGGINGFACE_EMBEDDING_PREFIX + title
                ),
                "abstract_embedding_1": embed_OPENAI.embed_query(abstract),
                "abstract_embedding_2": embed_HF.embed_query(
                    HUGGINGFACE_EMBEDDING_PREFIX + abstract
                ),
                "plan": plan,
                "resources": resources,
                "plan_embedding_1": plan_embed_1,
                "plan_embedding_2": plan_embed_2,
                "embedding1_model": OPENAI_EMBEDDING_MODEL_NAME,
                "embedding2_model": HUGGINGFACE_EMBEDDING_MODEL_NAME,
                "success": True,
                "error": None,
            }
        except Exception as e:
            plan_json = {
                "id": str(uuid4()),
                "title": title,
                "abstract": abstract,
                "title_embedding_1": None,
                "title_embedding_2": None,
                "abstract_embedding_1": None,
                "abstract_embedding_2": None,
                "plan": plan,
                "resources": [],
                "plan_embedding_1": None,
                "plan_embedding_2": None,
                "embedding1_model": OPENAI_EMBEDDING_MODEL_NAME,
                "embedding2_model": HUGGINGFACE_EMBEDDING_MODEL_NAME,
                "success": False,
                "error": str(e),
            }
        return plan_json

    def __num_tokens_from_string(
        self, string: str, encoding_name: str = "gpt-4o-mini"
    ) -> int:
        """Returns the number of tokens in a text string."""
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except ValueError:
            encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def __extract_title(self, string: str) -> str:
        """Extract a title from `string` that is max 7 words long."""
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract a concise title from the given text. The title must be maximum 7 words and capture the main topic.",
                    },
                    {
                        "role": "user",
                        "content": f"Extract a title from this text:\n\n{string[:1000]}",  # Limit input length
                    },
                ],
                max_tokens=20,
                temperature=0.1,
            )

            title = response.choices[0].message.content.strip()
            # Ensure max 7 words
            words = title.split()
            if len(words) > 7:
                title = " ".join(words[:7])
            return title

        except Exception as e:
            self.__logger.error(f"Error extracting title from string: {e}")
            return "None"

    def divide_into_chunks(
        self,
        papert_data: Dict,
        max_section_token_length: int = MAX_EMBEDDING_TOKEN_LENGTH,
    ):
        self.__logger.info(
            "Dividing sections if too large in plan and section content."
        )
        final_dict: Dict = {}
        start_dict = copy(papert_data)

        def is_reference_section(heading: str):
            """Returns True if heading is a reference section."""
            result = re.search(
                r"reference|further reading|see also|bibliography|external links",
                heading,
                re.IGNORECASE,
            )
            return result

        for heading, content in start_dict.items():
            content_length = len(content)
            if content_length == 0:
                final_dict[heading] = " "
            elif (
                content_length <= max_section_token_length
            ):  # characters length is always less than token length
                final_dict[heading] = content
            else:
                num_tokens = self.__num_tokens_from_string(content)
                self.__logger.info(
                    f"Content character length: {len(content)} tokens: {num_tokens} for '{heading}'"
                )
                # Each section must contain something, otherwise the embedding models fail
                if num_tokens == 0:
                    final_dict[heading] = " "
                # If the section is small enough, add it to the final dict
                elif num_tokens <= max_section_token_length:
                    final_dict[heading] = content
                # If section is too big, split into smaller sections, extract title, and add to final dict
                else:
                    # Split
                    char_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=max_section_token_length,
                        chunk_overlap=0,
                        # ' ' separator means sometimes sentences will be cut in two to ensure
                        # the chunk size is not exceeded
                        separators=["\n\n", "\n", ". ", ".", ", ", ",", " "],
                        length_function=self.__num_tokens_from_string,
                    )
                    splits: List[str] = char_splitter.split_text(content)
                    # Keep heading the same but add numbers to sections e.g. 'h2 Reference' -> 'h2 Reference 1'
                    # TODO - add a continue statement here?
                    if self.__type in [
                        DocType.WIKI,
                        DocType.ARXIV,
                    ] and is_reference_section(heading):
                        for i, split in enumerate(splits, start=1):
                            new_heading = f"{heading} {i}"
                            final_dict[new_heading] = split
                            self.__logger.info(
                                f"Added '{new_heading}' split original heading '{heading}'"
                            )
                    else:
                        # Create new titles for each split
                        for split in splits:
                            # Headings are of the form h1, h2, h3 etc. we split it into more of the same level
                            if self.__type == DocType.WIKI:
                                heading_level = int(heading[1])
                                title = self.__extract_title(split)
                                new_heading = f"h{heading_level} {title}"
                            # Heading levels aren't important for other doc_types
                            else:
                                new_heading = self.__extract_title(split)
                            final_dict[new_heading] = split
                            self.__logger.info(
                                f"Added '{new_heading}' split original heading '{heading}'"
                            )

        n_keys_start = len(start_dict.keys())
        n_keys_final = len(final_dict.keys())
        self.__logger.info(
            f"\n\tFinished dividing sections if too large in plan and section content."
            f"\n\tStarted with {n_keys_start} sections and got {n_keys_final} final sections."
            f"\n\tThat's a {n_keys_final / n_keys_start:.2f}x increase in sections"
        )

        return final_dict

    def generate_embeddings_plan_and_section_content(self, article_dict: Dict) -> Dict:
        """Given a dictionary of the article content, returns a dictionary with the title,
        abstract, plan and associated embeddings.
        """
        if self.__type == DocType.WIKI:
            headings = [
                key for key in article_dict.keys() if isinstance(article_dict[key], str)
            ]
            content = [
                value
                for key, value in article_dict.items()
                if isinstance(article_dict[key], str)
            ]
            # Wikipedia titles are of form 'h1 Example' so we remove the 'h1 '
            title = headings[0][3:]
            abstract = content[0]
            total_sections = len(headings) - 1
            start_index = 1
        elif self.__type == DocType.PATENT:
            headings = [
                key for key in article_dict.keys() if isinstance(article_dict[key], str)
            ]
            content = [
                value
                for key, value in article_dict.items()
                if isinstance(article_dict[key], str)
            ]
            # Titles are separated by tabs, the last element is the actual title
            title = content[0].split("\t")[-1].strip()
            # Remove illegal characters from title (it's used as a filename)
            title = "".join(i for i in title if i not in "\/:*?<>|")
            try:
                abstract = content[1]
            except IndexError:
                abstract = "no abstract"
            total_sections = len(headings) - 2
            start_index = 2
        elif self.__type == DocType.ARXIV:
            # Filtered keys and values
            headings = [
                key for key in article_dict.keys() if isinstance(article_dict[key], str)
            ]
            content = [
                value
                for key, value in article_dict.items()
                if isinstance(article_dict[key], str)
            ]
            # The first key/value pairs in arxiv dicts are {'Title': title, 'Abstract': abstract}
            # so we take the first two elements of content
            title = content[0]
            try:
                abstract = content[1]
            except IndexError:
                abstract = "no abstract"
            total_sections = len(headings) - 2
            start_index = 2
        else:
            raise ValueError("Unknown document type!")

        plan = self.generate_plan_embeddings(
            paper_data=article_dict,
            headings=headings,
            content=content,
            title=title,
            abstract=abstract,
            start_index=start_index,
            total_sections=total_sections,
            initial_plan=article_dict.get("plan", None),
            allow_parallel=ALLOW_parallel_gen_embed_section_content,
        )
        return plan

    def writeOuputJson(self, data):
        with open(self.__output_file, "w") as file:
            json.dump(data, file, indent=4)
        self.__logger.info(f"Successfully processed: {self.__source}")

    def writeOutputString(self, data):
        with open(self.__output_file, "w") as file:
            file.write(data)
        self.__logger.info(f"Successfully processed: {self.__source}")
