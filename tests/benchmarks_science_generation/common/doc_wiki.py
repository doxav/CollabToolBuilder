import re
import requests
from pathlib import Path
from logging import Logger
from bs4 import BeautifulSoup, Comment

from common.utils import DocType
from common.doc_base import Document


class DocWiki(Document):
    def __init__(self, source_str: str | Path, logger: Logger, output_dir: str = None):
        super().__init__(source_str, DocType.WIKI, logger, output_dir)

    def extract_plan_and_content(self, skip_if_exists: bool = False):
        super().generateOutputFile()

        # Get outputfile
        output_file = super().getOutputFile()

        # Skip if exists
        if skip_if_exists and output_file.exists():
            self.getLogger().info(f"\n\tFile already exists: {output_file.absolute()}")
            return {"title": str(input), "content": "File already exists."}

        paper_data = self.extract_paper_data()
        paper_data = self.divide_into_chunks(paper_data)

        num_sections = len(paper_data.keys())
        min_required_sections = 3
        if num_sections < min_required_sections:
            error_msg = (
                f"\n\tInput: {input}"
                f"\n\tInput document is too small. Found {num_sections} sections. We require at "
                f"least {min_required_sections} sections."
                f"\n\tSections Found: {list(paper_data.keys())}"
                f"\n\tSkipping this document and moving onto the next."
            )
            self.getLogger().error(error_msg)
            return {"title": str(input), "content": error_msg}
        plan_json = self.generate_embeddings_plan_and_section_content(paper_data)

        # Write output
        self.writeOuputJson(plan_json)
        return plan_json

    def extract_paper_data(self):
        """Extracts the content from a Wikipedia URL into a dictionary. The keys are the header
        names + indicator of header level e.g. 'h2 Definitions'. The values are the content
        underneath each header tags.

        Only extracts content the user will see. Ignores hidden content and Contents list.

        Parameters
        ----------
        url : str
            The URL of the Wikipedia page to extract content from.

        Returns
        -------
        article_dict : dict
            A dictionary of the content extracted from the Wikipedia page.
        """
        url = self.getInput()
        if not "wikipedia" in url:
            raise ValueError("URL is not a wikipedia URL. Received: " + url)

        r = requests.get(url)
        html_content = r.text

        # Create a BeautifulSoup object
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove unwanted tags: script, style, [document], head, title
        for element in soup(["script", "style", "head", "title", "[document]"]):
            element.decompose()

        # Also remove HTML comments
        for element in soup.find_all(string=lambda text: isinstance(text, Comment)):
            element.extract()

        # Define the header tags to find
        tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
        found_tags = soup.find_all(tags)

        # Extract tags and associated content into a dict
        article_dict = {}
        reference_mapping = {}  # Map citation numbers to references
        refer_key = None
        for tag in found_tags:
            content = []
            next_tag = tag.find_next()

            # Look for next tags until the next header tag
            while next_tag and next_tag.name not in tags:
                # Reference section can contain both p and li tags
                if "reference" in str(next_tag).lower() and next_tag.name in [
                    "p",
                    "li",
                ]:
                    ref_text = next_tag.get_text(strip=False)
                    content.append(ref_text)

                    # Extract citation number if present
                    cite_match = re.match(r"^\s*\^?\s*(\d+)\s*\.?\s*(.+)", ref_text)
                    if cite_match:
                        cite_num = int(cite_match.group(1))
                        cite_text = cite_match.group(2).strip()
                        reference_mapping[cite_num] = cite_text

                elif next_tag.name == "p":
                    content.append(next_tag.get_text(strip=False))
                next_tag = next_tag.find_next()

            key = f"{tag.name} {tag.get_text(strip=True)}"
            article_dict[key] = " ".join(content)

        # Clean up [edit] suffixes
        for key in list(
            article_dict.keys()
        ):  # Using list() to avoid changing the dictionary size during iteration
            if key.endswith("[edit]"):
                new_key = key.rsplit("[edit]", 1)[0]
                article_dict[new_key] = article_dict.pop(key)
        # Process references section
        ref_key = None
        article_dict["references"] = []
        for key in article_dict:
            if key[3:].lower() == "references":
                ref_key = key
                ref_content = article_dict[key]

                # Split by common reference delimiters
                ref_splits = re.split(r"\n\s*\^?\s*\d+\s*\.?\s*", ref_content)

                for i, reference in enumerate(ref_splits):
                    if not reference.strip():
                        continue

                    # Try to extract reference number from mapping or use sequential
                    ref_id = i + 1
                    if i + 1 in reference_mapping:
                        ref_text = reference_mapping[i + 1]
                    else:
                        ref_text = reference.strip()

                    article_dict["references"].append(
                        {"resource_id": ref_id, "resource_description": ref_text}
                    )
                break

        # Set references for the document
        self.setReferences(article_dict["references"])

        del article_dict["h2 Contents"]  # TODO: check what is this specific case
        if ref_key in article_dict:
            del article_dict[ref_key]

        num_sections = len(article_dict.keys())

        self.getLogger().info(
            f"Successfully downloaded content from Wikipedia page {url}. "
            f"Extracted {num_sections} sections."
        )

        self.getLogger().info("EXTRACTION OVERVIEW:\nTitle:")

        # print for each key in article_dict, display key and frst 50 characters of value
        for key in article_dict.keys():
            self.getLogger().info(f"{key}: {article_dict[key][:80]}...")

        return article_dict
