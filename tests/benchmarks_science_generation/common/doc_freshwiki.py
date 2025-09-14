import json
from pathlib import Path
from logging import Logger

from common.utils import DocType
from common.doc_base import Document


class DocFreshWiki(Document):
    def __init__(self, source_str: str | Path, logger: Logger, output_dir: str = None):
        # Add a new DocType for FreshWiki
        super().__init__(source_str, DocType.FRESHWIKI, logger, output_dir)

    def extract_plan_and_content(self, skip_if_exists: bool = False):
        super().generateOutputFile()

        # Get outputfile
        output_file = super().getOutputFile()

        # Skip if exists
        if skip_if_exists and output_file.exists():
            self.getLogger().info(f"\n\tFile already exists: {output_file.absolute()}")
            return {"title": str(self.getInput()), "content": "File already exists."}

        paper_data = self.extract_paper_data()
        paper_data = self.divide_into_chunks(paper_data)

        num_sections = len(paper_data.keys())
        min_required_sections = 3
        if num_sections < min_required_sections:
            error_msg = (
                f"\n\tInput: {self.getInput()}"
                f"\n\tInput document is too small. Found {num_sections} sections. We require at "
                f"least {min_required_sections} sections."
                f"\n\tSections Found: {list(paper_data.keys())}"
                f"\n\tSkipping this document and moving onto the next."
            )
            self.getLogger().error(error_msg)
            return {"title": str(self.getInput()), "content": error_msg}

        plan_json = self.generate_embeddings_plan_and_section_content(paper_data)

        # Write output
        self.writeOuputJson(plan_json)
        return plan_json

    def extract_paper_data(self):
        """Load a FreshWiki JSON file and convert to standard format."""
        with open(self.getInput(), "r", encoding="utf-8") as f:
            freshwiki_data = json.load(f)

        solution = freshwiki_data.get("solution", {})
        sections = solution.get("content", [])
        references_dict = solution.get("references", {})

        # Extract title and abstract
        title = solution.get("title", "").replace("_", " ")
        abstract = solution.get("summary", "")

        # Convert sections to flat dictionary format
        article_dict = {
            "title": title,
            "abstract": abstract,
        }

        # Process sections recursively
        section_counter = 0
        ref_list = []
        ref_mapping = {}  # Map reference URLs to IDs

        def process_section(section, level=1, parent_title=""):
            nonlocal section_counter, ref_list, ref_mapping

            section_title = section.get("section_title", "")
            full_title = f"h{level} {section_title}"

            # Extract content and references
            content_parts = []
            for content_item in section.get("section_content", []):
                sentence = content_item.get("sentence", "")
                content_parts.append(sentence)

                # Process references
                for ref_url in content_item.get("refs", []):
                    if ref_url not in ref_mapping:
                        ref_id = len(ref_list) + 1
                        ref_mapping[ref_url] = ref_id

                        # Find reference description from references dict
                        ref_desc = ref_url
                        for ref_key, ref_val in references_dict.items():
                            if ref_val == ref_url:
                                ref_desc = f"[{ref_key}] {ref_url}"
                                break

                        ref_list.append(
                            {"resource_id": ref_id, "resource_description": ref_desc}
                        )

            # Join content
            content = " ".join(content_parts)
            article_dict[full_title] = content

            # Process subsections
            for subsection in section.get("subsections", []):
                process_section(subsection, level + 1, section_title)

        # Process all sections
        for section in sections:
            process_section(section)

        # Add references list
        self.setReferences(ref_list)

        self.getLogger().info(
            f"EXTRACTION OVERVIEW:\nTitle: {title[:50]}..."
            f"\nAbstract: {abstract[:50]}..."
            f"\nSections: {len(article_dict) - 2}"
            f"\nReferences: {len(ref_list)}"
        )

        return article_dict
