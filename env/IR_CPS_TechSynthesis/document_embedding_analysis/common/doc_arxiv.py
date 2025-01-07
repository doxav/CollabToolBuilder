import re
from pathlib import Path
from logging import Logger
from pdfminer.high_level import extract_text

from common.utils import DocType
from common.doc_base import Document


class DocArxiv(Document):
    def __init__(self, source_str: str | Path, logger: Logger):
        super().__init__(source_str, DocType.ARXIV, logger)

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
        """Load an arXiv paper from disk to a dict with keys "Title", "Abstract", "Content",
        and "References"."""
        # Extract all text from pdf
        text = extract_text(self.getInput())

        # The pattern searches for "abstract" followed by any content.
        # Then, it looks for one of the potential following sections:
        # "I. Introduction", "1. Introduction", or "Contents".
        # We use a positive lookahead (?=...) to assert that the introduction or contents
        # pattern exists, but we don't include it in the main match.
        pattern = r"(?:abstract|this\s+paper)(.*?)(?=([i1][. ]+introduction|contents))"

        # The re.DOTALL flag allows the . in the pattern to match newline characters,
        match = re.search(pattern, text.lower(), re.DOTALL)

        abstract = ""
        abstract_end = 0
        if match:
            abstract_end = match.end()
            abstract = match.group(1).strip()

        # Extract references section
        pattern = r"references\s*\n"
        matches = [match for match in re.finditer(pattern, text.lower())]

        ref_list = []
        references = ""
        reference_start = len(text)
        if matches:
            final_match = matches[-1]
            reference_start = final_match.start()
            references = text[reference_start:]

            # Regex to extract reference number and description
            ref_pattern = r"\[(\d+)\]\s+(.*?)(?=\n\s*\[\d+\]|$)"
            # Find all matches
            matches = re.findall(ref_pattern, references, re.DOTALL)
            
            for ref_number, ref_description in matches:
                ref_list.append({
                    "resource_id": int(ref_number),
                    "resource_description": ref_description.replace('\n', '').strip()
                })
                
            ref_list = sorted(ref_list, key=lambda x: x["resource_id"])

        content = text[abstract_end:reference_start]

        self.getLogger().info(
            "EXTRACTION OVERVIEW:\nTitle:" +
            self.getTitle()[:50].replace("\n","\\n") + 
            "...\nAbstract:"+abstract[:50].replace("\n","\\n") + 
            "...\nContent:"+content[:50].replace("\n","\\n") + 
            "...\nReferences:"+references[:50].replace("\n","\\n")+"..."
        )

        article_dict = {
            "title": self.getTitle(),
            "abstract": abstract,
            "content": content,
            "references": ref_list,
        }
        self.setReferences(ref_list)
        return article_dict