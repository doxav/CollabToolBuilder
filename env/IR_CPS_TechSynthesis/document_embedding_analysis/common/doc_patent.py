import re
import requests
from pathlib import Path
from logging import Logger
from bs4 import BeautifulSoup, Comment

from common.utils import DocType
from common.doc_base import Document


class DocPatent(Document):
    def __init__(self, source_str: str | Path, logger: Logger):
        super().__init__(source_str, DocType.PATENT, logger)

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

    def extract_references(self, context: str):
        # Define refined regex patterns for U.S. Patents
        us_patent_patterns = [
            r'\bU\.S\. Patent No\.?\s*\d{1,3}(?:,\d{3})*\b',          # e.g., U.S. Patent No. 4,072,600
            r'\bUS Patent\s*\d{1,3}(?:,\d{3})*\b',                   # e.g., US Patent 4,072,600
            r'\bU\.S\. Pat\.? No\.?\s*\d{1,3}(?:,\d{3})*\b',         # e.g., U.S. Pat. No. 4,072,600
            r'\bUS\s*\d{1,3}(?:,\d{3})*\b',                           # e.g., US 4,072,600
            r'\bU\.S\. Pat\.?\s*\d{1,3}(?:,\d{3})*\b'                # e.g., U.S. Pat. 4,072,600
        ]

        # Define regex patterns for European Patents
        ep_patent_patterns = [
            r'\bEP[- ]A-\d{6,}\b',                        # e.g., EP-A-0259115
            r'\bEP\s*\d{1,}[- ]?[A-Z]{1,2}\d*\b',         # e.g., EP 0 123 456 B1 or EP2020123456A1
        ]

        # Define regex patterns for WIPO Patents
        wo_patent_patterns = [
            r'\bWO\s*\d{4}/\d{6,}\b',                     # e.g., WO 1988/123456
            r'\bWO\d{7,}A\d\b',                            # e.g., WO2020123456A1
        ]

        # Define regex patterns for other jurisdictions
        other_patent_patterns = [
            r'\bJP\s*\d{7,}\b',                            # e.g., JP 1234567
            r'\bCN\s*\d{10,}A\b',                           # e.g., CN 2012345678A
            r'\bDE\s*\d{6,}B\d\b',                          # e.g., DE 1023456B4
        ]

        # Define regex patterns for non-patent literature
        non_patent_patterns = [
            r'\b[A-Z][a-z]+ et al\.',                 # e.g., Occelli et al.
            r'\b[A-Z][a-z]+ and [A-Z][a-z]+',         # e.g., Smith and Jones
            r'\b[A-Z][a-z]+, [A-Z]\., [A-Z][a-z]+',   # e.g., Smith, J., Doe
            r'\b[A-Z][a-z]+ et al\.\s*\w+',           # e.g., Smith et al. discuss
            r'\b[A-Z][a-z]+, \w+\.\s*',               # e.g., Smith, J.
            r'\b[A-Z][a-z]+ et al\.\s*[\w\s,]+',      # e.g., Smith et al. discuss various
        ]
        # Combine all patent patterns
        all_patent_patterns = us_patent_patterns + ep_patent_patterns + wo_patent_patterns + other_patent_patterns

        # Compile regex patterns for better performance
        compiled_patent_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in all_patent_patterns]
        compiled_non_patent_patterns = [re.compile(pattern) for pattern in non_patent_patterns]

        # Function to extract patent references
        def extract_patent_references(text):
            references = []
            for pattern in compiled_patent_patterns:
                matches = pattern.findall(text)
                references.extend(matches)
            return references

        # Function to extract non-patent references
        def extract_non_patent_references(text):
            references = []
            for pattern in compiled_non_patent_patterns:
                matches = pattern.findall(text)
                references.extend(matches)
            return references

        # Clean and format the references
        def clean_reference(ref):
            # Remove extra spaces and normalize
            ref = ref.strip()
            # Remove trailing punctuation
            ref = ref.rstrip('.')
            # Remove trailing comments or annotations within parentheses
            ref = re.sub(r'\s*\(.*?\)', '', ref)
            # Remove commas in patent numbers for consistency
            ref = re.sub(r'(?<=\d),(?=\d)', '', ref)
            return ref

        # Extracting references
        patent_references = extract_patent_references(context)
        non_patent_references = extract_non_patent_references(context)

        # Combine all references
        all_references = patent_references# + non_patent_references

        # cleaned_references = [clean_reference(ref) for ref in all_references]

        # Remove duplicates
        unique_references = list(set(all_references))

        # Sort references for readability
        unique_references.sort()

        ref_list = []
        for reference in unique_references:
            ref_list.append({
                    "resource_id": len(ref_list)+1,
                    "resource_description": reference
                })
        return ref_list

    def extract_paper_data(self):
        """Read in a patent file and return a dict with keys as section titles and values the content.

        Parameters
        ----------
        patent_file : str
            Path to the patent file.

        Returns
        -------
        patent_dict : dict
            Dict with keys as section titles and values the content. Keys are ['title',
            'descr', 'claim_1', 'claim_2', ..., 'claim_n', 'pdfep']. Not all patents
            will have all keys. All will have 'title' at a minimum.
        """
        patent_file = self.getInput()
        self.getLogger().info(f"Loading patent file: {patent_file}")
        # Read file
        with open(patent_file, "r") as f:
            lines: list = f.readlines()

        # Get all english sections
        lines_en: list = [line for line in lines if "\ten\t" in line]

        # Convert into dict with keys as section titles and values the content
        patent_dict = {}
        total_claims = 1
        for x in lines_en:
            if "\tTITLE\t" in x:
                patent_dict["title"] = x
            elif "\tDESCR\t" in x:
                patent_dict["descr"] = x
            elif "\tCLAIM\t" in x:
                # Some patents have multiple claims, so we need to number them
                patent_dict[f"claim_{total_claims}"] = x
                total_claims += 1
            elif "\tPDFEP" in x:
                patent_dict["pdfep"] = x
            else:
                raise ValueError(
                    f"Expected sections in [TITLE, DESCR, CLAIM, PDFEP]. Received: {x}"
                )

        # Extract reference
        references = self.extract_references(''.join(lines_en))
        # patent_dict["references"] = references
        self.setReferences(references)

        self.getLogger().info(
            f"Extracted {len(patent_dict)} sections from patent file named: "
            f"{list(patent_dict.keys())}"
        )

        return patent_dict