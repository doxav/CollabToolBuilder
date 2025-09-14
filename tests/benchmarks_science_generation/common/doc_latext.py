import re
import os
import json
from typing import List, Any, Dict
from uuid import uuid4
from logging import Logger
from pathlib import Path
from common.utils import DocType
from common.doc_base import Document


class DocLatex(Document):
    def __init__(self, source_str: str | Path, logger: Logger, output_dir: str = None):
        super().__init__(source_str, DocType.LATEX, logger, output_dir)

    def __clean_context(self, content):
        # Remove lines starting with %
        content = "\n".join(
            line for line in content.split("\n") if not line.strip().startswith("%")
        )

        # Remove unnecessary LaTeX commands
        content = re.sub(r"\\(usepackage|documentclass)\{.*?\}", "", content)

        # Remove LaTeX environments
        # content = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', content, flags=re.DOTALL)

        # Remove extra whitespace
        content = re.sub(r"\s+", " ", content).strip()

        return content

    def __parse_content(self):
        parent_dir = self.getInput().parent
        tex_filename = self.getInput().name
        bib_filename = tex_filename.replace(".tex", ".bib")

        with open(self.getInput(), "r", encoding="utf8") as file:
            latex_content = self.__clean_context(file.read())

        with open(
            parent_dir / bib_filename, "r", encoding="utf-8", errors="replace"
        ) as file:
            bibtex_content = file.read()

        return latex_content, bibtex_content

    def __get_title(self, context: str):
        title_match = re.search(r"\\title\{(.+?)\}", context, re.DOTALL | re.IGNORECASE)
        return title_match.group(1).strip() if title_match else "No Title Found"

    def __get_abstract(self, context: str):
        abstract_match = (
            re.search(
                r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                context,
                re.DOTALL | re.IGNORECASE,
            )
            or re.search(r"\\abstract\{(.+?)\}", context, re.DOTALL | re.IGNORECASE)
            or re.search(r"\\abst[ \n](.*?)\\xabst", context, re.DOTALL | re.IGNORECASE)
            or re.search(
                r"\\section\*\{abstract\}(.+?)(?=\\section|\Z)",
                context,
                re.DOTALL | re.IGNORECASE,
            )
        )
        return (
            abstract_match.group(1).strip() if abstract_match else "No Abstract Found"
        )

    def __get_plan(self, sections: List, references: List, context: str):
        plan = []
        section_id = 0
        cited_references = set()

        for section_type, section_title in sections:
            # section id
            section_id += 1

            # get content
            original_section_title = section_title.split(" ", 1)[1]
            section_regex = rf"\\{section_type}\{{{re.escape(original_section_title)}\}}(.*?)(?=\\(?:sub)*section\{{|\\end\{{document\}})"
            section_content_match = re.search(section_regex, context, re.DOTALL)
            content = (
                section_content_match.group(1).strip() if section_content_match else ""
            )

            citations = re.findall(r"\\cite[t|p]*\{([^}]+)\}", content)
            resources_cited = list(
                {citation for citation in citations if citation in references}
            )
            cited_references.update(resources_cited)

            # plan.append(plan_create_section_entry(section_id, section_title, content, resources_cited, references))
            plan.append(
                {
                    "section_id": section_id,
                    "section": section_title,
                    "content": content,
                    "resources_used": sorted(
                        set(
                            [
                                i
                                for i, ref in enumerate(references)
                                if ref in resources_cited
                            ]
                        )
                    ),
                    "resources_cited_key": resources_cited,
                }
            )
        return plan, cited_references

    def __get_sections_hierarchical_numbering(
        self, sections: List[Any], method="counters"
    ) -> List:
        section_counters = {"section": 0, "subsection": 0, "subsubsection": 0}
        numbered_sections = []

        for section_type, section_title in sections:
            if method == "counters":
                if section_type == "section":
                    section_counters["section"] += 1
                    section_counters["subsection"] = 0
                    section_counters["subsubsection"] = 0
                    section_number = f"{section_counters['section']}"
                elif section_type == "subsection":
                    section_counters["subsection"] += 1
                    section_counters["subsubsection"] = 0
                    section_number = f"{section_counters['section']}.{section_counters['subsection']}"
                elif section_type == "subsubsection":
                    section_counters["subsubsection"] += 1
                    section_number = f"{section_counters['section']}.{section_counters['subsection']}.{section_counters['subsubsection']}"
                section_title = f"{section_number} {section_title}"
            numbered_sections.append((section_type, section_title))

        return numbered_sections

    def __get_sections(self, context: str):
        section_pattern = r"\\((?:sub)*section)\{(.+?)\}"
        sections = re.findall(section_pattern, context)
        return self.__get_sections_hierarchical_numbering(sections, method="counters")

    def __get_resources(self, context: str) -> List:
        resources = []
        entries = re.findall(
            r"@(\w+)\{([^,]+),(.+?)\n\}", context, re.DOTALL | re.IGNORECASE
        )
        for i, (entry_type, citation_key, content) in enumerate(entries):
            title_match = re.search(
                r"title\s*=\s*\{(.+?)\}", content, re.DOTALL | re.IGNORECASE
            )
            author_match = re.search(
                r"author\s*=\s*\{(.+?)\}", content, re.DOTALL | re.IGNORECASE
            )
            year_match = re.search(r"year\s*=\s*\{(.+?)\}", content)
            url_match = re.search(r"url\s*=\s*\{(.+?)\}", content)
            doi_match = re.search(r"doi\s*=\s*\{(.+?)\}", content, re.DOTALL)

            title = title_match.group(1).strip() if title_match else None
            author = author_match.group(1).strip() if author_match else None
            year = year_match.group(1).strip() if year_match else None
            url = (
                url_match.group(1).strip()
                if url_match
                else doi_match.group(1).strip() if doi_match else None
            )

            resources.append(
                {
                    "resource_id": i + 1,
                    "resource_key": citation_key.strip(),
                    "resource_description": f"{title if title else ''}\nAuthor:{author if author else ''}\nYear:{year if year else ''}",
                    "url": url,
                }
            )
        return resources

    def extract_paper_data(self):
        latext_context, bibtex_context = self.__parse_content()

        # Get sections
        numbered_sections = self.__get_sections(latext_context)

        # Extract references
        references = re.findall(
            r"@.*?\{(.*?),", bibtex_context, re.DOTALL | re.IGNORECASE
        )

        # Extract resources
        resources = self.__get_resources(bibtex_context)

        # Extract plan
        plan, resource_cited = self.__get_plan(
            numbered_sections, references, latext_context
        )
        self.setReferences(
            [res for res in resources if res["resource_key"] in resource_cited]
        )

        paper_data = {
            "id": str(uuid4()),
            "title": self.__get_title(latext_context),
            "abstract": self.__get_abstract(latext_context),
            "plan": plan,
            # "references": [res for res in resources if res['resource_key'] in resource_cited]
        }
        return paper_data

    def generate_embeddings_plan_and_section_content(self, paper_data: Dict) -> Dict:
        headings = [x["section"] for x in paper_data["plan"]]
        content = [x["content"] for x in paper_data["plan"]]
        title = paper_data["title"]
        abstract = paper_data["abstract"]

        start_index = 0
        total_sections = len(headings)
        plan = super().generate_plan_embeddings(
            paper_data=paper_data,
            headings=headings,
            content=content,
            title=title,
            abstract=abstract,
            start_index=start_index,
            total_sections=total_sections,
            initial_plan=paper_data.get("plan", None),
        )
        return plan

    def extract_plan_and_content(
        self, without_embeddings=False, skip_if_exists: bool = False
    ) -> Dict[str, Any]:
        super().generateOutputFile(without_embeddings=without_embeddings)

        # Get outputfile
        output_file = super().getOutputFile()

        # Skip if exists
        if skip_if_exists and output_file.exists():
            self.getLogger().info(f"\n\tFile already exists: {output_file.absolute()}")
            return {"title": str(input), "content": "File already exists."}

        paper_data = self.extract_paper_data()

        if without_embeddings:
            json_output = json.dumps(paper_data, indent=2)
            json_output = re.sub(
                r'("resources_cited_(id|key)?"\s*:\s*\[\n\s*([^]]+?)\s*\])',
                lambda m: m.group(0).replace("\n", "").replace(" ", ""),
                json_output,
            )
            # Write output
            self.writeOutputString(json_output)
            return paper_data

        # Generate embeddings
        paper_data_with_embeddings = self.generate_embeddings_plan_and_section_content(
            paper_data
        )
        # Write output
        self.writeOuputJson(paper_data_with_embeddings)
        return paper_data_with_embeddings
