import gc
import unittest
import json
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import asdict
from pathlib import Path

from env import DocumentStructure, SynthesisManager, Section, Document

import openai
EMBED_POSTFIX = "_2"

class TestSynthesisManagerLoading(unittest.TestCase):

    @unittest.skip("de-activate temporarely")
    def test_loading(self):
        target_path = "voyager/env/IR_CPS_TechSynthesis/document_embedding_analysis/output/wikipedia/Climate Change.json"

        # Initialize the DocumentStructure
        document = DocumentStructure(
            synthesis_type="test",
            initial_goal="Load data",
            title="title not loaded",
            context="context not loaded",
            embedding_model_name="intfloat/e5-base-v2",
            embedding_model_query_prefix="query: ",
        )

        # Initialize SynthesisManager
        sm = SynthesisManager(document)

        sm.set_targetJSON_comparison(target_path)
        data = sm.target_data
        sm.document.set_plan_field_with_embedding("title", data["title"])
        sm.document.set_plan_field_with_embedding("context", data["abstract"])

        # Measure target sections and total content length
        current_plan_sections_count, current_content_length, target_total_content_length = 0, 0, 0
        plan_length = len(data["plan"])
        for section in data["plan"]:
            target_total_content_length += len(section["content"])

        # Load sections from the JSON data
        for plan in data["plan"]:
            section_data = {
                "section_id": plan["section_id"],
                "parent_id": None,  # If parent_id is not available in the JSON
                "title": plan["section"],
                "content": plan["content"]
            }
            current_plan_sections_count += 1
            current_content_length += len(plan["content"])
            sm.add_section(Section(**section_data))
            plan_embedding = document.document_content.sections_list_embedding
            #print(f"plan similarity to target afer processing {current_plan_sections_count}/{plan_length} - " +str(round(cosine_similarity([plan_embedding], [data["plan_embedding"+EMBED_POSTFIX]])[0][0], 6)) + f" cosine similarity - total length: {int(current_content_length/target_total_content_length*100)}%")
            print(sm.get_distance_to_targetJSON(target_section_title_embedding_label = "section_embedding"+EMBED_POSTFIX, target_section_content_embedding_label = "content_embedding"+EMBED_POSTFIX, target_plan_embedding_label = "plan_embedding"+EMBED_POSTFIX))


        print(document.document_content.title + " => " + data["title"])

        # Compare text content
        self.assertEqual(document.document_content.title, data["title"])
        self.assertEqual(document.document_content.context, data["abstract"])

        # Compare the plan embeddings
        title_embedding = document.document_content.title_embedding
        abstract_embedding = document.document_content.context_embedding
        plan_embedding = document.document_content.sections_list_embedding
        self.assertTrue(cosine_similarity([title_embedding], [data["title_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)
        self.assertTrue(cosine_similarity([abstract_embedding], [data["abstract_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)
        self.assertTrue(cosine_similarity([plan_embedding], [data["plan_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)

        for i, section in enumerate(document.document_content.sections_list):
            #print(section.title + " => " + data["plan"][i]["section"])
            self.assertEqual(section.title, data["plan"][i]["section"])
            self.assertEqual(section.content, data["plan"][i]["content"])

        for i, section in enumerate(document.document_content.sections_list):
            section_embedding = section.title_embedding
            content_embedding = section.content_embedding
            #print(title +  " => " + data["plan"][i]["section"])
            #print(cosine_similarity([section_embedding], [data["plan"][i]["section_embedding"+EMBED_POSTFIX]])[0][0])
            #print(cosine_similarity([content_embedding], [data["plan"][i]["content_embedding"+EMBED_POSTFIX]])[0][0])
            self.assertTrue(cosine_similarity([section_embedding], [data["plan"][i]["section_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)
            self.assertTrue(cosine_similarity([content_embedding], [data["plan"][i]["content_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)

        print(f"Final plan similarity to target afer processing {current_plan_sections_count}/{plan_length} - " +str(round(cosine_similarity([plan_embedding], [data["plan_embedding"+EMBED_POSTFIX]])[0][0], 6)) + f" cosine similarity - total length: {int(current_content_length/target_total_content_length*100)}%")

    def test_loading_title_first(self):
        target_path = "voyager/env/IR_CPS_TechSynthesis/document_embedding_analysis/output/wikipedia/Climate Change.json"

        # Initialize the DocumentStructure
        document = DocumentStructure(
            synthesis_type="test",
            initial_goal="Load data",
            title="title not loaded",
            context="context not loaded",
            embedding_model_name="intfloat/e5-base-v2", # "text-embedding-ada-002" or "intfloat/e5-base-v2"
            embedding_model_query_prefix="query: ",  # "query: " for e5-base-v2
        )

        # Initialize SynthesisManager
        sm = SynthesisManager(document)

        sm.set_targetJSON_comparison(target_path)
        data = sm.target_data
        sm.document.set_plan_field_with_embedding("title", data["title"])
        sm.document.set_plan_field_with_embedding("context", data["abstract"])

        # Measure target sections and total content length
        current_plan_sections_count, current_content_length, target_total_content_length = 0, 0, 0
        plan_length = len(data["plan"])
        for section in data["plan"]:
            target_total_content_length += len(section["content"])

        print(document.document_content.title + " => " + data["title"])

        # Load sections from the JSON data
        for plan in data["plan"]:
            section_data = {
                "section_id": int(plan["section_id"]),
                "parent_id": None,  # If parent_id is not available in the JSON
                "title": plan["section"],
                "content": ""
            }
            current_plan_sections_count += 1
            sm.add_section(Section(**section_data))
            plan_embedding = document.document_content.sections_list_embedding
            #print(f"plan similarity to target afer processing {current_plan_sections_count}/{plan_length} - " +str(round(cosine_similarity([plan_embedding], [data["plan_embedding"+EMBED_POSTFIX]])[0][0], 6)) + f" cosine similarity - total length: {int(current_content_length/target_total_content_length*100)}%")
            print(sm.get_distance_to_targetJSON(target_section_title_embedding_label = "section_embedding"+EMBED_POSTFIX, target_section_content_embedding_label = "content_embedding"+EMBED_POSTFIX, target_plan_embedding_label = "plan_embedding"+EMBED_POSTFIX))
            #print(sm.get_plan_status())

        for plan in data["plan"]:
            sm.document.set_plan_field_with_embedding("content", plan["content"], section_id=plan["section_id"])
            current_content_length += len(plan["content"])
            sm.document.update_plan_embedding()
            print(sm.get_distance_to_targetJSON(target_section_title_embedding_label = "section_embedding"+EMBED_POSTFIX, target_section_content_embedding_label = "content_embedding"+EMBED_POSTFIX, target_plan_embedding_label = "plan_embedding"+EMBED_POSTFIX))

        # Compare text content
        self.assertEqual(document.document_content.title, data["title"])
        self.assertEqual(document.document_content.context, data["abstract"])

        # Compare the plan embeddings
        title_embedding = document.document_content.title_embedding
        abstract_embedding = document.document_content.context_embedding
        plan_embedding = document.document_content.sections_list_embedding
        self.assertTrue(cosine_similarity([title_embedding], [data["title_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)
        self.assertTrue(cosine_similarity([abstract_embedding], [data["abstract_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)
        self.assertTrue(cosine_similarity([plan_embedding], [data["plan_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)

        for i, section in enumerate(document.document_content.sections_list):
            #print(section.title + " => " + data["plan"][i]["section"])
            self.assertEqual(section.title, data["plan"][i]["section"])
            self.assertEqual(section.content, data["plan"][i]["content"])

        for i, section in enumerate(document.document_content.sections_list):
            section_embedding = section.title_embedding
            content_embedding = section.content_embedding
            self.assertTrue(cosine_similarity([section_embedding], [data["plan"][i]["section_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)
            self.assertTrue(cosine_similarity([content_embedding], [data["plan"][i]["content_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)

        print(f"Final plan similarity to target afer processing {current_plan_sections_count}/{plan_length} - " +str(round(cosine_similarity([plan_embedding], [data["plan_embedding"+EMBED_POSTFIX]])[0][0], 6)) + f" cosine similarity - total length: {int(current_content_length/target_total_content_length*100)}%")

    @unittest.skip("long test, de-activate temporarely")
    def test_loading_all_files(self):
        directories = [
            'voyager/env/IR_CPS_TechSynthesis/document_embedding_analysis/output/wikipedia',
            'voyager/env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv',
            'voyager/env/IR_CPS_TechSynthesis/document_embedding_analysis/output/patent'
        ]
        
        for directory in directories:
            for file_path in Path(directory).glob('*.json'):
                with self.subTest(file=str(file_path)):
                    self.load_and_test(file_path)
                gc.collect()  # manually run the garbage collector

    def load_and_test(self, target_path: str):
        document=DocumentStructure(synthesis_type='test', initial_goal='Load data', title='title not loaded', context='context not loaded', embedding_model_name='intfloat/e5-base-v2', embedding_model_query_prefix='query: ')
        sm = SynthesisManager(document)
        sm.set_targetJSON_comparison(target_path)
        data = sm.target_data
        sm.document.set_plan_field_with_embedding('title', data['title'])
        sm.document.set_plan_field_with_embedding('context', data['abstract'])
        current_plan_sections_count, current_content_length, target_total_content_length = 0, 0, 0
        plan_length = len(data['plan'])

        for section in data["plan"]:
            target_total_content_length += len(section["content"])

        # Load sections from the JSON data
        for plan in data["plan"]:
            section_data = {
                "section_id": plan["section_id"],
                "parent_id": None,  # If parent_id is not available in the JSON
                "title": plan["section"],
                "content": plan["content"]
            }
            current_plan_sections_count += 1
            current_content_length += len(plan["content"])
            sm.add_section(Section(**section_data))
            plan_embedding = document.document_content.sections_list_embedding
            #print(f"plan similarity to target afer processing {current_plan_sections_count}/{plan_length} - " +str(round(cosine_similarity([plan_embedding], [data["plan_embedding"+EMBED_POSTFIX]])[0][0], 6)) + f" cosine similarity - total length: {int(current_content_length/target_total_content_length*100)}%")
            #print(sm.get_distance_to_targetJSON(target_section_title_embedding_label = "section_embedding"+EMBED_POSTFIX, target_section_content_embedding_label = "content_embedding"+EMBED_POSTFIX, target_plan_embedding_label = "plan_embedding"+EMBED_POSTFIX))

        print(document.document_content.title)

        # Compare text content
        self.assertEqual(document.document_content.title, data["title"])
        self.assertEqual(document.document_content.context, data["abstract"])

        # Compare the plan embeddings
        title_embedding = document.document_content.title_embedding
        abstract_embedding = document.document_content.context_embedding
        plan_embedding = document.document_content.sections_list_embedding
        self.assertTrue(cosine_similarity([title_embedding], [data["title_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)
        self.assertTrue(cosine_similarity([abstract_embedding], [data["abstract_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)
        self.assertTrue(cosine_similarity([plan_embedding], [data["plan_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)

        for i, section in enumerate(document.document_content.sections_list):
            #print(section.title + " => " + data["plan"][i]["section"])
            self.assertEqual(section.title, data["plan"][i]["section"])
            self.assertEqual(section.content, data["plan"][i]["content"])

        for i, section in enumerate(document.document_content.sections_list):
            section_embedding = section.title_embedding
            content_embedding = section.content_embedding
            #print(title +  " => " + data["plan"][i]["section"])
            #print(cosine_similarity([section_embedding], [data["plan"][i]["section_embedding"+EMBED_POSTFIX]])[0][0])
            #print(cosine_similarity([content_embedding], [data["plan"][i]["content_embedding"+EMBED_POSTFIX]])[0][0])
            self.assertTrue(cosine_similarity([section_embedding], [data["plan"][i]["section_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)
            self.assertTrue(cosine_similarity([content_embedding], [data["plan"][i]["content_embedding"+EMBED_POSTFIX]])[0][0] > 0.95)

        print(sm.get_distance_to_targetJSON(target_section_title_embedding_label = "section_embedding"+EMBED_POSTFIX, target_section_content_embedding_label = "content_embedding"+EMBED_POSTFIX, target_plan_embedding_label = "plan_embedding"+EMBED_POSTFIX))

        del document  # Delete the document object explicitly
        del sm  # Delete the SynthesisManager object explicitly
        del data  # Delete the data dictionary explicitly

if __name__ == '__main__':
    unittest.main()
