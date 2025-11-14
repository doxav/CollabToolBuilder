import unittest
from unittest.mock import patch, Mock
from env import DocumentStructure, SynthesisManager, Section, Document

class TestDocumentStructure(unittest.TestCase):

    def setUp(self):
        self.doc = DocumentStructure(
            synthesis_type="example_type",
            initial_goal="example_goal",
            title="example_title",
            context="example_context"
        )

    def test_init(self):
        self.assertEqual(self.doc.synthesis_type, "example_type")
        self.assertEqual(self.doc.initial_goal, "example_goal")

    # def test_get_embedding(self):
    #     with patch.object(self.doc.embedding_model, 'embed_query', return_value=[1.0]):
    #         embedding = self.doc.get_embedding("test")
    #         self.assertEqual(embedding, [1.0])

    def test_set_plan_field_with_embedding(self):
        self.doc.set_plan_field_with_embedding("title", "test_title")
        self.assertEqual(self.doc.document_content.title, "test_title")

    def test_update_sections_embeddings(self):
        # Test by adding a section and checking if embeddings are generated
        self.doc.document_content.sections_list.append(Section(
            section_id= 1,
            parent_id= 0,
            title = 'Test title',
            #title_embedding': [],
            content = 'Test content',
            #content_embedding': [],
            title_validation_status = 0,
            content_progress_validation_status = 0,
            #local_feedback_to_process': [],
            #local_feedback_processed = [],
        ))
        self.doc.update_sections_embeddings()
        self.assertNotEqual([], self.doc.document_content.sections_list[0].title_embedding)
        self.assertNotEqual([], self.doc.document_content.sections_list[0].content_embedding)

    def test_get_state(self):
        state = self.doc.get_state()
        self.assertIsInstance(state, dict)
        self.assertIn('synthesis_type', state)
        
    def test_restore_state(self):
        initial_state = self.doc.get_state(save=True)
        self.doc.synthesis_type = "changed_type"
        self.doc.restore_state()
        self.assertEqual(self.doc.synthesis_type, initial_state['synthesis_type'])
        
    def test_reset(self):
        self.doc.reset()
        self.assertEqual(self.doc.document_content.title, '')
        self.assertEqual(self.doc.document_content.context, '')
        self.assertEqual(len(self.doc.document_content.sections_list), 0)

    # Mocking embedding retrieval
    @patch('env.DocumentStructure.get_embedding', return_value=[0.5, 0.5])
    def test_add_event(self, mock_embedding):
        initial_event_count = len(self.doc.events)
        self.doc.add_event({'action': 'test_action'})
        self.assertEqual(len(self.doc.events), initial_event_count + 1)

    @patch('env.DocumentStructure.get_embedding', return_value=[0.5, 0.5])
    def test_update_plan_embedding(self, mock_embedding):
        # Given a section with content but no content embedding
        self.doc.document_content.sections_list.append(Section(section_id=1, parent_id=0, title='Test title', content='Test content', title_validation_status=0, content_progress_validation_status=0))
        self.doc.update_sections_embeddings(batch_update=True, force_update=False)
        self.doc.update_plan_embedding()
        # Expect the sections_list_embedding to be updated
        self.assertNotEqual([], self.doc.document_content.sections_list_embedding)

    @patch('env.DocumentStructure.get_embedding', return_value=[0.5, 0.5])
    def test_get_embedding(self, mock_embedding):
        # Getting embedding for the text "Hello"
        result = self.doc.get_embedding("Hello")
        # The result should match the mock embedding
        self.assertEqual(result, [0.5, 0.5])

class TestSynthesisManager(unittest.TestCase):

    def setUp(self):
        doc = DocumentStructure(
            synthesis_type="example_type",
            initial_goal="example_goal",
            title="example_title",
            context="example_context"
        )
        self.manager = SynthesisManager(doc)

    def test_validate_section_format(self):
        valid_section = {
            'section_id': 1,
            'parent_id': 0,
            'title': 'Test title',
            'title_embedding': [0.1, 0.2, 0.3],
            'content': 'Test content',
            'content_embedding': [0.1, 0.2, 0.3],
            'title_validation_status': 0,
            'content_progress_validation_status': 0,
            'local_feedback_to_process': [],
            'local_feedback_processed': [],
        }
        self.assertTrue(self.manager.validate_section_format(valid_section))

        invalid_section = {
            'section_id': 1,
            'parent_id': 'invalid',  # <-- Invalid type
            # ... rest remains the same
        }
        self.assertFalse(self.manager.validate_section_format(invalid_section))

    def test_add_section(self):
        section = Section(section_id=2, parent_id=0, title='Another title', content='Another content')
        self.manager.add_section(section)
        self.assertEqual(len(self.manager.document.document_content.sections_list), 1)
        self.assertEqual(self.manager.document.document_content.sections_list[0].title, 'Another title')


    def test_add_sections(self):
        section1 = Section(section_id=2, parent_id=0, title='Title 1', content='Content 1')
        section2 = Section(section_id=3, parent_id=0, title='Title 2', content='Content 2')
        self.manager.add_sections([section1, section2])
        self.assertEqual(len(self.manager.document.document_content.sections_list), 2)
        self.assertEqual(self.manager.document.document_content.sections_list[1].title, 'Title 2')

    def test_remove_section(self):
        section = Section(section_id=2, parent_id=0, title='Another title', content='Another content')
        self.manager.document.document_content.sections_list.append(section)
        self.manager.remove_section(2)
        self.assertEqual(len(self.manager.document.document_content.sections_list), 0)



    def test_edit_content_text(self):
        section = Section(section_id=2, parent_id=0, title='Another title', content='Another content')
        self.manager.document.document_content.sections_list.append(section)
        self.manager.edit_section(2, 'Edited content')
        self.assertEqual(self.manager.document.document_content.sections_list[0].content, 'Edited content')
        
    # Mocking embedding retrieval
    @patch('env.DocumentStructure.get_embedding', return_value=[0.5, 0.5])
    def test_rate_section_content_progress_validation_status(self, mock_embedding):
        section = Section(section_id=2, parent_id=0, title='Another title', content='Another content')
        self.manager.document.document_content.sections_list.append(section)
        self.manager.rate_section_content_progress_validation_status(2, 5)
        self.assertEqual(self.manager.document.document_content.sections_list[0].content_progress_validation_status, 5)

    def test_get_sections(self):
        section1 = Section(section_id=2, parent_id=0, title='Title 1', content='Content 1')
        section2 = Section(section_id=3, parent_id=0, title='Title 2', content='Content 2')
        self.manager.document.document_content.sections_list.extend([section1, section2])
        sections = self.manager.get_sections([2, 3])
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, 'Title 1')

    def test_remove_sections(self):
        section1 = Section(section_id=2, parent_id=0, title='Title 1', content='Content 1')
        section2 = Section(section_id=3, parent_id=0, title='Title 2', content='Content 2')
        self.manager.document.document_content.sections_list.extend([section1, section2])
        self.manager.remove_sections([2, 3])
        self.assertEqual(len(self.manager.document.document_content.sections_list), 0)

    # Note: For split_section, it's a bit complex. A simple test is added below, but you might want more cases.
    def test_split_section(self):
        section = Section(section_id=2, parent_id=0, title='Original Title', content='First Half Second Half')
        self.manager.document.document_content.sections_list.append(section)
        self.manager.split_section(2, 'New Title', 10)  # splitting by index
        self.assertEqual(self.manager.document.document_content.sections_list[0].content, 'First Half')
        self.assertEqual(self.manager.document.document_content.sections_list[1].title, 'New Title')

    def test_add_section_feedback_to_process(self):
        section = Section(section_id=2, parent_id=0, title='Title', content='Content')
        self.manager.document.document_content.sections_list.append(section)
        self.manager.add_section_feedback_to_process(2, 'Feedback 1')
        self.assertEqual(len(self.manager.document.document_content.sections_list[0].local_feedback_to_process), 1)
        self.assertEqual(self.manager.document.document_content.sections_list[0].local_feedback_to_process[0], 'Feedback 1')

    def test_get_section_feedback_to_process(self):
        section = Section(section_id=2, parent_id=0, title='Title', content='Content', local_feedback_to_process=['Feedback 1', 'Feedback 2'])
        self.manager.document.document_content.sections_list.append(section)
        feedbacks = self.manager.get_section_feedback_to_process(2)
        self.assertEqual(len(feedbacks), 2)
        self.assertIn('Feedback 1', feedbacks)

    def test_set_section_feedback_processed(self):
        section = Section(section_id=2, parent_id=0, title='Title', content='Content', local_feedback_to_process=['Feedback 1'])
        self.manager.document.document_content.sections_list.append(section)
        self.manager.set_section_feedback_processed(2, 'Feedback 1')
        self.assertEqual(len(self.manager.document.document_content.sections_list[0].local_feedback_processed), 1)
        self.assertIn('Feedback 1', self.manager.document.document_content.sections_list[0].local_feedback_processed)

    def test_add_global_feedback_to_process(self):
        self.manager.add_global_feedback_to_process('Global Feedback 1')
        self.assertIn('Global Feedback 1', self.manager.document.global_feedback_to_process)

    def test_get_global_feedback_to_process(self):
        self.manager.document.global_feedback_to_process = ['Global Feedback 1']
        feedbacks = self.manager.get_global_feedback_to_process()
        self.assertIn('Global Feedback 1', feedbacks)

    def test_set_global_feedback_processed(self):
        self.manager.document.global_feedback_to_process = ['Global Feedback 1']
        self.manager.set_global_feedback_processed('Global Feedback 1')
        self.assertIn('Global Feedback 1', self.manager.document.global_feedback_processed)

    def test_get_resources(self):
        resources = self.manager.get_all_resources()
        self.assertIsInstance(resources, list)  # assuming it returns a list

    def test_add_resources(self):
        self.manager.add_or_update_result_in_resources('Resource Name', ['Tag 1'], 'Link 1')
        self.assertEqual(self.manager.document.resources[0]['name'], 'Resource Name')

    def test_remove_resources(self):
        self.manager.document.resources = [{'id': 1, 'name': 'Resource 1', 'tags': [], 'content_or_link': 'Link 1'}]
        self.manager.remove_resource(1)
        self.assertEqual(len(self.manager.document.resources), 0)

    def test_restore_last_state(self):
        state = self.manager.restore_last_state()
        # Add some assertions depending on what you expect restore_state() to do.

    def test_list_all_previous_document_events(self):
        events = self.manager.list_all_previous_document_events()
        self.assertIsInstance(events, list)  # assuming it returns a list

if __name__ == "__main__":
    unittest.main()
