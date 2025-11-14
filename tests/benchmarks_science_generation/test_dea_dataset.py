import json
import re
import math
import os
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pytest
import wikipedia
from markdownify import markdownify as md
import pathlib

from tests.benchmarks_science_generation.benchmark.benchmark import load_freshwiki
from tests.benchmarks_science_generation.benchmark.doc_eval import DEA_evaluation

# Add necessary paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
ctb = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../CollabToolBuilder"))
if ctb not in sys.path: sys.path.append(ctb)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.env import EnvironmentManager
from utils.human_llm_config import HumanLLMConfig
from config import embedding_function

TOPIC = "Taylor Hawkins"
RESULT_FOLER = Path("benchmark/results/wikipedia")
WIKI_MD = RESULT_FOLER / f"{TOPIC}.md"
# TARGET_JSON = Path(f"benchmark/dea/output/wikipedia/{TOPIC}.json")
# helper

def wikipedia_to_markdown(title, lang="en"):
    
    wikipedia.set_lang(lang)
    try:
        page = wikipedia.page(title)
        html = page.html()
        markdown = md(html, heading_style="ATX")
        markdown = f"# {page.title}\n\n{markdown}"
        return markdown
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error: {e.options}")
    except wikipedia.exceptions.PageError:
        print("Page not found.")
    return None


# Test fixtures and setup
@pytest.fixture
def test_data_paths():
    """Fixture providing paths to test data files."""
    return {
        'markdown_path': WIKI_MD
    }


@pytest.fixture
def target_data():
    """Fixture providing loaded target JSON data."""
    # TODO: ensure this is the right topic (first one is Taylor Hawkins so for now it work)
    loader = load_freshwiki()
    topic, intent, solution= next(loader)
    len_title_1 = len(solution.get('title_embedding_1', ''))
    len_title_2 = len(solution.get('title_embedding_2', ''))
    return solution


@pytest.fixture
def markdown_content(test_data_paths):
    """Fixture providing loaded markdown content."""
    if not test_data_paths['markdown_path'].exists():
        # If markdown file does not exist, create it from Wikipedia
        markdown_content = wikipedia_to_markdown("Taylor Hawkins")
        if markdown_content:
            with test_data_paths['markdown_path'].open('w', encoding='utf-8') as f:
                f.write(markdown_content)
        else:
            raise FileNotFoundError(f"Could not create markdown content for {test_data_paths['markdown_path']}")
    with test_data_paths['markdown_path'].open('r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def converted_environment(target_data):
    """Fixture providing configured DEA environment."""
    config = HumanLLMConfig()
    config.common_vectordb_config.embedding_function = embedding_function
    llmORchains_list = config.get_llmORchains_list()
    config.initialize()
    
    env = EnvironmentManager(
        env_type="techsynthesis",
        title=target_data["title"],
        context=target_data["abstract"],
        target_file_path=target_data['target_file_path'],
        id=target_data["id"],
        llm=llmORchains_list["default_llm"],
        embedding_model_name=config.common_vectordb_config.embedding_function,
    ).get_environment()
    env.reset()
    return env


@pytest.fixture
def converted_environment(converted_environment, markdown_content):
    """Fixture providing environment after markdown conversion."""
    converted_environment.synthesis_manager.GetFromMarkdown(markdown_content)
    return converted_environment


def test_basic_conversion_validation(converted_environment, target_data):
    """Test 1: Basic conversion validation - sections extraction and basic info."""
    converted_sections = converted_environment.synthesis_manager.get_all_sections()
    
    target_sections_count = len(target_data.get('plan', []))
    converted_sections_count = len(converted_sections)
    title_match = converted_environment.document.title == target_data.get('title', '')
    
    # Assertions
    assert converted_sections_count > 0, "No sections were extracted from markdown"
    assert converted_sections_count >= target_sections_count * 0.5, f"Too few sections extracted: {converted_sections_count} vs target {target_sections_count}"
    
    # Log results for analysis
    print(f"Target sections count: {target_sections_count}")
    print(f"Converted sections count: {converted_sections_count}")
    print(f"Title match: {title_match}")


def test_section_by_section_comparison(converted_environment, target_data):
    """Test 2: Compare sections between target and converted data."""
    converted_sections = converted_environment.synthesis_manager.get_all_sections()
    target_sections = target_data.get('plan', [])
    
    # Extract section titles
    target_titles = [section.get('title', section.get('section', 'NO_TITLE')) for section in target_sections]
    converted_titles = [section.title for section in converted_sections]
    
    # Check for common sections (fuzzy matching)
    common_sections = 0
    for conv_title in converted_titles:
        for target_title in target_titles:
            # Simple word overlap check
            conv_words = set(conv_title.lower().split())
            target_words = set(target_title.lower().split())
            overlap = len(conv_words & target_words) / max(1, len(conv_words))
            if overlap > 0.3:  # 30% word overlap threshold
                common_sections += 1
                break
    
    # Assertions
    assert len(converted_titles) > 0, "No section titles extracted"
    coverage_ratio = common_sections / max(1, len(target_titles))
    assert coverage_ratio >= 0.3, f"Poor section coverage: {coverage_ratio:.2f} (found {common_sections}/{len(target_titles)} sections)"
    
    print(f"Section coverage ratio: {coverage_ratio:.3f}")
    print(f"Common sections found: {common_sections}/{len(target_titles)}")


def test_content_extraction_validation(converted_environment, markdown_content):
    """Test 3: Validate content extraction from markdown."""
    converted_sections = converted_environment.synthesis_manager.get_all_sections()
    
    total_markdown_chars = len(markdown_content)
    total_converted_chars = sum(len(s.content) for s in converted_sections)
    content_ratio = total_converted_chars / total_markdown_chars if total_markdown_chars > 0 else 0
    
    empty_sections = [s for s in converted_sections if not s.content.strip()]
    empty_sections_count = len(empty_sections)
    non_empty_ratio = (len(converted_sections) - empty_sections_count) / max(1, len(converted_sections))
    
    # Assertions
    assert content_ratio > 0.05, f"Extremely low content extraction ratio: {content_ratio:.3f}"
    assert non_empty_ratio >= 0.7, f"Too many empty sections: {empty_sections_count}/{len(converted_sections)}"
    
    print(f"Content extraction ratio: {content_ratio:.3f}")
    print(f"Non-empty sections ratio: {non_empty_ratio:.3f}")
    print(f"Empty sections: {empty_sections_count}")


def test_embedding_consistency_check(converted_environment, target_data):
    """Test 4: Check embedding generation and consistency."""
    converted_sections = converted_environment.synthesis_manager.get_all_sections()
    target_sections = target_data.get('plan', [])
    
    # Check embedding coverage
    sections_with_content_embeddings = [s for s in converted_sections if s.content_embedding]
    sections_with_title_embeddings = [s for s in converted_sections if s.title_embedding]
    
    content_embedding_coverage = len(sections_with_content_embeddings) / max(1, len(converted_sections))
    title_embedding_coverage = len(sections_with_title_embeddings) / max(1, len(converted_sections))
    
    # Check embedding dimensions
    embedding_dim = None
    target_embedding_dim = None
    dimension_match = False
    
    if sections_with_content_embeddings:
        embedding_dim = len(sections_with_content_embeddings[0].content_embedding)
        
        if target_sections:
            target_embedding_key = "content_embedding_2" if "content_embedding_2" in target_sections[0] else "content_embedding_1"
            target_embedding = target_sections[0].get(target_embedding_key, [])
            target_embedding_dim = len(target_embedding)
            dimension_match = embedding_dim == target_embedding_dim
    
    # Assertions
    assert content_embedding_coverage >= 0.8, f"Poor content embedding coverage: {content_embedding_coverage:.3f}"
    assert title_embedding_coverage >= 0.8, f"Poor title embedding coverage: {title_embedding_coverage:.3f}"
    
    if embedding_dim and target_embedding_dim:
        assert dimension_match, f"Embedding dimension mismatch: {embedding_dim} vs {target_embedding_dim}"
    
    print(f"Content embedding coverage: {content_embedding_coverage:.3f}")
    print(f"Title embedding coverage: {title_embedding_coverage:.3f}")
    print(f"Embedding dimensions match: {dimension_match}")


def test_resource_extraction_validation(converted_environment, markdown_content, target_data):
    """Test 5: Validate resource and citation extraction."""
    ressources = converted_environment.document.resources
    total_resources = len(ressources)
    converted_sections = converted_environment.synthesis_manager.get_all_sections()
    sections_with_resources = [s for s in converted_sections if s.resources]
    
    # Count citations in markdown

    citation_patterns = [r'\[\d+\]', r'\[.*?\]\(.*?\)', r'\[\^.*?\]']
    # citation_list = [match for pattern in citation_patterns for match in re.findall(pattern, markdown_content)]
    citation_list = set()
    for pattern in citation_patterns:
        for match in re.findall(pattern, markdown_content):
            cleaned = match.strip("[]")
            if cleaned.isnumeric():
                citation_list.add(int(cleaned))
    # Remove citation numbers that are not in the valid range (0 to len(citation_list)-1)
    valid_citations = set(i for i in citation_list if 0 <= i < len(citation_list))
    citation_list = valid_citations
    target_ressources = target_data.get('resources', [])

    total_citations = len(citation_list)
    target_resources_count = len(target_ressources)
    # Assertions
    if total_citations > 0:
        resource_extraction_ratio = total_resources / total_citations
        assert resource_extraction_ratio >= 0.1, f"Poor resource extraction: {total_resources} resources from {total_citations} citations"
    
    print(f"Total resources extracted: {total_resources}")
    print(f"Sections with resources: {len(sections_with_resources)}")
    print(f"Citations in markdown: {total_citations}")
    print(f"Target resources count: {target_resources_count}")


def test_embedding_similarity_spot_check(converted_environment, target_data):
    """Test 6: Spot check embedding similarities for matched sections."""
    converted_sections = converted_environment.synthesis_manager.get_all_sections()
    target_sections = target_data.get('plan', [])
    
    if not converted_sections or not target_sections:
        pytest.skip("No sections available for similarity check")
    
    similarity_checks = []
    high_similarity_count = 0
    
    for conv_section in converted_sections[:3]:  # Check first 3 sections
        best_match = None
        best_title_similarity = -1
        
        for target_section in target_sections:
            target_title = target_section.get('title', target_section.get('section', ''))
            title_words_conv = set(conv_section.title.lower().split())
            title_words_target = set(target_title.lower().split())
            title_similarity = len(title_words_conv & title_words_target) / max(1, len(title_words_conv))
            
            if title_similarity > best_title_similarity:
                best_title_similarity = title_similarity
                best_match = target_section
        
        if best_match and best_title_similarity > 0.3:  # Reasonable title match
            # Check embedding similarity if both exist
            if conv_section.content_embedding and best_match:
                target_embedding_key = "content_embedding_2" if "content_embedding_2" in best_match else "content_embedding_1"
                target_embedding = best_match.get(target_embedding_key)
                
                if target_embedding:
                    similarity = cosine_similarity([conv_section.content_embedding], [target_embedding])[0][0]
                    similarity_checks.append({
                        'title_similarity': best_title_similarity,
                        'content_similarity': similarity
                    })
                    
                    if similarity > 0.5:  # Reasonable content similarity
                        high_similarity_count += 1
                    
                    print(f"Section '{conv_section.title}': title_sim={best_title_similarity:.3f}, content_sim={similarity:.3f}")
    
    # Assertions
    if similarity_checks:
        avg_content_similarity = sum(check['content_similarity'] for check in similarity_checks) / len(similarity_checks)
        assert avg_content_similarity > 0.0, f"All content similarities are negative or zero: {avg_content_similarity:.3f}"
        
        # At least some sections should have reasonable similarity
        assert high_similarity_count > 0, f"No sections with decent similarity found ({len(similarity_checks)} checked)"
    
    print(f"Similarity checks performed: {len(similarity_checks)}")
    print(f"High similarity sections: {high_similarity_count}")


def test_plan_embedding_update_check(converted_environment):
    """Test 7: Check plan-level embedding generation."""
    # Force update plan embeddings
    converted_environment.synthesis_manager.document.update_plan_embedding()
    
    plan_embedding = converted_environment.synthesis_manager.document.document_content.sections_list_embedding
    plan_titles_embedding = converted_environment.synthesis_manager.document.document_content.sections_list_title_embedding
    plan_contents_embedding = converted_environment.synthesis_manager.document.document_content.sections_list_content_embedding
    
    plan_embedding_exists = plan_embedding is not None and len(plan_embedding) > 0
    plan_titles_embedding_exists = plan_titles_embedding is not None and len(plan_titles_embedding) > 0
    plan_contents_embedding_exists = plan_contents_embedding is not None and len(plan_contents_embedding) > 0
    
    # Assertions
    assert plan_embedding_exists, "Plan embedding not generated"
    
    print(f"Plan embedding generated: {plan_embedding_exists}")
    print(f"Plan titles embedding generated: {plan_titles_embedding_exists}")
    print(f"Plan contents embedding generated: {plan_contents_embedding_exists}")
    
    if plan_embedding_exists:
        print(f"Plan embedding dimension: {len(plan_embedding)}")


def test_distance_calculation_validation(converted_environment):
    """Test 8: Validate distance calculation metrics."""
    # Get the actual distance scores
    distance_scores = converted_environment.synthesis_manager.get_distance_to_targetJSON()
    
    # Check for NaN or invalid values
    invalid_scores = {}
    for k, v in distance_scores.items():
        if not isinstance(v, (int, float)) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            invalid_scores[k] = v
    
    # Assertions
    assert len(distance_scores) > 0, "No distance scores calculated"
    assert len(invalid_scores) == 0, f"Invalid scores detected: {invalid_scores}"
    
    # Check that key metrics exist
    expected_metrics = ['plan_embedding_similarity', 'current_sections_count', 'sections_count_ratio_to_target']
    missing_metrics = [metric for metric in expected_metrics if metric not in distance_scores]
    assert len(missing_metrics) == 0, f"Missing expected metrics: {missing_metrics}"
    
    print("Distance scores calculated successfully:")
    for key, value in distance_scores.items():
        print(f"  {key}: {value}")


def test_raw_similarity_comparison(converted_environment, target_data):
    """Test 9: Compare raw embedding similarities."""
    # Force update plan embeddings
    converted_environment.synthesis_manager.document.update_plan_embedding()
    plan_embedding = converted_environment.synthesis_manager.document.document_content.sections_list_embedding
    
    target_plan_embedding = target_data.get('plan_embedding_2', target_data.get('plan_embedding_1'))
    
    if not (target_plan_embedding and plan_embedding):
        pytest.skip("Plan embeddings not available for comparison")
    
    raw_similarity = cosine_similarity([plan_embedding], [target_plan_embedding])[0][0]
    
    # Check normalization
    min_cs = getattr(converted_environment.synthesis_manager, 'min_cosine_similarity', -1)
    normalized_similarity = (raw_similarity - min_cs) / (1 - min_cs) if min_cs != 1 else 0
    
    # Assertions
    assert not math.isnan(raw_similarity), "Raw similarity is NaN"
    assert not math.isinf(raw_similarity), "Raw similarity is infinite"
    assert -1 <= raw_similarity <= 1, f"Raw similarity out of valid range: {raw_similarity}"
    
    print(f"Raw plan embedding similarity: {raw_similarity:.3f}")
    print(f"Min cosine similarity threshold: {min_cs:.3f}")
    print(f"Normalized similarity: {normalized_similarity:.3f}")


def test_simple_markdown_conversion():
    """Test 10: Simple markdown conversion test with known input."""
    simple_markdown = """# Test Title

This is the abstract content.

## Section 1
Content for section 1 with [citation](http://example.com).

## Section 2  
Content for section 2 with more text.
"""
    
    # Setup simple environment
    config = HumanLLMConfig()
    config.common_vectordb_config.embedding_function = embedding_function
    config.initialize()
    
    env = EnvironmentManager(
        env_type="techsynthesis",
        title="Test",
        context="",
        target_file_path="",
        id="simple_test",
        llm=config.get_llmORchains_list()["default_llm"],
        embedding_model_name="text-embedding-ada-002"
    ).get_environment()
    
    env.reset()
    env.synthesis_manager.GetFromMarkdown(simple_markdown)
    simple_sections = env.synthesis_manager.get_all_sections()
    
    # Assertions
    assert len(simple_sections) >= 2, f"Expected at least 2 sections, got {len(simple_sections)}"
    assert env.document.title == "Test Title", f"Title mismatch: '{env.document.title}' vs 'Test Title'"
    
    # Check section titles
    section_titles = [s.title for s in simple_sections]
    assert "Section 1" in section_titles, f"Section 1 not found in {section_titles}"
    assert "Section 2" in section_titles, f"Section 2 not found in {section_titles}"
    
    print(f"Simple test - sections extracted: {len(simple_sections)}")
    print(f"Simple test - title: '{env.document.title}'")
    print(f"Simple test - section titles: {section_titles}")


# Integration test that runs multiple diagnostics
def test_comprehensive_dea_conversion(converted_environment,markdown_content):
    """Comprehensive test that runs key diagnostics and provides overall assessment."""

    
    # Get key metrics
    converted_sections = converted_environment.synthesis_manager.get_all_sections()
    distance_scores = converted_environment.synthesis_manager.get_distance_to_targetJSON()
    
    # Calculate overall health score
    issues = 0
    
    # Check section count
    target_sections_count = len(target_data.get('plan', []))
    sections_ratio = len(converted_sections) / max(1, target_sections_count)
    if sections_ratio < 0.5:
        issues += 1
    
    # Check content extraction
    total_markdown_chars = len(markdown_content)
    total_converted_chars = sum(len(s.content) for s in converted_sections)
    content_ratio = total_converted_chars / total_markdown_chars if total_markdown_chars > 0 else 0
    if content_ratio < 0.1:
        issues += 1
    
    # Check embedding coverage
    sections_with_embeddings = [s for s in converted_sections if s.content_embedding]
    embedding_coverage = len(sections_with_embeddings) / max(1, len(converted_sections))
    if embedding_coverage < 0.8:
        issues += 1
    
    # Check distance scores validity
    invalid_scores = sum(1 for v in distance_scores.values() 
                        if not isinstance(v, (int, float)) or 
                        (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))
    if invalid_scores > 0:
        issues += 1
    
    # Overall assessment
    health = 'GOOD' if issues == 0 else 'FAIR' if issues <= 2 else 'POOR'
    
    # Assertions for critical failures
    assert len(converted_sections) > 0, "Critical: No sections extracted"
    assert content_ratio > 0.01, f"Critical: Extremely low content extraction: {content_ratio:.4f}"
    assert invalid_scores == 0, f"Critical: Invalid distance scores: {invalid_scores}"
    
    print(f"\n=== COMPREHENSIVE TEST RESULTS ===")
    print(f"Overall Health: {health}")
    print(f"Issues Found: {issues}")
    print(f"Sections Extracted: {len(converted_sections)} (target: {target_sections_count})")
    print(f"Content Extraction Ratio: {content_ratio:.3f}")
    print(f"Embedding Coverage: {embedding_coverage:.3f}")
    print(f"Distance Scores Valid: {len(distance_scores) - invalid_scores}/{len(distance_scores)}")
    
    return {
        'health': health,
        'issues': issues,
        'sections_extracted': len(converted_sections),
        'content_ratio': content_ratio,
        'embedding_coverage': embedding_coverage,
        'distance_scores': distance_scores
    }
    
def test_DEA_evaluation(target_data, markdown_content):
    """Test 11: Run DEA evaluation on the converted environment."""
    # Run DEA evaluation
    scores = DEA_evaluation(content=markdown_content, solution=target_data, content_type="markdown")
    print()
    for k, v in scores.items():
        print(f"{k}: {v}")
      


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])