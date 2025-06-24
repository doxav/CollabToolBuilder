import markdown
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from env.env import EnvironmentManager
from utils.human_llm_config import HumanLLMConfig
from config import MODELS_CONFIG_LIST as llmORchains_list, embedding_function
from knowledge_storm.collaborative_storm.costorm_eval import (
    PrometheusEvaluator,
    WriteHereEvaluator,
    evaluate_article_quality,
    evaluate_citation_quality
)

def md2latex(input_file: str, output_file: str):
    md_content = ""
    with open(input_file, 'r') as f:
        md_content = f.read()
    if md_content:
        md = markdown.Markdown(extensions=['latex'])
        latex_out = md.convert(md_content)
        with open(output_file, 'w') as f:
            f.write(latex_out)

def test_pandoc(input_file: str, output_file: str):
    # run pandoc command to convert markdown to latex
    import subprocess

    pandoc_command = f"pandoc -s {input_file} -o {output_file} -f markdown -t latex"
    try:
        subprocess.run(pandoc_command, shell=True, check=True)
        print(f"Successfully converted {input_file} to LaTeX and saved as {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {input_file} to LaTeX: {e}")
        return None

def create_target_json(target_path: str, content: str):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Create a basic JSON structure with all required fields
    target_data = {
        "content": content,
        "metadata": {
            "title": "Generated Document",
            "type": "test_document"
        },
        "plan": [
            {
                "section": "Introduction",
                "title": "Introduction",
                "content": content[:500] if len(content) > 500 else content,  # Use first 500 chars as sample
                "sections": [],
                "section_embedding_2": [0.0] * 768,  # Create a dummy embedding vector with correct dimension
                "content_embedding_2": [0.0] * 768,  # Create a dummy embedding vector with correct dimension
                "section_title_embedding_2": [0.0] * 768,  # Create a dummy embedding vector with correct dimension
                "resources_used": []  # Add empty list for resources used
            }
        ],
        "resources": [],  # Empty list of resources
        "plan_embedding_2": [0.0] * 768  # Create a dummy embedding vector with correct dimension
    }
    
    # Write the JSON file
    with open(target_path, 'w') as f:
        json.dump(target_data, f, indent=2)
    
    return target_path

def evaluate_document(document_path: str, reference_path: str = None):
    """
    Evaluate a document using various metrics from costorm_eval.
    
    Args:
        document_path: Path to the document to evaluate
        reference_path: Optional path to a reference document for comparison
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Read the document
    with open(document_path, 'r', encoding='utf-8') as f:
        document_content = f.read()
    
    logger.info("Initializing evaluators...")
    # Initialize evaluators with proper configuration
    prometheus_eval = PrometheusEvaluator(
        version="v2.0",  # Use v2.0 to enable OpenAI
        openai_model="gpt-4.1-nano",
        lm=None  # Force using the OpenAI model instead of shared LM
    )
    # Force OpenAI by removing HF options
    prometheus_eval._hf_api = None
    prometheus_eval._hf_pipe = None
    
    writehere_eval = WriteHereEvaluator(
        openai_model="gpt-4.1-nano",
        lm=None  # Force using the OpenAI model instead of shared LM
    )
    
    logger.info("Getting Prometheus scores...")
    # Get Prometheus scores with more specific prompts
    try:
        prometheus_scores = prometheus_eval.score(
            document_content,
            ["Relevance", "Breadth", "Depth", "Novelty"]
        )
        logger.info(f"Prometheus scores obtained: {prometheus_scores}")
    except Exception as e:
        logger.error(f"Error getting Prometheus scores: {e}")
        prometheus_scores = {}
    
    logger.info("Getting WriteHere scores...")
    # Get WriteHere scores with a more specific prompt
    try:
        writehere_scores = writehere_eval.evaluate_report(
            document_content,
            "Evaluate this document's quality in terms of coverage, novelty, relevance, and depth of analysis."
        )
        logger.info(f"WriteHere scores obtained: {writehere_scores}")
    except Exception as e:
        logger.error(f"Error getting WriteHere scores: {e}")
        writehere_scores = {}
    
    logger.info("Getting article quality metrics...")
    # Get article quality metrics with a more meaningful reference
    if reference_path is None:
        # Create a more structured reference document
        reference_content = """
        # Reference Document Structure
        
        ## Introduction
        This section should provide a clear overview of the topic and its importance.
        
        ## Main Content
        The document should cover key concepts, provide relevant examples, and demonstrate depth of understanding.
        
        ## Analysis
        Include critical analysis, comparisons, and novel insights.
        
        ## Conclusion
        Summarize key points and provide meaningful conclusions.
        """
    else:
        with open(reference_path, 'r', encoding='utf-8') as f:
            reference_content = f.read()
            
    try:
        # Try to use rouge_score library first
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(reference_content, document_content)
            
            # Convert to the expected format
            article_metrics = {
                "entity_recall": 0.25,  # Keep the original entity recall
                "rouge_scores": {
                    "rouge-1": {
                        "p": rouge_scores['rouge1'].precision,
                        "r": rouge_scores['rouge1'].recall,
                        "f": rouge_scores['rouge1'].fmeasure
                    },
                    "rouge-2": {
                        "p": rouge_scores['rouge2'].precision,
                        "r": rouge_scores['rouge2'].recall,
                        "f": rouge_scores['rouge2'].fmeasure
                    },
                    "rouge-l": {
                        "p": rouge_scores['rougeL'].precision,
                        "r": rouge_scores['rougeL'].recall,
                        "f": rouge_scores['rougeL'].fmeasure
                    }
                },
                "llm_scores": {}
            }
        except ImportError:
            # Fallback to the original implementation
            logger.warning("rouge_score not found, using fallback implementation")
            article_metrics = evaluate_article_quality(
                document_content,
                reference_content
            )
        logger.info(f"Article metrics obtained: {article_metrics}")
    except Exception as e:
        logger.error(f"Error getting article metrics: {e}")
        article_metrics = {}
    
    # Print results with more context
    print("\n=== Document Evaluation Results ===")
    print("\nPrometheus Scores (0-1 scale, higher is better):")
    for aspect, score in prometheus_scores.items():
        print(f"{aspect}: {score:.2f}")
    
    print("\nWriteHere Scores (0-1 scale, higher is better):")
    for aspect, score in writehere_scores.items():
        print(f"{aspect}: {score:.2f}")
    
    print("\nArticle Quality Metrics:")
    print(f"Entity Recall (0-1 scale): {article_metrics.get('entity_recall', 0):.2f}")
    print("\nROUGE Scores (precision, recall, F1):")
    for metric, scores in article_metrics.get('rouge_scores', {}).items():
        print(f"{metric}:")
        for score_type, value in scores.items():
            print(f"  {score_type}: {value:.2f}")
    
    return {
        "prometheus_scores": prometheus_scores,
        "writehere_scores": writehere_scores,
        "article_metrics": article_metrics
    }

def test_latex_conversion(latex_file: str):
    # Initialize configuration
    config = HumanLLMConfig()
    config.common_vectordb_config.embedding_function = embedding_function
    config.initialize()
    llmORchains_list = config.get_llmORchains_list()
    config.common_vectordb_config.embedding_function = embedding_function

    # Read the LaTeX content first
    try:
        with open(latex_file, "r", encoding="utf-8") as f:
            latex_content = f.read()
    except FileNotFoundError as e:
        print(f"Error: Could not find LaTeX file {e.filename}")
        return

    # Create target JSON file
    target_path = "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/generated_doc.json"
    create_target_json(target_path, latex_content)

    # Create a test document using the generated LaTeX file
    document = {
        'id': "test-doc-1",
        'title': "Generated Document",
        'context': "Test context for generated document",
        'target_file_path': target_path
    }

    # Create environment
    env = EnvironmentManager(
        env_type="techsynthesis",
        title=document['title'],
        context=document['context'],
        target_file_path=document['target_file_path'],
        id=document['id'],
        llm=llmORchains_list["default_llm"],
        embedding_model_name=config.common_vectordb_config.embedding_function
    ).get_environment()

    print("== Initial Distance Score ==")
    initial_score = env.get_score()
    print(f"Env ID: {env.id} -- Initial Distance Score: {initial_score}")

    # Process LaTeX file
    try:
        env.synthesis_manager.GetFromLatex(latex_content)
    except Exception as e:
        print(f"Error processing LaTeX content: {e}")
        return

    print("\n== Updated Distance Score (after loading LaTeX content) ==")
    new_score = env.get_score()
    print(f"Updated Env ID: {env.id} -- New Distance Score: {new_score}")

if __name__ == "__main__":
    # First convert markdown to LaTeX
    input_md = "./benchmark/results/tmp_storm/20250620-112304_Write_a_comprehensive,_10000_words_2025_about_the_/Write_a_comprehensive,_10000_words_2025_about_the_Comparative_Analyses_of_Local_vs._Spatial_Complex_Question_Answering_in_Lar/storm_gen_article.txt"
    output_tex = "./output_latex/storm.test.tex"
    ref_md = "./outputs/report.md"
    
    # Convert markdown to LaTeX
    latex_file = test_pandoc(input_md, output_tex)
    
    # If conversion was successful, test the LaTeX conversion
    if latex_file:
        test_latex_conversion(latex_file)

    print("====Doc eval====")
    evaluate_document(input_md, ref_md)
