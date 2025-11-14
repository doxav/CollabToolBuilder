# CollabToolBuilder & HumanLLM
## Progressive Guide to Human-AI Collaborative Systems

## Table of Contents

1. [Why ?](#why)
2. [System Architecture & Theory](#system-architecture--theory)
3. [Quick Start Guide](#quick-start-guide)
4. [Progressive Tutorial: From Basic to Advanced](#progressive-tutorial-from-basic-to-advanced)
5. [CollabToolBuilder: Advanced Function Development](#collabtoolbuilder-advanced-function-development)
11. [Tools How‑To (Code Examples)](docs/README_tools.md)
6. [Scientific Applications & Use Cases](#scientific-applications--use-cases)
7. [Interactive Interface Documentation](#interactive-interface-documentation)
8. [Performance Analysis & Optimization](#performance-analysis--optimization)
9. [Technical Reference](#technical-reference)
10. [Contributing & Research](#contributing--research)

---

## Why ?

### The Human-AI Collaboration

Most AI systems operate in isolation, producing outputs without incorporating human expertise during the inference process. This approach misses critical opportunities for:

- **Domain expertise integration**: Leveraging human knowledge that isn't captured in training data
- **Real-time error correction**: Identifying and fixing issues as they emerge
- **Contextual adaptation**: Adjusting behavior based on specific use cases and requirements
- **Continuous learning**: Improving system performance through iterative feedback

### Research Background

HumanLLM focus on human-in-the-loop (HITL) designs and implements principles from several research domains:

1. **Human-Computer Interaction (HCI)**: Interactive machine learning systems that adapt to user feedback
2. **Active Learning**: Systems that strategically query humans for the most valuable information
3. **Reinforcement Learning from Human Feedback (RLHF)**: Incorporating human preferences into model optimization
4. **Meta-Learning**: Systems that learn how to learn more effectively through experience

![Human-LLM Collaborative Architecture](images/HumanLLM_schematic.png)
*The core HumanLLM architecture showing the seamless integration between human guidance and AI processing. Humans can intervene before and after each inference to provide context, refine prompts, critique outputs, and guide the learning process. This bidirectional collaboration ensures optimal AI performance while maintaining human oversight and control.*

### Key Innovation: Bidirectional Collaboration

Unlike traditional AI systems where humans only provide input and receive output, HumanLLM enables **bidirectional collaboration**:

- **Pre-inference intervention**: Humans can modify prompts, adjust parameters, and provide context
- **Post-inference intervention**: Humans can critique, refine, and improve outputs
- **Learning integration**: The system learns from human feedback to improve future performance

---

## CollabToolBuilder: A Major Application of HumanLLM

### The Tool Factory Concept

CollabToolBuilder is an application of HumanLLM that uses **four coordinated HumanLLM agents** to create new functions and capabilities.

![Interactive Learning Loop](images/readme_learningloop.gif)
*The CollabToolBuilder learning loop: A 4-agent system that iteratively proposes, codes, validates, and refines new tools. This system uses HumanLLM agents to: (1) propose tasks, (2) generate code, (3) critique implementations, and (4) create documentation. Each agent incorporates human feedback to improve the tool development process.*

### Four-Agent Architecture

1. **Task Proposal Agent**: Analyzes goals and proposes specific implementations
2. **Code Generation Agent**: Creates functional implementations based on proposals  
3. **Critique Agent**: Reviews and suggests improvements to generated code
4. **Documentation Agent**: Creates comprehensive documentation and usage examples

### Relationship to HumanLLM Core

- **CollabToolBuilder is NOT part of HumanLLM core architecture**
- **CollabToolBuilder USES HumanLLM** to create intelligent development agents
- **CollabToolBuilder CAN CREATE TOOLS** that are then made available to HumanLLM agents
- **This represents the "tool factory" pattern** for expanding AI capabilities

For a practical, code-first guide to building tools with CollabToolBuilder and using them via HumanLLM (legacy functions and newer tools API), see: [docs/README_tools.md](docs/README_tools.md)

```bash
# Launch CollabToolBuilder in UI mode (separate from HumanLLM core)
python learn.py
open frontend/index.html
```

---

## System Architecture & Theory

### Core Components

1. **HumanLLM Engine**: Core inference system with human intervention capabilities
2. **Dynamic Configuration Manager**: Adaptive system that modifies behavior based on context and rules
3. **Vector Database Integration**: Optional RAG capabilities via ChromaDB/ElasticSearch
4. **Usage Tracking**: Basic metrics collection for optimization

![Human-LLM Collaborative Architecture](images/HumanLLM_schematic.png)
*The core HumanLLM architecture showing the seamless integration between human guidance and AI processing. The system provides intervention points before and after each inference, with optional vector database integration for document retrieval.*

### Implementation Foundation

#### 1. Dynamic Configuration System

The system implements **rule-based configuration**, where model behavior changes based on:
- **Pattern matching**: Regex patterns in user messages trigger specific configurations
- **Frequency triggers**: Every N interactions can modify behavior
- **Usage tracking**: Basic quota and usage monitoring
- **LLM switching**: Premium vs default model selection based on rules

#### 2. Human Intervention Points

Human feedback is integrated through:
- **Skip rounds**: Configurable automation intervals (`skip_rounds` parameter)
- **Pre-inference modification**: Users can modify prompts and parameters
- **Post-inference review**: Users can critique and improve outputs
- **Dynamic parameter adjustment**: Real-time configuration changes

#### 3. Optional basic RAG Integration

HumanLLM provides optional Retrieval-Augmented Generation capabilities:

```python
# Add external documents to the knowledge base
assistant.add_rag_document( file_path="documentation.pdf", agent_name="research_assistant")

# Use RAG-enhanced prompts with {rag_context} placeholder
rag_prompt = assistant.load_prompt_with_rag( prompt_name="research_template", template_data={"topic": "machine learning"})

# The {rag_context} placeholder is automatically replaced with relevant document content

# Retrieve RAG documents programmatically
rag_docs, metadata = assistant.get_rag_documents( agent_name="research_assistant", query="specific topic")
```

**RAG Features:**
- **Document Loading**: PDF, JSON, HTML, Markdown support
- **Automatic Indexing**: Vector storage with metadata filtering  
- **Prompt Integration**: `{rag_context}` placeholder replacement
- **Selective Retrieval**: Query-based document filtering

### Storage Architecture

**Vector Database Support**:
- **ChromaDB or ElasticSearch**: Configurable vector storage for RAG document embeddings
- **Local File Storage**: Persistence directories for vector databases
- **Basic Logging**: Interaction logs and usage tracking in local files

**Configuration Options** (in `config.py`):
```python
vector_store_type = "elasticsearch"  # or "chroma" 
chroma_persist_dir = 'human_llm_vector_chromadb'
kibana_url_port = 'http://127.0.0.1:5601'  # Optional visualization
```

---

## Quick Start Guide

### Prerequisites

- **Python 3.10+** (Required for modern async/await patterns)
- **Virtual Environment** (Recommended: venv or conda)
- **Code Editor** (VSCode recommended for CLI mode)
- **OpenAI API Key** (For LLM access)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd CollabToolBuilder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

```bash
# Copy configuration template
cp config.py.template config.py

# Edit config.py with your settings
# - OpenAI API keys
# - ElasticSearch/ChromaDB settings
# - Interface preferences
```

### First Example - Minimal Setup

```python
from humanllm import HumanLLM

# Simplest possible configuration
minimal_assistant = HumanLLM(
    system_prompt="You are a helpful assistant.",
    llmORchains_list={},  # Empty for testing
    skip_rounds=10        # Automate for 10 interactions
)

# The system is now ready for use!
# response = minimal_assistant.invoke("Hello, world!")
```

---

## Progressive Tutorial: From Basic to Advanced

### Level 1: Basic Human-LLM Interaction

**Scientific Context**: Start with simple human-AI collaboration patterns to understand the fundamental interaction mechanisms.

```python
from humanllm import HumanLLM
from langchain_openai import ChatOpenAI

# Basic assistant with human oversight
basic_assistant = HumanLLM(
    system_prompt="You are a helpful assistant specialized in explaining concepts clearly.",
    llmORchains_list={ "main": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7) },
    skip_rounds=3  # Human intervention every 3 interactions
)

# Usage: The system will automatically request human feedback every 3 queries
```

**Learning Outcomes**:
- Understanding human intervention timing
- Basic system prompt configuration
- Simple LLM integration patterns

### Level 2: Dual-Model Architecture

**Scientific Context**: Implement resource optimization through model selection - use efficient models for simple tasks, powerful models for complex ones.

```python
# Dual-model setup: efficiency + capability
dual_model_system = HumanLLM(
    system_prompt="You are an intelligent assistant that adapts to task complexity.",
    llmORchains_list={
        "default_llm": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5),  # Fast & efficient
        "premium_llm": ChatOpenAI(model="gpt-4", temperature=0.3)           # Powerful & precise
    },
    default_llmORchain="default_llm",     # Start with efficient model
    premium_llmORchain="premium_llm",     # Escalate to powerful model when needed
    premium_llm_by_default=False,         # Optimize for efficiency first
    skip_rounds=5
)
```

**Learning Outcomes**:
- Resource optimization strategies
- Model selection principles
- Cost-performance trade-offs

### Level 3: Context-Aware Dynamic Configuration

**Scientific Context**: Implement adaptive behavior based on query analysis - different tasks require different optimization strategies.

```python
# Context-aware configuration
context_aware_config = {
    "research": {
        "temperature": 0.2,              # Low temperature for factual accuracy
        "use_premium_llm": True,         # High-quality model for research
        "quota": 15                      # Limit expensive research queries
    },
    "creative": {
        "temperature": 0.9,              # High temperature for creativity
        "use_premium_llm": False,        # Default model sufficient for creativity
        "num_parallel_inferences": 3,   # Multiple attempts for better creativity
        "quota": 25
    },
    "analysis": {
        "temperature": 0.3,              # Moderate temperature for analysis
        "use_premium_llm": True,         # Premium model for complex analysis
        "num_parallel_inferences": 2    # Parallel processing for reliability
    }
}

adaptive_assistant = HumanLLM(
    system_prompt="You are an adaptive research assistant that optimizes performance based on task type.",
    llmORchains_list={
        "default_llm": ChatOpenAI(model="gpt-3.5-turbo"),
        "premium_llm": ChatOpenAI(model="gpt-4")
    },
    default_llmORchain="default_llm",
    premium_llmORchain="premium_llm",
    skip_rounds=7,
    dynamic_llm_config=context_aware_config
)
```

**Learning Outcomes**:
- Adaptive system design
- Context classification strategies
- Performance optimization techniques

### Level 4: Rule-Based Intelligent Switching

**Scientific Context**: Implement sophisticated decision-making through pattern recognition and rule evaluation.

```python
# Advanced rule-based configuration
intelligent_rules = {
    "scientific_mode": {
        "rules": {
            "regex": {
                "patterns": [
                    r"research|study|analysis|experiment|hypothesis|data|statistical?",
                    r"peer.review|methodology|results|conclusion|literature"
                ],
                "target": "user_message"
            }
        },
        "modifications": {
            "temperature": 0.1,              # Maximum precision for scientific work
            "use_premium_llm": True,         # Best model for scientific accuracy
            "system_prompt_append": [
                "\nAdhere to scientific methodology. Cite sources when possible. "
                "Distinguish between established facts and hypotheses."
            ]
        }
    },
    "creative_mode": {
        "rules": {
            "regex": {
                "patterns": [r"creative|story|poem|imagine|invent|brainstorm|artistic"],
                "target": "user_message"
            }
        },
        "modifications": {
            "temperature": 0.9,              # Maximum creativity
            "use_premium_llm": False,        # Efficiency for creative tasks
            "num_parallel_inferences": 4,   # Multiple creative attempts
            "system_prompt_prepend": [
                "Think creatively and outside the box. "
            ]
        }
    },
    "debugging_mode": {
        "rules": {
            "regex": {
                "patterns": [r"debug|error|bug|exception|fix|troubleshoot"],
                "target": "user_message"
            }
        },
        "modifications": {
            "temperature": 0.05,             # Ultra-precise for debugging
            "use_premium_llm": True,         # Best model for complex debugging
            "invoke_kwargs": {
                "max_tokens": 2000           # Longer responses for detailed debugging
            },
            "system_prompt_append": [
                "\nApproach debugging systematically. Analyze the error, "
                "identify potential causes, and provide step-by-step solutions."
            ]
        }
    }
}

intelligent_assistant = HumanLLM(
    system_prompt="You are an intelligent assistant that adapts its approach based on the type of task presented.",
    llmORchains_list={
        "default_llm": ChatOpenAI(model="gpt-3.5-turbo"),
        "premium_llm": ChatOpenAI(model="gpt-4")
    },
    default_llmORchain="default_llm",
    premium_llmORchain="premium_llm",
    skip_rounds=8,
    dynamic_llm_config=intelligent_rules
)
```

**Learning Outcomes**:
- Advanced pattern recognition
- Rule-based decision systems
- Context-sensitive optimization

### Level 5: Self-Optimizing Systems with Trace Learning

**Scientific Context**: Implement machine learning optimization that automatically improves system performance through experience.

```python
# Self-optimizing system with trace learning
learning_system_config = {
    "adaptive_learning": {
        "rules": {
            "frequency": {
                "every_n": 10               # Trigger optimization every 10 interactions
            }
        },
        "modifications": [
            {
                "type": "trace",            # Use trace optimization
                "optimizer": "performance_optimizer",
                "config": {
                    "optimizer_kind": "optoprimev2",
                    "optimizer_kwargs": {
                        "learning_rate": 0.01,
                        "max_iterations": 100,
                        "convergence_threshold": 0.001
                    }
                },
                "targets": ["temperature", "system_prompt", "num_parallel_inferences"]
            },
            {
                "type": "simple",           # Additional fixed optimizations
                "payload": {
                    "use_premium_llm": True,
                    "invoke_kwargs": {
                        "max_tokens": 1500
                    }
                }
            }
        ]
    },
    "quality_enhancement": {
        "rules": {
            "regex": {
                "patterns": [r"improve|optimize|enhance|better|quality"],
                "target": "user_message"
            }
        },
        "modifications": {
            "temperature": 0.2,
            "use_premium_llm": True,
            "num_parallel_inferences": 3
        }
    }
}

self_optimizing_assistant = HumanLLM(
    system_prompt="You are a self-improving assistant that learns from interactions to optimize performance.",
    llmORchains_list={
        "default_llm": ChatOpenAI(model="gpt-3.5-turbo"),
        "premium_llm": ChatOpenAI(model="gpt-4")
    },
    default_llmORchain="default_llm",
    premium_llmORchain="premium_llm",
    skip_rounds=6,
    dynamic_llm_config=learning_system_config
)
```

**Learning Outcomes**:
- Machine learning integration
- Performance optimization theory
- Self-adaptive systems design

---

## CollabToolBuilder: Detailed Implementation Guide

### Architecture Deep Dive

As introduced earlier, CollabToolBuilder is a sophisticated application built on top of HumanLLM that demonstrates the full potential of human-AI collaborative development. This section provides detailed implementation guidance.

### Scientific Motivation

CollabToolBuilder implements the following concepts:

1. **Multi-Agent Systems**: Four specialized HumanLLM agents working in coordination
2. **Iterative Development Theory**: Continuous improvement through feedback loops
3. **Collaborative Intelligence**: Combining human creativity with AI execution
4. **Meta-Programming**: AI systems that create and improve other AI systems

### Four-Agent Implementation Details

Each agent in the CollabToolBuilder system is a specialized HumanLLM instance:

```python
# Example: Task Proposal Agent configuration
task_agent = HumanLLM(
    system_prompt="You are a task analysis expert. Break down complex goals into implementable tasks.",
    llmORchains_list={"default_llm": ChatOpenAI(model="gpt-4", temperature=0.3)},
    skip_rounds=2,  # Frequent human oversight for task quality
    dynamic_llm_config={
        "complex_analysis": {"temperature": 0.2, "use_premium_llm": True},
        "creative_ideation": {"temperature": 0.7, "use_premium_llm": False}
    }
)
```

### Complete Four-Agent System Implementation

```python
from humanllm import HumanLLM
from langchain.llms import ChatOpenAI

# Configure the complete four-agent CollabToolBuilder system
agents = {
    "task_agent": HumanLLM(
        system_prompt="Break down complex tasks into implementable components.",
        llmORchains_list={"default_llm": ChatOpenAI(model="gpt-4", temperature=0.3)},
        skip_rounds=3
    ),
    "code_agent": HumanLLM(
        system_prompt="Generate clean, well-documented code implementations.",
        llmORchains_list={"default_llm": ChatOpenAI(model="gpt-4", temperature=0.1)},
        premium_llmORchain=ChatOpenAI(model="gpt-4-turbo", temperature=0.1),
        dynamic_llm_config={
            "complex_implementation": {"use_premium_llm": True},
            "simple_functions": {"use_premium_llm": False}
        }
    ),
    "test_agent": HumanLLM(
        system_prompt="Create comprehensive test suites with edge case coverage.",
        llmORchains_list={"default_llm": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)},
        skip_rounds=2
    ),
    "review_agent": HumanLLM(
        system_prompt="Perform thorough code review focusing on best practices.",
        llmORchains_list={"default_llm": ChatOpenAI(model="gpt-4", temperature=0.0)},
        premium_llmORchain=ChatOpenAI(model="gpt-4-turbo", temperature=0.0),
        premium_llm_by_default=True
    )
}

# CollabToolBuilder's learning loop implementation
def collaborative_development_cycle(goal):
    """The core learning loop of CollabToolBuilder"""
    # Phase 1: Task decomposition
    tasks = agents["task_agent"].prompt(f"Break down this goal: {goal}")
    
    # Phase 2: Implementation with human oversight
    for task in tasks:
        code = agents["code_agent"].prompt(f"Implement: {task}")
        tests = agents["test_agent"].prompt(f"Test this code: {code}")
        review = agents["review_agent"].prompt(f"Review: {code}\nTests: {tests}")
        
        # Human decision point - key to the learning loop
        human_feedback = input("Accept implementation? (y/n/modify): ")
        if human_feedback == "modify":
            improvements = input("What improvements do you suggest? ")
            # Feed back into the system for learning
        
    return final_implementation
```

### Launch Instructions

```bash
# Start the collaborative development environment
python learn.py

# Access the web interface
open frontend/index.html
```

### Development Methodology

![HumanLLM Interactive Interface](images/IHM.png)
*The HumanLLM web interface showing all available controls and options. Users can modify system prompts, add instructions, configure LLM parameters, view previous results, and control the inference process through an intuitive graphical interface.*

#### Phase 1: Task Conceptualization
1. **Problem Definition**: Clearly articulate the function requirements
2. **Scope Analysis**: Determine complexity and resource requirements
3. **Success Criteria**: Define measurable outcomes

#### Phase 2: Collaborative Implementation
1. **AI-Generated Proposals**: System generates initial implementations
2. **Human Review**: Expert evaluation and modification
3. **Iterative Refinement**: Continuous improvement through feedback

#### Phase 3: Validation & Optimization
1. **Automated Testing**: Comprehensive test suite execution
2. **Performance Analysis**: Benchmarking and optimization
3. **Documentation Generation**: Automatic documentation creation

### Development Principles

1. **Human-AI Symbiosis**: Leverage strengths of both human and AI capabilities
2. **Incremental Improvement**: Small, testable changes rather than large modifications
3. **Quality Assurance**: Multi-stage validation and testing
4. **Knowledge Preservation**: Capture and reuse successful patterns

---

## Scientific Applications & Use Cases

### Research & Academia

```python
# Research assistant for scientific literature analysis
research_assistant = HumanLLM(
    system_prompt="""You are a scientific research assistant specializing in literature analysis.
    Follow rigorous scientific methodology: cite sources, distinguish facts from hypotheses,
    maintain objectivity, and acknowledge limitations.""",
    llmORchains_list={
        "default_llm": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2),
        "premium_llm": ChatOpenAI(model="gpt-4", temperature=0.1)
    },
    dynamic_llm_config={
        "literature_review": {
            "temperature": 0.1,
            "use_premium_llm": True,
            "quota": 20
        },
        "hypothesis_generation": {
            "temperature": 0.6,
            "use_premium_llm": True,
            "num_parallel_inferences": 3
        }
    },
    skip_rounds=3
)
```

### Software Development

```python
# Code review and development assistant
dev_assistant = HumanLLM(
    system_prompt="""You are a senior software engineer specializing in code review and development.
    Focus on code quality, security, performance, and maintainability.""",
    llmORchains_list={
        "code_reviewer": ChatOpenAI(model="gpt-4", temperature=0.2),
        "creative_coder": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    },
    dynamic_llm_config={
        "security_review": {"temperature": 0.05, "use_premium_llm": True},
        "creative_coding": {"temperature": 0.8, "use_premium_llm": False},
        "debugging": {"temperature": 0.1, "use_premium_llm": True}
    },
    skip_rounds=5
)
```

### Educational Applications

```python
# Adaptive tutoring system
tutor_system = HumanLLM(
    system_prompt="""You are an adaptive tutor that adjusts teaching style based on student needs.
    Use pedagogical best practices: scaffolding, active learning, and personalized feedback.""",
    dynamic_llm_config={
        "beginner_level": {"temperature": 0.3, "system_prompt_append": ["Use simple language and examples."]},
        "advanced_level": {"temperature": 0.5, "use_premium_llm": True},
        "assessment": {"temperature": 0.1, "use_premium_llm": True}
    },
    skip_rounds=2  # Frequent human oversight for educational quality
)
```

---

## Interactive Interface Documentation

### Before Inference Controls

The interface provides comprehensive control over the AI system before each inference:

#### Core Configuration
1. **System Prompt Modification (A)**: Customize the AI's role and behavior
2. **Context Addition (B)**: Provide additional information and constraints
3. **Parameter Adjustment (C)**: Fine-tune temperature, model selection, and inference count

#### Historical Analysis
4. **Previous Results Review (E)**: Analyze past performance and patterns
5. **Filtered Results View (F)**: Focus on modified, scored, or commented results
6. **Comment Logging (D)**: Track feedback and improvement suggestions

#### Automation Controls
7. **Skip Round Management (G)**: Configure autonomous operation periods
8. **Model Selection (H, I)**: Switch between default and premium LLMs
9. **Parallel Processing (J)**: Adjust concurrent inference configuration

### After Inference Controls

Post-inference intervention capabilities:

1. **Output Modification (A)**: Direct editing of AI responses
2. **Critical Analysis (B)**: Structured critique and improvement suggestions
3. **Prompt Optimization (C)**: Identify more effective prompting strategies
4. **Quality Assessment (D)**: Scoring and detailed evaluation
5. **Process Control (E)**: Return to pre-inference state for adjustments

### Real-Time Features

- **Interactive Interface**: Web-based control panel for system management
- **Dynamic Configuration**: Real-time parameter and prompt modifications
- **Usage Monitoring**: Basic tracking of interactions and resource usage
- **Optional RAG Integration**: Document retrieval when configured

---

## Performance Analysis & Optimization

### Available Metrics and Monitoring

The system provides basic performance tracking:

#### Usage Tracking
- **Interaction Count**: Number of calls per configuration type
- **Token Consumption**: Basic token usage tracking
- **Cost Tracking**: Simple cost accumulation per help type
- **Success/Failure Rates**: Basic outcome recording

#### Configuration Management  
- **Dynamic Rule Matching**: Pattern-based configuration switching
- **Quota Management**: Usage limits per configuration type
- **LLM Selection**: Automatic premium/default model switching
- **Parameter Adjustment**: Rule-based temperature and prompt modifications

### Optimization Features

#### Rule-Based Optimization
```python
# Example: Available optimization patterns in the code
dynamic_config = {
    "research_mode": {
        "rules": {"regex": {"patterns": [r"research|study"], "target": "user_message"}},
        "modifications": {"temperature": 0.1, "use_premium_llm": True, "quota": 10}
    }
}
```

#### Trace Learning Support
- **Trace Optimization**: Integration with external optimizers (optoprimev2)
- **Parameter Tuning**: Automated adjustment of temperature, prompts, and inference settings
- **Outcome Recording**: Basic metrics collection for optimization feedback

---

## Technical Reference

### API Documentation

#### Core Classes

```python
class HumanLLM:
    def __init__(
        self,
        system_prompt: str = None,
        llmORchains_list: Dict = None,
        default_llmORchain: str = "default_llm",
        premium_llmORchain: str = "premium_llm",
        premium_llm_by_default: bool = False,
        skip_rounds: int = 1,
        dynamic_llm_config: Dict = None,
        num_parallel_inferences: int = 1,
        temperature_min: float = 0.0,
        temperature_max: float = 1.0
    ):
        """
        Initialize HumanLLM system with comprehensive configuration options.
        
        Parameters:
        -----------
        system_prompt : str
            Base instruction set for the AI system
        llmORchains_list : Dict
            Named dictionary of available LLM instances
        dynamic_llm_config : Dict
            Context-aware configuration rules and modifications
        skip_rounds : int
            Number of autonomous iterations before human intervention
        """
```

#### Configuration Schema

```python
# Dynamic configuration structure
{
    "context_name": {
        "rules": {
            "regex": {
                "patterns": ["pattern1", "pattern2"],
                "target": "user_message"
            },
            "frequency": {"every_n": 10},
            "similarity": {"threshold": 0.8, "min_similar": 3}
        },
        "modifications": {
            "temperature": 0.5,
            "use_premium_llm": True,
            "num_parallel_inferences": 2,
            "invoke_kwargs": {"max_tokens": 1000},
            "system_prompt_append": ["Additional instructions"],
            "quota": 15
        }
    }
}
```

### Generation Techniques

HumanLLM surfaces multiple strategies for proposing candidates during an inference round. Set the behaviour via `generation_technique` (and related arguments such as temperatures or expert lists).

- `temperature_variation` – sweep between `temp_min` and `temp_max` to sample diverse candidates.
- `iterative_alternatives` – iteratively request alternative answers using previous drafts as counter-examples.
- `mixture_of_agents_generation` (aliases: `moa`, `multi_llm`) – delegate generation to different models or personas for broader coverage.
- Other project-specific techniques may exist; the value is passed straight through, so custom strategies can plug in without modification.

#### Common option: `draft_patch_mode`

`draft_patch_mode` is a wrapper, not a new `generation_technique`. When enabled (via constructor kwargs or `invoke_kwargs`), HumanLLM will:

1. Produce a deterministic baseline draft at temperature `0.0`.
2. Generate `K` full rewrites using the currently selected `generation_technique` and parameters.
3. Convert each rewrite into a unified diff against the baseline.
4. Return a list shaped like `["DR AFT\n<full text>", "--- a/answer.md...", ...]` which feeds directly into patch-based selectors.

You can toggle the behaviour through dynamic config:

```json
{
  "modifications": {
    "invoke_kwargs": {
      "draft_patch_mode": true,
      "patch_k": 4,
      "patch_validate": true
    }
  }
}
```

### Selection Techniques

Candidate aggregation is controlled by `selection_technique`:

- `concat` – append all candidates sequentially.
- `best_of_n` – ask an LLM judge to pick the best draft.
- `patch_hunk_vote` – merge unified diffs/JSON patches by hunk, using majority votes for conflicts and returning the final full text.
- `patch_best_of_n` – apply each patch independently, cluster identical finals, and choose the largest (breaking ties with minimal diff size).
- `moa` / `mixture_of_agents` – synthesis selector for multi-agent pipelines.
- `last` – fall back to the most recent candidate.

### Advanced Features

#### Trace Optimization
```python
# Trace learning configuration
{
    "optimizer_config": {
        "type": "trace",
        "optimizer": "performance_optimizer",
        "config": {
            "optimizer_kind": "optoprimev2",
            "optimizer_kwargs": {
                "learning_rate": 0.01,
                "max_iterations": 100
            }
        },
        "targets": ["temperature", "system_prompt"]
    }
}
```

#### Rule Evaluation System
- **Regex Rules**: Pattern matching in user input
- **Frequency Rules**: Time or interaction-based triggers
- **Similarity Rules**: Content similarity analysis
- **Custom Rules**: User-defined evaluation functions

### Integration Guidelines

#### Vector Database Integration
- **ChromaDB**: Local vector storage with persistence
- **ElasticSearch**: Distributed vector search with Kibana visualization
- **Configuration**: Switchable backend via `vector_store_type` setting

#### LLM Provider Support
- **Primary**: OpenAI models (GPT-3.5, GPT-4, GPT-4-turbo)
- **Framework**: LangChain integration for provider abstraction
- **Configuration**: Multiple model support via `llmORchains_list`

#### Deployment Options
- **Local**: Direct Python execution with web interface
- **Docker**: Container support with docker-compose configuration

---

## Contributing & Research

### Research Opportunities

The HumanLLM framework opens several research directions:

1. **Optimal Intervention Timing**: When should humans intervene for maximum benefit?
2. **Preference Learning**: How to learn user preferences from minimal feedback?
3. **Collaborative Intelligence**: Optimal division of labor between humans and AI
4. **Meta-Learning**: Systems that learn how to learn from human feedback

### Contributing Guidelines

#### Code Contributions
1. **Test-Driven Development**: All code must include comprehensive tests
2. **Scientific Validation**: Features should be backed by research or empirical evidence
3. **Documentation**: Include both technical docs and scientific context
4. **Performance Analysis**: Benchmark new features for efficiency and quality

#### Research Contributions
1. **Empirical Studies**: Controlled experiments with human subjects
2. **Algorithmic Innovations**: Novel approaches to human-AI collaboration
3. **Domain Applications**: Specialized implementations for specific fields
4. **Evaluation Frameworks**: New metrics and assessment methodologies

### Future Directions

#### Short-term Goals
- Enhanced rule evaluation systems
- Improved optimization algorithms
- Better user interface design
- Expanded LLM provider support

#### Long-term Vision
- **Autonomous Research Assistants**: AI systems that can conduct independent research
- **Collaborative Programming**: Human-AI teams that develop software together
- **Educational Transformation**: Personalized AI tutors for every learner
- **Scientific Discovery**: AI systems that contribute to human knowledge

---

## Conclusion

HumanLLM provides a framework for collaborative human-AI interaction, emphasizing configurable intervention, rule-based adaptation, and incremental improvement.

The framework supports a progression from basic interaction to more advanced adaptive configurations. It can be applied in research, education, and software development contexts.

These collaboration patterns aim to help create AI systems that complement human work rather than replace it.

---

*For the latest updates, examples, and research findings, visit our repository and join our community of researchers and practitioners advancing human-AI collaboration.*
