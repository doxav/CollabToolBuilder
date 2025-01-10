# 📄 Document Embedding Analysis
*Harnessing embeddings for direct content comparison analysis.*

## 🎤 Introduction

Given an outline of an article (all the headings and subheadings), how well can a Large Language Model (LLM) generate a similar unknown content? That's the purpose of this dataset. 

Taking an Arxiv paper, a patent, or a Wikipedia article like [Wikipedia page for self-driving cars](https://en.wikipedia.org/wiki/Self-driving_car) as an example, we see the following headings: **Definitions, Automated driver assistance system, Autonomous vs. automated, Autonomous versus cooperative** and associated content for each heading, etc. We then try to generate a similar plan and content without prior knowledge of this content (only title and abstract), and compare this content with the ground truth.

We built this dataset to be able to compare semantic structure and content for each section, as well as its length. Then, for any sections that were longer than a max embedding token limit, split section up and give them new unique titles. 

Rverything extracted (headings and content) is then analysed - e.g., calculating the Rouge-L, MAUVE and cosine similarity scores. Some scores would only accept a max number of tokens (we had to split them above).

## 💻 How to Run the Code

#### 1. Download code + create env
```
$ git clone https://github.com/codeananda/document_extraction.git
$ cd document_extraction
$ pip install -r requirements.txt
```

#### 2. Set your OpenAI key

1. Go to https://platform.openai.com/account/api-keys to create a new key (if you don't have one already).
2. Rename `.env_template` to `.env`
3. Add your key in next to `OPENAI_API_KEY`

## 🤔 Questions

### How to modify `max_section_length` to > 512, and/or set a new embedding

You must change MAX_EMBEDDING_TOKEN_LENGTH (check if this max token is compatible with your huggingface embedding, MAUVE, and OpenAI).

* Set a compatible huggingface embeddings code

HUGGINGFACE_EMBEDDING_MODEL_NAME = "nomic-embed-text-v1" # "e5-base-v2"
HUGGINGFACE_EMBEDDING_PATH = "nomic-ai/nomic-embed-text-v1" # "intfloat/e5-base-v2"
HUGGINGFACE_EMBEDDING_PREFIX = "" # e.g. e5-base-v2 might require "query: " to better match QA pairs


### How do I manually edit text extracted from pdfminer?

* Run `load_arxiv_paper(path_to_paper)`
* Write output to disk using `json.dump`
* Modify the `Content` key.
* Load content from disk using `json.load`
* See `extract_plan_and_content` for functions to pass to next 