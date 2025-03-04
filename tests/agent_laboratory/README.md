1. Install dependency packages
```bash
pip install -r requirements.txt
```
2. Set environment variables
```bash
export user_id="test_id"
export OPENAPI_KEY="XXXX"
```
3. Run ai_lab_repo.py at root level
```python
python tests/agent_laboratory/ai_lab_repo.py --llm-backend "humanllm" --resarch-topic "AI research in healthcare" --copilot-mode "true"
```