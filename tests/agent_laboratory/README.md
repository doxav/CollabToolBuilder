1. Install dependency packages
```bash
pip install -r requirements.txt
```
2. Set environment variables
```bash
export user_id="test_id"
export OPENAPI_KEY="XXXX"
```

3. Clone the GitHub Repository: Begin by cloning the repository using the command:
```bash
git clone git@github.com:SamuelSchmidgall/AgentLaboratory.git
```

4. Rename utils.py to agent_utils.py
```bash
cd tests/agent_laboratory/AgentLaboratory
mv utils.py agent_utils.py

5. Apply patch file
```bash
git apply ../inference_query_model_changes.patch
```
```
6. Run ai_lab_repo.py at root level
```python
python tests/agent_laboratory/AgentLaboratory/ai_lab_repo.py --llm-backend "humanllm" --research-topic "AI research in healthcare" --copilot-mode "true"
```