1. Install dependency packages
```bash
pip install -r requirements.txt
```

2. Set environment variables
```bash
export COMPOSIO_API_KEY="9qqydtl9chb7br7ssf8mvv"
```


3. Clone the GitHub Repository: Begin by cloning the repository using the command:
```bash
git clone git@github.com:ComposioHQ/composio.git
```

4. Apply patch file
```bash
git apply benchmark_changes.patch
```

6. Run the bash
```bash
 python tests/swe_agent/benchmark.py --test-instance-ids "django__django-14434"
 ```