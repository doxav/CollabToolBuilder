from setuptools import setup

setup(
    name="humanllm",
    version="0.1",
    # 1) single-file module at project root:
    py_modules=["humanllm"],

    # 2) two package aliases that both point to ./utils:
    packages=[
        "humanllm_utils",
        "hllm_utils",
    ],
    package_dir={
        "humanllm_utils": "utils",
        "hllm_utils": "utils",
    },

    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
)

