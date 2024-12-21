from datasets import load_dataset
from env.SWEBench.env import *
from primitives.swe_primititves.generate_patch import *

dataset = load_dataset(path="ahsanirfan961/swe-bech-lite-bm25-13k-take3", split='train')
dataset = dataset.select(range(1, 2))

for data in dataset:
    problem = SWEProblem.parse_obj(data)
    
    env = SWEBenchEnvironment(problem)

    # print(get_abs_current_dir())
    # print(find_files.invoke({"file_name": "multiclass"}))
    # print(ls.invoke({}))
    # print(edit_file.invoke({"path": "CODE_OF_CONDUCT.md", "line_number": 8, "num_lines": 1, "new_content": "from datasets import load_dataset"}))

    bot = SWEManager()

    print(generate_patch(bot, problem, env))


    