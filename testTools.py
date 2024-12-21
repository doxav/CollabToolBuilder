from datasets import load_dataset
from env.SWEBench.env import *
from primitives.swe_primititves.generate_patch import *
from primitives.swe_primititves.find_buggy_code import *

dataset = load_dataset(path="ahsanirfan961/swe-bech-lite-bm25-13k-take3", split='train')
dataset = dataset.select(range(0, 1))

for data in dataset:
    problem = SWEProblem.parse_obj(data)
    
    env = SWEBenchEnvironment(problem)

    # print(get_abs_current_dir())
    # print(find_files.invoke({"file_name": "multiclass"}))
    # print(ls.invoke({}))
    # print(edit_lines_in_file.invoke({"file_path": "setup.py", "n": 11, "m": 11, "replacement_text": 'print("hello world")'}))

    bot = SWEManager()

    # print(find_buggy_code(bot, problem, env))
    print(generate_patch(bot, problem, env))


    # print(env.get_score(code))

    