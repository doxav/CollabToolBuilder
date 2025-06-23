from primitives.swe_primitives.find_buggy_code import find_buggy_code
from datasets import load_dataset
from env.SWEBench.env import SWEBenchEnvironment, SWEManager, SWEProblem

dataset = load_dataset(path="ahsanirfan961/swe-bech-lite-bm25-13k-take3", split='train')
dataset = dataset.select(range(1, 2))
for data in dataset:
    problem = SWEProblem.parse_obj(data)
    env = SWEBenchEnvironment(problem)
    bot = SWEManager(target_dir="env/SWEBench/repos/scikit-learn")

    find_buggy_code(bot) #, env, problem)

    break