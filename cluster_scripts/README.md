Commands to use scripts on cluster/macstudio

```bash
# Launch on macstudio
sbatch run_multi.sh optuna_opti_Synthesis_coder macstudio

# Launch on GPU (Cluster)
sbatch --gres=gpu:1 run_multi.sh optuna_opti_Synthesis_coder gpu

# Launch on GPT4 (Cluster)
sbatch run_multi.sh optuna_opti_Synthesis_coder GPT4
```
