import os
import re

template_path = "slurms/math/configs/run_gen_3b_base_time_exp.slurm"

with open(template_path, "r") as f:
    template = f.read()

models = ["base", "evolved"]
weightings = ["linear", "cutoff", "log", "sigmoid"]

for model in models:
    for weight in weightings:
        # e.g. gen_3b_base_time_linear
        new_suffix = f"time_{weight}"
        
        content = template
        
        # 1. Replace the model part if needed
        if model == "evolved":
            content = content.replace("gen_3b_base", "gen_3b_evolved")
            content = content.replace("matu-3b_base", "matu-3b_evol")
            
        # Replace the directory/file name suffixes
        content = content.replace("time_exp", f"time_{weight}")
        
        # Replace the arg
        content = content.replace("--time-weighting exp", f"--time-weighting {weight}")
        
        file_name = f"slurms/math/configs/run_gen_3b_{model}_time_{weight}.slurm"
        with open(file_name, "w") as f:
            f.write(content)
            
        print(f"Generated {file_name}")

print("All SLURM files generated successfully!")
