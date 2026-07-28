import os
import re

template_path = "run_gen_3b_base_time_exp.slurm"

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
            
        # 2. Replace time_exp with the new weighting
        content = content.replace("time_exp", new_suffix)
        
        # 3. Replace the actual argparse parameter "--time-weighting exp" -> "--time-weighting linear"
        # The line is: "--time-weighting exp"
        # Since we just replaced time_exp with time_linear, the argument would have become
        # "--time-weighting linear" which is incorrect if it was something else, but actually
        # the argument in the template is "--time-weighting exp" which becomes "--time-weighting linear" if weight is linear.
        # Wait, the template has "--time-weighting exp", which doesn't contain "time_exp" exactly as one word.
        # Ah, the template has: "--time-weighting exp". Let's be careful.
        # We should explicitly replace "--time-weighting exp" with f"--time-weighting {weight}"
        
        # Let's fix this properly:
        # We will take the fresh template for each iteration
        content = template
        
        if model == "evolved":
            content = content.replace("gen_3b_base", "gen_3b_evolved")
            content = content.replace("matu-3b_base", "matu-3b_evol")
            
        # Replace the directory/file name suffixes
        content = content.replace("time_exp", f"time_{weight}")
        
        # Replace the arg
        content = content.replace("--time-weighting exp", f"--time-weighting {weight}")
        
        file_name = f"run_gen_3b_{model}_time_{weight}.slurm"
        with open(file_name, "w") as f:
            f.write(content)
            
        print(f"Generated {file_name}")

print("All SLURM files generated successfully!")
