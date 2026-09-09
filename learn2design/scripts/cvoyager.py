"""Test script ConstrainedVoyagerProblem."""

from dfbench.problems import ConstrainedVoyagerProblem
from dfbench import Objective

from learn2design.example_algorithms import ComponentPolicyAdam 

SEED = 42

# Create ConstrainedVoyager instance
problem = ConstrainedVoyagerProblem()
obj = Objective(
    problem,
    verbose=1,
    max_time=60 * 5,  # 5 minutes
    print_every=1,
    save_params_history=True,
    save_to_file_every=100,
    display_mode="live",  # Use "log" for a non-interactive terminal
)

optimizer = ComponentPolicyAdam()


# Run optimization
optimizer.optimize(
    obj,
    # learning_rate=0.1,
    random_seed=SEED,  # This is only for random param generation in AdamGD
)

# Save run data
obj.save_run_data(hyper_param_str="lr0.1")  # Optional hyperparameter string will be included in the filename
# Output loss plot, sensitivity plot, final parameters(JSON), and loss history(JSON) which can be toggled by arguments
obj.output_to_files(hyper_param_str="lr0.1")

print("Best loss:")
print(f"    {obj.best_loss:.6f}")
print("Total evaluations:")
print(f"    {obj.eval_count}")
print("First parameters:")
print(f"    {obj.params_history_bounded[0]}")
print("Best parameters:")
print(f"    {obj.best_params_bounded}")
print("Seed:")
print(f"    {SEED}")
