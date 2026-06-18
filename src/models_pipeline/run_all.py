"""
Batch execution script to run the machine learning pipeline (pipeline.py)
sequentially across all ten target refactoring types.
"""
import subprocess
import sys

MAIN_SCRIPT_NAME = "pipeline.py"

targets = [
    "Change Variable Type", "Change Parameter Type", "Change Return Type",
    "Extract Method", "Move Method", "Rename Method", "Rename Variable",
    "Rename Parameter", "Extract Variable", "Add Parameter"
]

TUNING = True

print("Starting batch execution for all refactoring methods...")
print(f"Total targets number: {len(targets)}\n")

for i, target in enumerate(targets, 1):
    print(f"{'='*50}")
    print(f"[{i}/{len(targets)}] EXECUTING TARGET: {target}")
    print(f"{'='*50}")
    
    command = [sys.executable, MAIN_SCRIPT_NAME, "--target", target]
    
    if TUNING:
        command.append("--tune")
        
    try:
        subprocess.run(command, check=True)
        print(f"\n[SUCCESS] {target} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Execution failed for '{target}' error code: {e.returncode}.\n")


print("Script successfully terminated!")