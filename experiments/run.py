# experiments/urn.py

import os
import sys
import subprocess


SCRIPT_LIST = [
    "run_training.py",
    "run_evaluation.py",
    "run_prediction.py",    
]

def run(script_list):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for i, script_file in enumerate(script_list, 1):
        script_path = os.path.join(base_dir, script_file)

        print(f"\n>> [RUN {i}/{len(script_list)}] {script_file}")
        print("-" * 80)

        if not os.path.exists(script_path):
            print(f"[Error] Script not found: {script_path}")
            break

        try:
            subprocess.run([sys.executable, script_path], check=True)
            print("\n" + "-" * 80)
            print(f">> Completed: {script_file}")

        except subprocess.CalledProcessError as e:
            print(f"[Error] Script '{script_file}' failed (exit code={e.returncode})")
            break

        except Exception as e:
            print(f"[Error] Unexpected error while running '{script_file}': {e}")
            break


if __name__ == "__main__":

    run(SCRIPT_LIST)