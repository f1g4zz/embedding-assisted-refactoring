import os
import subprocess
import sys
from pathlib import Path

# --- PATH CONFIGURATION ---
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
MAIN_DIR = BASE_DIR / "ck_merged"
METHODS_DIR = BASE_DIR / "projects_output" / "embeddings"
METADATA_DIR = BASE_DIR / "projects_output" / "metadata"
OUTPUT_DIR = BASE_DIR / "embedded"

# Resolve enrich_embeddings.py relative to the runner script location
ENRICH_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "enrich_embeddings.py"))

def run_automation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting scan in: {MAIN_DIR}")
    
    main_files = list(MAIN_DIR.glob("*_merged.csv"))
    
    if not main_files:
        print("[!] No _merged.csv file found. Check the path.")
        return

    print(f"[*] Projects found: {len(main_files)}")

    for main_path in main_files:
        project_name = main_path.name.replace("_merged.csv", "")
        
        methods_path = METHODS_DIR / f"{project_name}.vec"
        metadata_path = METADATA_DIR / f"{project_name}_meta.csv"
        output_path = OUTPUT_DIR / f"{project_name}_embedded.csv"

        print(f"\n>>> PROCESSING PROJECT: {project_name.upper()}")

        if not methods_path.exists():
            print(f"[SKIP] Missing embeddings file: {methods_path}")
            continue
        if not metadata_path.exists():
            print(f"[SKIP] Missing metadata file: {metadata_path}")
            continue

        cmd = [
            sys.executable, ENRICH_SCRIPT,
            "--main", str(main_path),
            "--methods", str(methods_path),
            "--metadata", str(metadata_path),
            "--output", str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"[OK] Project {project_name} successfully completed.")
            else:
                print(f"[ERROR] The script returned an error for {project_name}.")
        
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to start script for {project_name}: {e}")

    print("\n" + "="*40)
    print("BATCH PROCESSING COMPLETED")
    print("="*40)

if __name__ == "__main__":
    run_automation()
