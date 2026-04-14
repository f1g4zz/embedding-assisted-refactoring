import os
import subprocess
import sys

BASE_DIR = "/home/a.lanza-thesis/progetti_tesi"
JAVA_BIN = "/home/a.lanza-thesis/.conda/envs/designite_env/bin/java"
JAR_PATH = "/home/a.lanza-thesis/bin/DesigniteJava/DesigniteJava.jar"
SOURCE_ROOT = os.path.join(BASE_DIR, "dataset_full")
OUTPUT_BASE = os.path.join(BASE_DIR, "designite_results")
LOG_FILE = os.path.join(BASE_DIR, "pipeline_designite.log")

os.makedirs(OUTPUT_BASE, exist_ok=True)

if not os.path.exists(JAVA_BIN):
    print(f"ERROR: Java bin not found {JAVA_BIN}")
    sys.exit(1)

projects = sorted([d for d in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, d))])

print(f"--- START: {len(projects)} projects found ---")

with open(LOG_FILE, "a") as log:
    for p_name in projects:
        p_src = os.path.join(SOURCE_ROOT, p_name)
        p_out = os.path.join(OUTPUT_BASE, p_name)

        if os.path.exists(os.path.join(p_out, "TypeMetrics.csv")):
            continue

        os.makedirs(p_out, exist_ok=True)
        log.write(f"\n[ANALYSIS] {p_name}\n")
        log.flush()

        cmd = [JAVA_BIN, "-Xmx90G", "-jar", JAR_PATH, "-i", p_src, "-o", p_out]

        try:
            subprocess.run(cmd, stdout=log, stderr=log, text=True, check=True)
            print(f"Completed: {p_name}")
        except Exception as e:
            print(f"Error{p_name}: {e}")
            log.write(f"ERROR  {p_name}: {str(e)}\n")
            log.flush()

print("--- FINE ANALISI ---")