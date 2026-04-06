import os
import subprocess
import time
import sys
import glob
import shutil
import numpy as np

# --- CONFIGURAZIONE ---
SOURCE_ROOT = "/mnt/d/papersEvolution/DesigniteJava/projects"
OUTPUT_BASE = "/mnt/d/papersEvolution/DesigniteJava/projects_output"
LOG_FILE = os.path.join(OUTPUT_BASE, "detailed_pipeline.log")
LINE_BIN = "./line"
EMB_SIZE = "64"
MAX_FILES_PER_CHUNK = 500

GIGANTI = ["thunderbird-android", "dbeaver", "sonarqube", "geode", "asterixdb",
           "cassandra", "hibernate-orm", "solr", "lucene", "intellij-plugins","freeplane",
           "deeplearning4j", "camel", "graal", "cloudstack-archive", "hive",
           "cloudstack", "hadoop-common", "hbase", "drools", "hadoop", "azure-sdk-for-java",
           "wildfly", "payara", "h-store", "sakai","causeway", "metasfresh"]

PROJECTS = [
     "h-store", "causeway", "thredds", "deeplearning4k", "sakai", "freeplane", "metasfresh", "azure-sdk-for-java"
]

for sub in ["cpgs", "dots", "metadata", "edgelists", "embeddings"]:
    os.makedirs(os.path.join(OUTPUT_BASE, sub), exist_ok=True)

def safe_log(message):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        f.flush()

def get_tasks(p_name, p_path):
    if p_name not in GIGANTI:
        return [(p_name, p_path)]
    tasks = []
    curr_dirs = []; count = 0; idx = 1
    for r, d, files in os.walk(p_path):
        if any(x in r.lower() for x in ["test", "target", "out", "/."]): continue
        j_files = [f for f in files if f.endswith(".java")]
        if not j_files: continue
        if count + len(j_files) > MAX_FILES_PER_CHUNK and curr_dirs:
            common = os.path.commonpath(curr_dirs)
            tasks.append((f"{p_name}_chunk{idx}", common))
            idx += 1; curr_dirs = []; count = 0
        curr_dirs.append(r); count += len(j_files)
    if curr_dirs:
        tasks.append((f"{p_name}_chunk{idx}", os.path.commonpath(curr_dirs)))
    return tasks

print(f"=== PIPELINE AVVIATA (SHUTDOWN PROTECTED) ===", flush=True)

for i, p_name in enumerate(PROJECTS, 1):
    check_f = os.path.join(OUTPUT_BASE, "embeddings", f"{p_name}.vec")
    check_c = os.path.join(OUTPUT_BASE, "embeddings", f"{p_name}_chunk1.vec")
    if os.path.exists(check_f) or os.path.exists(check_c):
        print(f"[{i}/{len(PROJECTS)}] >>> SKIP: {p_name} <<<", flush=True)
        continue

    subprocess.run(["pkill", "-9", "-f", "java"], capture_output=True)
    if os.path.exists("workspace"):
        shutil.rmtree("workspace", ignore_errors=True)

    p_path = os.path.join(SOURCE_ROOT, p_name, p_name)
    if not os.path.exists(p_path): p_path = os.path.join(SOURCE_ROOT, p_name)

    tasks = get_tasks(p_name, p_path)

    for sub_id, sub_path in tasks:
        emb_final = os.path.join(OUTPUT_BASE, "embeddings", f"{sub_id}.vec")
        if os.path.exists(emb_final) and os.path.getsize(emb_final) > 0:
            continue

        print(f"\n[{i}/{len(PROJECTS)}] >>> ANALISI: {sub_id} <<<", flush=True)
        
        cpg_out = os.path.join(OUTPUT_BASE, "cpgs", f"{sub_id}.bin")
        dot_out = os.path.join(OUTPUT_BASE, "dots", sub_id)
        met_out = os.path.join(OUTPUT_BASE, "metadata", f"{sub_id}_meta.csv")
        edg_out = os.path.join(OUTPUT_BASE, "edgelists", f"{sub_id}.edgelist")
        emb_raw = os.path.join(OUTPUT_BASE, "embeddings", f"{sub_id}_raw.vec")

        try:
            my_env = os.environ.copy()
            # FLAG ANTI-AGONIA: Se Java impazzisce, si chiude da solo
            my_env["JAVA_OPTS"] = "-Xmx5G -XX:+UseG1GC -XX:+ExitOnOutOfMemoryError -XX:GCTimeLimit=70 -XX:GCHeapFreeLimit=10 -Djoern.java.no_full_resolver=true"

            if not (os.path.exists(edg_out) and os.path.exists(met_out)):
                # --- STEP 1: PARSE CON TIMEOUT ---
                try:
                    print(f"  [1/4] Joern Parse...", end="", flush=True)
                    subprocess.run(["joern-parse", sub_path, "--output", cpg_out], 
                                   env=my_env, check=True, capture_output=True, timeout=3600)
                    print(" OK.", flush=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    print(f"\n  [!] CRASH/TIMEOUT in Parse su {sub_id}. Salto chunk.")
                    subprocess.run(["pkill", "-9", "-f", "java"], capture_output=True)
                    safe_log(f"FAILED PARSE: {sub_id}")
                    continue

                # --- STEP 2: EXPORT ---
                if os.path.exists(dot_out): shutil.rmtree(dot_out)
                print(f"  [2/4] Export CFG...", end="", flush=True)
                subprocess.run(["joern-export", cpg_out, "--repr", "cfg", "--out", dot_out], 
                               env=my_env, check=True, capture_output=True)
                dot_files = glob.glob(os.path.join(dot_out, "**/*.dot"), recursive=True)
                print(f" OK ({len(dot_files)} file)", flush=True)

                # --- STEP 3: METADATA ---
                print(f"  [3/4] Metadati Rich...", end="", flush=True)
                scala = f'import io.shiftleft.semanticcpg.language._\nimportCpg("{cpg_out}")\nval writer = new java.io.PrintWriter("{met_out}")\ncpg.method.filter(_.lineNumber.isDefined).foreach {{ m => val nodeIds = m.ast.id.l.mkString(",")\nval fullClassName = m.typeDecl.fullName.headOption.getOrElse("NoClass")\nwriter.println(s"${{m.id}}|${{m.name}}|${{fullClassName}}|${{m.lineNumber.get}}|${{m.filename}}|${{nodeIds}}") }}\nwriter.close()'
                tmp_sc = f"tmp_{sub_id}.sc"
                with open(tmp_sc, "w") as f_sc: f_sc.write(scala)
                subprocess.run(["joern", "--script", tmp_sc], env=my_env, check=True, capture_output=True)
                if os.path.exists(tmp_sc): os.remove(tmp_sc)
                print(" OK.", flush=True)

                # --- STEP 4: EDGELIST ---
                print(f"  [4/4] Conversione Archi...", end="", flush=True)
                edge_count = 0
                with open(edg_out, "w") as out_f:
                    for d_file in dot_files:
                        try:
                            with open(d_file, 'r') as f:
                                for line in f:
                                    if "->" in line:
                                        parts = line.split("->")
                                        s = parts[0].strip().strip('"')
                                        d = parts[1].split("[")[0].strip().strip('"')
                                        if s.isdigit() and d.isdigit():
                                            out_f.write(f"{s} {d} 1\n"); edge_count += 1
                        except: continue
                print(f" OK ({edge_count} archi)", flush=True)

            else:
                print(f"  [>>>] Dati intermedi trovati. Salto a LiNE...", flush=True)
                edge_count = sum(1 for _ in open(edg_out))

            # --- STEP 5: LINE ---
            if edge_count > 0:
                print(f"  [5] LiNE & Pooling...", end="", flush=True)
                subprocess.run([LINE_BIN, "-train", edg_out, "-output", emb_raw, "-size", EMB_SIZE, "-order", "2", "-samples", "100", "-threads", "20"], check=True, capture_output=True)
                
                embeddings = {}
                if os.path.exists(emb_raw):
                    with open(emb_raw, "r") as f_emb:
                        next(f_emb)
                        for line in f_emb:
                            p = line.strip().split()
                            if len(p) > 1: embeddings[p[0]] = np.array([float(x) for x in p[1:]], dtype=float)
                    
                    with open(emb_final, "w") as out_v:
                        with open(met_out, "r") as f_meta:
                            for m_line in f_meta:
                                parts = m_line.strip().split("|")
                                if len(parts) < 6: continue
                                m_id, node_ids = parts[0], parts[5].split(",")
                                vecs = [embeddings[nid] for nid in node_ids if nid in embeddings]
                                if vecs: out_v.write(f"{m_id} " + " ".join(map(str, np.mean(vecs, axis=0))) + "\n")
                
                print(" OK.", flush=True)
                safe_log(f"SUCCESS: {sub_id}")
                if os.path.exists(emb_raw): os.remove(emb_raw)

        except Exception as e:
            subprocess.run(["pkill", "-9", "-f", "java"], capture_output=True)
            print(f"\n  [X] ERRORE GENERICO su {sub_id}: {str(e)[:100]}", flush=True)
            safe_log(f"FAILED: {sub_id} - {str(e)[:50]}")

print("\n=== PIPELINE TERMINATA ===")
