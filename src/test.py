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
MAX_FILES_PER_CHUNK = 10000
MAX_FILE_SIZE_KB = 150  # Soglia per ignorare i "file mostro" (150KB è molto sicuro)

GIGANTI = ["thunderbird-android", "dbeaver", "sonarqube", "geode", "asterixdb",
           "cassandra", "hibernate-orm", "solr", "lucene", "intellij-plugins",
           "deeplearning4j", "camel", "graal", "cloudstack-archive", "hive",
           "cloudstack", "hadoop-common", "hbase", "drools", "hadoop",
           "wildfly", "payara", "h-store","gwt"]

PROJECTS = [
    "JOML", "xp", "fred", "xipki", "basex", "jhotdraw", "cuba",
    "H2-Research", "antlr4", "batfish", "ontop", "birt",
    "ceylon-compiler", "cyberduck", "pinpoint", "lawnchair", "alluxio",
    "choco-solver", "closure-compiler", "buck", "owlapi", "flow",
    "stratosphere", "processing", "gridgain", "groovy", "triplea", "qpid-jms-amqp-0-x", "directory-server",
    "yacy_search_server", "stratio-cassandra", "h2database", "adempiere",
    "felix", "gwt", "sakai", "qpid-broker-j", "bazel", "netty", "metasfresh",
    "causeway", "openejb", "OsmAnd", "cxf", "WordPress-Android",
    "jackrabbit-oak", "phoenix", "mule", "elassandra", "Maxine-VM",
    "qpid", "orientdb", "nuxeo", "tuscany-sca-1.x", "tomcat",
    "ballerina-lang", "midpoint", "opennms", "ovirt-engine", "ignite",
    "thunderbird-android", "dbeaver", "sonarqube", "geode", "asterixdb",
    "cassandra", "hibernate-orm", "solr", "lucene", "intellij-plugins",
    "deeplearning4j", "camel", "graal", "cloudstack-archive", "hive",
    "cloudstack", "hadoop-common", "hbase", "drools", "hadoop",
    "wildfly", "payara", "h-store", "freeplane", "thredds"
]

# --- FUNZIONI DI SUPPORTO ---

def filter_monster_files(p_path, threshold_kb):
    """Rinomina i file troppo grandi in .bak per nasconderli a Joern."""
    ignored = []
    for r, d, files in os.walk(p_path):
        for f in files:
            if f.endswith(".java"):
                f_full = os.path.join(r, f)
                size_kb = os.path.getsize(f_full) / 1024
                if size_kb > threshold_kb:
                    os.rename(f_full, f_full + ".bak")
                    ignored.append(f)
    return ignored

def restore_monster_files(p_path):
    """Ripristina i file .bak in .java."""
    for r, d, files in os.walk(p_path):
        for f in files:
            if f.endswith(".java.bak"):
                f_full = os.path.join(r, f)
                os.rename(f_full, f_full[:-4])

def get_tasks(p_name, p_path):
    if p_name not in GIGANTI: return [(p_name, p_path)]
    tasks = []
    curr_dirs = []; count = 0; idx = 1
    for r, d, files in os.walk(p_path):
        if any(x in r.lower() for x in ["test", "target", "out", "/."]): continue
        j_files = [f for f in files if f.endswith(".java")] # Joern vede solo .java
        if not j_files: continue
        if count + len(j_files) > MAX_FILES_PER_CHUNK and curr_dirs:
            tasks.append((f"{p_name}_chunk{idx}", os.path.commonpath(curr_dirs)))
            idx += 1; curr_dirs = []; count = 0
        curr_dirs.append(r); count += len(j_files)
    if curr_dirs: tasks.append((f"{p_name}_chunk{idx}", os.path.commonpath(curr_dirs)))
    return tasks

def safe_log(message):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

# --- CORE PIPELINE ---

for sub in ["cpgs", "dots", "metadata", "edgelists", "embeddings"]:
    os.makedirs(os.path.join(OUTPUT_BASE, sub), exist_ok=True)

print(f"=== PIPELINE AVVIATA (ANTI-MONSTER MODE) ===", flush=True)

for i, p_name in enumerate(PROJECTS, 1):
    # FAST SKIP
    check_f = os.path.join(OUTPUT_BASE, "embeddings", f"{p_name}.vec")
    check_c = os.path.join(OUTPUT_BASE, "embeddings", f"{p_name}_chunk1.vec")
    if os.path.exists(check_f) or os.path.exists(check_c):
        print(f"[{i}/{len(PROJECTS)}] >>> SKIP: {p_name} <<<", flush=True)
        continue

    # PREPARAZIONE PROGETTO
    subprocess.run(["pkill", "-9", "-f", "java"], capture_output=True)
    if os.path.exists("workspace"): shutil.rmtree("workspace", ignore_errors=True)

    p_path = os.path.join(SOURCE_ROOT, p_name, p_name)
    if not os.path.exists(p_path): p_path = os.path.join(SOURCE_ROOT, p_name)

    # --- FILTRO FILE MOSTRO ---
    print(f"  [*] Controllo file > {MAX_FILE_SIZE_KB}KB...", end="", flush=True)
    ignored = filter_monster_files(p_path, MAX_FILE_SIZE_KB)
    print(f" ignorati {len(ignored)} file.", flush=True)

    tasks = get_tasks(p_name, p_path)

    for sub_id, sub_path in tasks:
        emb_final = os.path.join(OUTPUT_BASE, "embeddings", f"{sub_id}.vec")
        if os.path.exists(emb_final): continue

        print(f"\n[{i}/{len(PROJECTS)}] >>> ANALISI: {sub_id} <<<", flush=True)
        
        cpg_out = os.path.join(OUTPUT_BASE, "cpgs", f"{sub_id}.bin")
        dot_out = os.path.join(OUTPUT_BASE, "dots", sub_id)
        met_out = os.path.join(OUTPUT_BASE, "metadata", f"{sub_id}_meta.csv")
        edg_out = os.path.join(OUTPUT_BASE, "edgelists", f"{sub_id}.edgelist")
        emb_raw = os.path.join(OUTPUT_BASE, "embeddings", f"{sub_id}_raw.vec")

        try:
            my_env = os.environ.copy()
            # RAM bilanciata e No Full Resolver per stabilità
            my_env["JAVA_OPTS"] = "-Xmx5G"

            # 1. PARSE
            print(f"  [1/4] Joern Parse...", end="", flush=True)
            subprocess.run(["joern-parse", sub_path, "--output", cpg_out], env=my_env, check=True, capture_output=True)
            print(" OK.", flush=True)

            # 2. EXPORT
            if os.path.exists(dot_out): shutil.rmtree(dot_out)
            print(f"  [2/4] Export CFG...", end="", flush=True)
            subprocess.run(["joern-export", cpg_out, "--repr", "cfg", "--out", dot_out], env=my_env, check=True, capture_output=True)
            dot_files = glob.glob(os.path.join(dot_out, "**/*.dot"), recursive=True)
            print(f" OK ({len(dot_files)} file)", flush=True)

            # 3. METADATA
            print(f"  [3/4] Metadata...", end="", flush=True)
            scala = f'import io.shiftleft.semanticcpg.language._\nimportCpg("{cpg_out}")\nval writer = new java.io.PrintWriter("{met_out}")\ncpg.method.filter(_.lineNumber.isDefined).foreach {{ m => val nodeIds = m.ast.id.l.mkString(",")\nval fullClassName = m.typeDecl.fullName.headOption.getOrElse("NoClass")\nwriter.println(s"${{m.id}}|${{m.name}}|${{fullClassName}}|${{m.lineNumber.get}}|${{m.filename}}|${{nodeIds}}") }}\nwriter.close()'
            tmp_sc = f"tmp_{sub_id}.sc"
            with open(tmp_sc, "w") as f_sc: f_sc.write(scala)
            subprocess.run(["joern", "--script", tmp_sc], env=my_env, check=True, capture_output=True)
            if os.path.exists(tmp_sc): os.remove(tmp_sc)
            print(" OK.", flush=True)

            # 4. EDGELIST (TURBO)
            print(f"  [4/4] Edgelist...", end="", flush=True)
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

            # 5. LINE & POOLING
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

        except Exception as e:
            subprocess.run(["pkill", "-9", "-f", "java"], capture_output=True)
            print(f"\n  [X] ERRORE: {sub_id} -> {str(e)[:100]}", flush=True)
            safe_log(f"FAILED: {sub_id} - {str(e)[:100]}")

    # RIPRISTINO FILE DEL PROGETTO
    restore_monster_files(p_path)

print("\n=== PIPELINE TERMINATA ===")