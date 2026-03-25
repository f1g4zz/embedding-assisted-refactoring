cat << 'EOF' > ~/progetti_tesi/pipeline_tesi.py
import os, subprocess, time, glob, shutil, csv, re
import numpy as np

# --- CONFIG ---
HOME = os.path.expanduser("~")
SOURCE_ROOT = os.path.join(HOME, "progetti_tesi/dataset_full")
OUTPUT_BASE = os.path.join(HOME, "progetti_tesi/projects_output")
LINE_BIN = os.path.join(HOME, "progetti_tesi/line")
JOERN_BIN_PATH = os.path.join(HOME, "bin/joern/install/joern-cli/bin")

PROJECTS = ["Antlr4", "Ballerina-lang", "Birt", "Cloudstack-archive", "CloudStack", "CUBA", "Drools", "Graal", "Hadoop-common", "Hadoop", "Hbase", "Ignite", "JHotDraw", "Midpoint", "OpenNMS", "oVirt-engine", "Sakai", "Stratosphere", "Tomcat", "XiPKI", "Adempiere", "Alluxio", "AsterixDB", "Azure-sdk-for-java", "BaseX", "Batfish", "Bazel", "Buck", "Camel", "Cassandra", "Causeway", "Ceylon-compiler", "Choco-solver", "Closure-compiler", "Cyberduck", "CXF", "DBeaver", "Deeplearning4j", "Directory-server", "Elassandra", "Felix", "Flow", "Fred", "Freeplane", "Geode", "GridGain", "Groovy", "GWT", "H-store", "H2database", "Hibernate-orm", "Hive", "Intellij-plugins", "Jackrabbit-oak", "JOML", "Knime-core", "Lawnchair", "Lucene", "Maxine-VM", "Metasfresh", "Mule", "Netty", "Nuxeo", "Ontop", "OpenEJB", "OrientDB", "OsmAnd", "OWLAPI", "Payara", "Phoenix", "Pinpoint", "Processing", "Qpid", "Qpid-broker-j", "Qpid-jms-amqp-0-x", "Shardingsphere", "Solr", "Sonarqube", "Stratio-cassandra", "Thredds", "Thunderbird-android", "TripleA", "Tuscany-sca-1.x", "Wildfly", "WordPress-Android", "XP", "YaCy_search_server"]

for sub in ["cpgs", "csv_export", "metadata", "edgelists", "embeddings"]:
    os.makedirs(os.path.join(OUTPUT_BASE, sub), exist_ok=True)

my_env = os.environ.copy()
my_env["PATH"] = JOERN_BIN_PATH + ":" + my_env["PATH"]
my_env["JAVA_OPTS"] = "-Xmx48G"
my_env["LD_LIBRARY_PATH"] = os.path.join(HOME, ".conda/envs/joern_line_env/lib") + ":" + my_env.get("LD_LIBRARY_PATH", "")

for p_name in PROJECTS:
    base_path = os.path.join(SOURCE_ROOT, p_name)
    if not os.path.exists(base_path): continue
    final_vec = os.path.join(OUTPUT_BASE, "embeddings", f"{p_name}.vec")
    if os.path.exists(final_vec): continue

    print(f"\n>>> {p_name}")
    cpg = os.path.join(OUTPUT_BASE, "cpgs", f"{p_name}.bin")
    csv_dir = os.path.join(OUTPUT_BASE, "csv_export", p_name)
    meta_csv = os.path.join(OUTPUT_BASE, "metadata", f"{p_name}_meta.csv")
    edge_lst = os.path.join(OUTPUT_BASE, "edgelists", f"{p_name}.edgelist")
    raw_emb = os.path.join(OUTPUT_BASE, "embeddings", f"{p_name}_raw.vec")

    try:
        # 1 & 2: Joern Parse & Export
        subprocess.run([os.path.join(JOERN_BIN_PATH, "joern-parse"), base_path, "--output", cpg], env=my_env, check=True, capture_output=True)
        if os.path.exists(csv_dir): shutil.rmtree(csv_dir)
        subprocess.run([os.path.join(JOERN_BIN_PATH, "joern-export"), cpg, "--repr", "all", "--format", "neo4jcsv", "--out", csv_dir], env=my_env, check=True, capture_output=True)

        # 3. Processing con normalizzazione ID
        nodes_file = glob.glob(os.path.join(csv_dir, "nodes_*.csv"))[0]
        edges_file = glob.glob(os.path.join(csv_dir, "edges_*.csv"))[0]
        methods = {} 
        
        with open(nodes_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5 or row[1] != 'METHOD': continue
                m_id = str(row[0]).strip()
                methods[m_id] = {'n': row[4], 'c': row[5].split(':')[0], 'nodes': [m_id]}

        with open(edges_file, 'r') as f, open(edge_lst, 'w') as fe:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3: continue
                s, e, t = str(row[0]).strip(), str(row[1]).strip(), row[2]
                if t == 'CFG': fe.write(f"{s} {e} 1\n")
                if t == 'AST' and s in methods: methods[s]['nodes'].append(e)

        with open(meta_csv, 'w') as fm:
            for mid, d in methods.items():
                fm.write(f"{mid}|{d['n']}|{d['c']}|0|unk|{','.join(set(d['nodes']))}\n")
        
        # 4. LINE
        print("  Running LINE...", end="", flush=True)
        subprocess.run([LINE_BIN, "-train", edge_lst, "-output", raw_emb, "-size", "64", "-order", "2", "-samples", "100", "-threads", "1"], env=my_env, check=True, capture_output=True)
        
        # 5. POOLING (Verifica corrispondenza)
        n_map = {}
        if os.path.exists(raw_emb):
            with open(raw_emb, "r") as f:
                header = next(f).split()
                print(f" (LINE nodes: {header[0]})", end="")
                for l in f:
                    pt = l.strip().split()
                    if len(pt) > 1: n_map[str(pt[0]).strip()] = np.array([float(x) for x in pt[1:]], dtype=np.float32)
            
            saved = 0
            with open(final_vec, "w") as fo:
                with open(meta_csv, "r") as fm:
                    for ml in fm:
                        pts = ml.strip().split("|")
                        # Prendi solo i vettori dei nodi che LINE ha effettivamente trovato
                        vecs = [n_map[nid] for nid in pts[5].split(",") if nid in n_map]
                        if vecs:
                            fo.write(f"{pts[0]} " + " ".join(map(str, np.mean(vecs, axis=0))) + "\n")
                            saved += 1
            print(f" DONE. Saved: {saved}")
        else:
            print(" FAILED (No LINE output)")

    except Exception as e:
        print(f" ERROR: {str(e)}")
    
    if os.path.exists(cpg): os.remove(cpg)
    if os.path.exists(raw_emb): os.remove(raw_emb)

print("\n=== PIPELINE TERMINATA ===")
EOF