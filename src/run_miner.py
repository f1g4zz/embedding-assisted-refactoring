import subprocess
import os
import sys

# Project List
projects = [
    "ceylon-compiler", "payara", "shardingsphere", "freeplane", "triplea", 
    "thredds", "thunderbird-android", "H2-Research", "qpid-jms-amqp-0-x", 
    "dbeaver", "cyberduck", "directory-server", "xp", "yacy_search_server", 
    "stratio-cassandra", "h2database", "pinpoint", "lawnchair", "adempiere", 
    "felix", "h-store", "gwt", "alluxio", "qpid-broker-j", 
    "sonarqube", "fred", "choco-solver", "closure-compiler", "bazel", 
    "geode", "asterixdb", "wildfly", "buck", "owlapi", "cassandra", 
    "knime-core", "netty", "flow", "metasfresh", "hibernate-orm", 
    "causeway", "OsmAnd", "cxf", "WordPress-Android", 
    "solr", "lucene", "batfish", "intellij-plugins", "jackrabbit-oak", 
     "phoenix", "mule", "elassandra", "ontop", 
    "Maxine-VM", "qpid", "deeplearning4j", "camel", "orientdb", 
    "nuxeo", "tuscany-sca-1.x", "basex", "tomcat", "xipki", 
    "stratosphere", "ballerina-lang", "antlr4", "midpoint", "graal", 
    "birt", "opennms", "processing", "cloudstack-archive", "gridgain", 
    "groovy", "hive", "ovirt-engine", "ignite", "cloudstack", 
    "hadoop-common", "sakai", "hbase", "drools", "hadoop", 
    "jhotdraw", "cuba"
]

def run_miner():
    matches_dir = os.path.join("matches", "filtered", "matches")
    movements_dir = os.path.join("matches", "filtered", "movements")
    
    for folder in [matches_dir, movements_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print("Creata cartella: " + folder)

    for project in projects:
        out_matches = os.path.join(matches_dir, "matches_" + project + "_filtered.csv")
        out_track = os.path.join(movements_dir, "movements_" + project + "_filtered.csv")

        # Skip if file already exists
        if os.path.exists(out_matches):
            print(">>> Project " + project + " already exists. Skipping.")
            continue

        print("--- Mining: " + project + " ---")
        
        designite_input = "filtered\\" + project + "_filtered.csv"
        #designite_input = "analyses\\" + project + "\\" + project + ".csv"
        refminer_input = "mined_projects\\" + project + ".json"

        command = [
            sys.executable, "miner.py",
            "--designite", designite_input,
            "--refminer", refminer_input,
            "--out_matches", out_matches,
            "--out_track", out_track
        ]

        try:
            subprocess.check_call(command)
            print("OK: " + project + " succesfully completed.")
        except subprocess.CalledProcessError as e:
            print("ERRORE: miner.py failed on " + project)
        except Exception as e:
            print("ERROR " + project + ": " + str(e))
        
        print("-" * 40)

if __name__ == "__main__":
    try:
        import pandas
        print("Ambient OK: Pandas found (v" + str(pandas.__version__) + ")")
        run_miner()
    except ImportError:
        print("ERROR: Pandas not found.")
        print("tryin to execute install pip: " + sys.executable + " -m pip install pandas")