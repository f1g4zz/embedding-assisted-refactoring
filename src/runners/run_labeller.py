import subprocess
import os
import sys

# Resolve DesigniteJava root and src root dynamically
project_thesis_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

SRC_PATH = os.path.join(project_thesis_src_path, "core")

# Project List
projects = [
    "ceylon-compiler", "payara", "shardingsphere", "freeplane", "triplea", 
    "thredds", "thunderbird-android", "H2-Research", "qpid-jms-amqp-0-x", 
    "dbeaver", "cyberduck", "directory-server", "xp", "yacy_search_server", 
    "stratio-cassandra", "h2database", "pinpoint", "lawnchair", "adempiere",
    "JOML", "felix", "h-store", "gwt", "alluxio", "qpid-broker-j", 
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

def run_labeller():
    script_path = os.path.join(SRC_PATH, "labeller.py")
    labelled_dir = os.path.join(BASE_PATH, "matches", "filtered", "labelled")
    
    if not os.path.exists(labelled_dir):
        os.makedirs(labelled_dir)
        print("Folder Created: " + labelled_dir)

    for project in projects:
        
        output_file = os.path.join(labelled_dir, "labelled_" + project + "_filtered.csv")

        # Skip if file already exists
        if os.path.exists(output_file):
            print(">>> Project " + project + " already labeled. Skipping.")
            continue

        print("--- Labelling : " + project + " ---")
        designite_input = os.path.join(BASE_PATH, "filtered", project + "_filtered.csv")
        matches_input = os.path.join(BASE_PATH, "matches", "filtered", "matches", "matches_" + project + "_filtered.csv")

        command = [
            sys.executable, script_path,
            "--designite", designite_input,
            "--matches", matches_input,
            "--out", output_file
        ]

        try:
            subprocess.check_call(command)
            print("OK: Labelling completed for " + project)
        except subprocess.CalledProcessError:
            print("ERROR: labeller.py failed on " + project)
        except Exception as e:
            print("ERROR " + project + ": " + str(e))
        
        print("-" * 40)

if __name__ == "__main__":
    run_labeller()
