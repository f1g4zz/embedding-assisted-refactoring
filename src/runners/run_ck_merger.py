import subprocess
import os
import sys

# Resolve DesigniteJava root and src root dynamically
project_thesis_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

SRC_PATH = os.path.join(project_thesis_src_path, "core")

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

def run_all_mergers():
    script_path = os.path.join(SRC_PATH, "ck_merger.py")
    
    for project in projects:
        main_input = os.path.join(BASE_PATH, "matches", "filtered", "labelled", "labelled_" + project + "_filtered.csv")
        ck_methods = os.path.join(BASE_PATH, "ck_analysis", project, "method.csv")
        ck_classes = os.path.join(BASE_PATH, "ck_analysis", project, "class.csv")
        out_matches = os.path.join(BASE_PATH, "ck_merged", project + "_merged.csv")
        out_track = os.path.join(BASE_PATH, "ck_merged", project + "_tracking_log.csv")

        if not os.path.exists(main_input):
            print(">>> Skip " + project + ": Missing labelled_filtered file.", flush=True)
            continue
        if not os.path.exists(ck_methods) or not os.path.exists(ck_classes):
            print(">>> Skip " + project + ": Missing results in ck_analysis.", flush=True)
            continue
            
        if os.path.exists(out_matches):
            print(">>> Skip " + project + ": Already merged previously.", flush=True)
            continue

        print("--- Merging CK Data: " + project + " ---", flush=True)
        
        command = [
            sys.executable, script_path,
            "--main", main_input,
            "--methods", ck_methods,
            "--classes", ck_classes,
            "--out_matches", out_matches,
            "--out_track", out_track
        ]

        try:
            subprocess.check_call(command)
            print("OK: Merge completed for " + project, flush=True)
        except subprocess.CalledProcessError:
            print("ERROR: ck_merger.py failed for " + project, flush=True)
        except Exception as e:
            print("ERROR on " + project + ": " + str(e), flush=True)
        
        print("-" * 40, flush=True)

if __name__ == "__main__":
    run_all_mergers()
