import subprocess
import os
import sys

BASE_PATH = r"C:\Users\lanza\Downloads\papersEvolution\DesigniteJava"
SRC_PATH = os.path.join(BASE_PATH, "project_thesis", "src")

projects = [
    "ceylon-compiler", "payara", "shardingsphere", "freeplane", "triplea", 
    "thredds", "thunderbird-android", "H2-Research", "qpid-jms-amqp-0-x", 
    "dbeaver", "cyberduck", "directory-server", "xp", "yacy_search_server", 
    "stratio-cassandra", "h2database", "pinpoint", "lawnchair", "adempiere",
    "JOML", "felix", "h-store", "gwt", "alluxio", "qpid-broker-j", 
    "sonarqube", "fred", "choco-solver", "closure-compiler", "bazel", 
    "geode", "asterixdb", "wildfly", "buck", "owlapi", "cassandra", 
    "knime-core", "netty", "flow", "metasfresh", "hibernate-orm", 
    "causeway", "openejb", "OsmAnd", "cxf", "WordPress-Android", 
    "solr", "lucene", "batfish", "intellij-plugins", "jackrabbit-oak", 
    "azure-sdk-for-java", "phoenix", "mule", "elassandra", "ontop", 
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
            print(">>> Skip " + project + ": Manca il file labelled_filtered.", flush=True)
            continue
        if not os.path.exists(ck_methods) or not os.path.exists(ck_classes):
            print(">>> Skip " + project + ": Mancano i risultati in ck_analysis.", flush=True)
            continue
            
        if os.path.exists(out_matches):
            print(">>> Skip " + project + ": Gia' unito in precedenza.", flush=True)
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
            print("OK: Merge completato per " + project, flush=True)
        except subprocess.CalledProcessError:
            print("ERRORE: ck_merger.py ha fallito per " + project, flush=True)
        except Exception as e:
            print("ERRORE su " + project + ": " + str(e), flush=True)
        
        print("-" * 40, flush=True)

if __name__ == "__main__":
    run_all_mergers()