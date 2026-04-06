import subprocess
import os
import sys

SRC_PATH = r"D:\papersEvolution\DesigniteJava\project_thesis\src"

ANALYSES_PATH = r"D:\papersEvolution\DesigniteJava\analyses"

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

def run_build_dataset_after():
    script_path = os.path.join(SRC_PATH, "build_dataset.py")
    base_after_dir = os.path.join(ANALYSES_PATH, "after")
    
    if not os.path.exists(base_after_dir):
        os.makedirs(base_after_dir)
        print("Created directory: " + base_after_dir)

    for project in projects:
     
        project_after_dir = os.path.join(base_after_dir, project)
        if not os.path.exists(project_after_dir):
            os.makedirs(project_after_dir)

        output_file = os.path.join(project_after_dir, project + "_after.csv")

        if os.path.exists(output_file):
            print(">>> Project " + project + " (after) already exists. Skipping.")
            continue

        print("--- Building Dataset (After): " + project + " ---")
        
  
        project_source_path = os.path.join(base_after_dir, project )

        if not os.path.exists(project_source_path):
            print(">>> ERROR: Source folder not found for " + project)
            print("    Path checked: " + project_source_path)
            continue

        command = [
            sys.executable, script_path,
            "--project", project_source_path,
            "--output", output_file,
            "--no-embeddings"
        ]

        try:
            subprocess.check_call(command)
            print("OK: Dataset built for " + project)
        except subprocess.CalledProcessError:
            print("ERROR: build_dataset.py failed for " + project)
        except Exception as e:
            print("ERROR on " + project + ": " + str(e))
        
        print("-" * 40)

if __name__ == "__main__":
    run_build_dataset_after()