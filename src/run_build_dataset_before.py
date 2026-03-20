import subprocess
import os
import sys


BASE_PATH = r"C:\Users\lanza\Downloads\papersEvolution\DesigniteJava\project_thesis\src"


ANALYSES_PATH = r"C:\Users\lanza\Downloads\papersEvolution\DesigniteJava\analyses"

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

def run_build_before():
    script_path = os.path.join(BASE_PATH, "build_dataset.py")

    for project in projects:
        project_dir = os.path.join(ANALYSES_PATH, project)
        
        if not os.path.exists(project_dir):
            print(">>> Folder not found for " + project + ". Skipping.")
            print("    Path checked: " + project_dir)
            continue

        output_file = os.path.join(project_dir, "complex_methods_" + project + ".csv")

        if os.path.exists(output_file):
            print(">>> Project " + project + " already done. Skipping.")
            continue

        print("--- Building Dataset (Before): " + project + " ---")
        
        command = [
            sys.executable, script_path,
            "--project", project_dir,
            "--output", output_file
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
    run_build_before()