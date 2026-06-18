import subprocess
import os
import sys

# Resolve BASE_PATH dynamically (3 parents up from project_thesis/src/runners/run_ck_analysis.py to get DesigniteJava/)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

JAR_PATH = os.path.join(BASE_PATH, "ck", "ck", "target", "ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar")
PROJECTS_ROOT = os.path.join(BASE_PATH, "projects")
RESULTS_ROOT = os.path.join(BASE_PATH, "ck_analysis")

projects = [
    "ceylon-compiler", "payara", "shardingsphere", "freeplane", "triplea", 
    "thredds", "thunderbird-android", "H2-Research", "qpid-jms-amqp-0-x", 
    "dbeaver", "cyberduck", "directory-server", "xp", "yacy_search_server", 
    "stratio-cassandra", "h2database", "pinpoint", "lawnchair", "adempiere",
    "JOML", "felix", "h-store", "gwt", "alluxio", "qpid-broker-j", 
    "sonarqube", "fred", "choco-solver", "closure-compiler", "bazel", 
    "geode", "asterixdb", "wildfly", "buck", "owlapi", "cassandra", 
    "knime-core", "netty", "flow", "metasfresh", "hibernate-orm", 
    "causeway",  "OsmAnd", "cxf", "WordPress-Android", 
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

def run_ck_analysis():
    if not os.path.exists(JAR_PATH):
        print("ERROR: JAR not found: " + JAR_PATH)
        return

    for project in projects:

        project_path = os.path.join(PROJECTS_ROOT, project, project)
        
        project_results_dir = os.path.join(RESULTS_ROOT, project)

        check_file = os.path.join(project_results_dir, "class.csv")

        if os.path.exists(check_file):
            print(">>> Project " + project + " already done. Skipping.")
            continue

        if not os.path.exists(project_results_dir):
            os.makedirs(project_results_dir)
            print("Folder Created: " + project_results_dir)

        print("--- Running CK Analysis: " + project + " ---")

        if not os.path.exists(project_path):
            print(">>> ERROR: Source not found " + project_path)
            continue


        command = [
            "java", "-jar", JAR_PATH,
            project_path,
            "true", "0", "true",
            project_results_dir + os.sep 
        ]

        try:
            subprocess.check_call(command)
            print("OK: Analysis completed " + project)
        except subprocess.CalledProcessError:
            print("ERROR: CK has failed for " + project)
        except Exception as e:
            print("ERROR " + project + ": " + str(e))
        
        print("-" * 40)

if __name__ == "__main__":
    run_ck_analysis()
