import os
import shutil
import sys

# Resolve BASE_PATH dynamically (3 parents up from project_thesis/src/utils/archiver_after.py to get DesigniteJava/)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

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

def archive_files():
    """
    Moves complex_methods CSV reports for the 'after' project versions to a single archived directory.
    """
    archive_dir = os.path.join(BASE_PATH, "analyses", "after", "archived")

    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        print("Created archive directory: " + archive_dir)

    for project in projects:

        project_folder = project + "_after"
        filename = "complex_methods_" + project + "_after.csv"
        
        source_path = os.path.join(BASE_PATH, "analyses", "after", project_folder, filename)
        destination_path = os.path.join(archive_dir, filename)

        if os.path.exists(source_path):
            try:
                shutil.move(source_path, destination_path)
                print("MOVED: " + project + " -> " + filename)
            except Exception as e:
                print("ERROR on " + project + ": " + str(e))
        else:
            if os.path.exists(destination_path):
                print("--- " + project + ": File already exists in archived directory.")
            else:
                print("--- " + project + ": File NOT found in " + source_path)

if __name__ == "__main__":
    archive_files()
