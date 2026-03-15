import subprocess
import os
import sys

# Lista completa dei progetti
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

def run_labeller():
    # Setup cartella di output
    labelled_dir = os.path.join("matches", "filtered", "labelled")
    
    if not os.path.exists(labelled_dir):
        os.makedirs(labelled_dir)
        print("Cartella creata: " + labelled_dir)

    for project in projects:
        # Percorso del file finale
        output_file = os.path.join(labelled_dir, "labelled_" + project + "_filtered.csv")

        # Salta se gia' fatto
        if os.path.exists(output_file):
            print(">>> Progetto " + project + " gia' etichettato. Salto.")
            continue

        print("--- Labelling progetto: " + project + " ---")
        
        # Percorsi input basati sul tuo comando template
        designite_input = "analyses\\after\\" + project + "_after\\complex_methods_" + project + "_after.csv"
        matches_input = "matches\\filtered\\matches\\matches_" + project + "_filtered.csv"

        # Costruzione comando
        command = [
            sys.executable, "labeller.py",
            "--designite", designite_input,
            "--refminer", matches_input,
            "--out", output_file
        ]

        try:
            # Esecuzione
            subprocess.check_call(command)
            print("OK: Labelling completato per " + project)
        except subprocess.CalledProcessError:
            print("ERRORE: labeller.py ha fallito per " + project)
        except Exception as e:
            print("ERRORE IMPREVISTO su " + project + ": " + str(e))
        
        print("-" * 40)

if __name__ == "__main__":
    run_labeller()