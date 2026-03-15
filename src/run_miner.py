import subprocess
import os
import sys

# Lista completa dei progetti
projects = [
    "ceylon-compiler", "payara", "shardingsphere", "freeplane", "triplea", 
    "thredds", "thunderbird-android", "H2-Research", "qpid-jms-amqp-0-x", 
    "dbeaver", "cyberduck", "directory-server", "xp", "yacy_search_server", 
    "stratio-cassandra", "h2database", "pinpoint", "lawnchair", "adempiere", 
    "felix", "h-store", "gwt", "alluxio", "qpid-broker-j", 
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

def run_miner():
    # Definiamo le cartelle di output
    matches_dir = os.path.join("matches", "filtered", "matches")
    movements_dir = os.path.join("matches", "filtered", "movements")
    
    # Creazione cartelle (compatibile Python 2.x)
    for folder in [matches_dir, movements_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print("Creata cartella: " + folder)

    for project in projects:
        # Costruiamo i nomi dei file di output per il controllo
        out_matches = os.path.join(matches_dir, "matches_" + project + "_filtered.csv")
        out_track = os.path.join(movements_dir, "movements_" + project + "_filtered.csv")

        # Se il file esiste gia', saltiamo il progetto (permette di riprendere da dove si e' interrotto)
        if os.path.exists(out_matches):
            print(">>> Progetto " + project + " gia' elaborato. Salto.")
            continue

        print("--- Inizio Mining: " + project + " ---")
        
        # Percorsi input
        designite_input = "filtered\\" + project + "_filtered.csv"
        refminer_input = "mined_projects\\" + project + ".json"

        # Comando: sys.executable assicura l'uso del Python del venv attivo
        command = [
            sys.executable, "miner.py",
            "--designite", designite_input,
            "--refminer", refminer_input,
            "--out_matches", out_matches,
            "--out_track", out_track
        ]

        try:
            # Eseguiamo il comando e mostriamo l'output in tempo reale
            subprocess.check_call(command)
            print("OK: " + project + " completato correttamente.")
        except subprocess.CalledProcessError as e:
            print("ERRORE: miner.py ha fallito per il progetto " + project)
        except Exception as e:
            print("ERRORE IMPREVISTO su " + project + ": " + str(e))
        
        print("-" * 40)

if __name__ == "__main__":
    # Verifica preliminare di pandas nello script principale
    try:
        import pandas
        print("Ambiente OK: Pandas trovato (v" + str(pandas.__version__) + ")")
        run_miner()
    except ImportError:
        print("ERRORE CRITICO: Pandas non trovato nell'ambiente attivo.")
        print("Prova a eseguire: " + sys.executable + " -m pip install pandas")