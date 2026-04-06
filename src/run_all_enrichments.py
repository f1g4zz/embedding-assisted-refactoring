import os
import subprocess
import sys
from pathlib import Path

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = Path(r"D:\papersEvolution\DesigniteJava")
MAIN_DIR = BASE_DIR / "ck_merged"
METHODS_DIR = BASE_DIR / "projects_output" / "embeddings"
METADATA_DIR = BASE_DIR / "projects_output" / "metadata"
OUTPUT_DIR = BASE_DIR / "embedded"

# Percorso dello script di arricchimento creato in precedenza
ENRICH_SCRIPT = "enrich_embeddings.py" 

def run_automation():
    # Creiamo la cartella di output se non esiste
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Inizio scansione in: {MAIN_DIR}")
    
    # 1. Cerchiamo tutti i file che finiscono con _merged.csv nella cartella main
    main_files = list(MAIN_DIR.glob("*_merged.csv"))
    
    if not main_files:
        print("[!] Nessun file _merged.csv trovato. Controlla il percorso.")
        return

    print(f"[*] Progetti trovati: {len(main_files)}")

    for main_path in main_files:
        # 2. Recuperiamo il nome del progetto (es. 'adempiere' da 'adempiere_merged.csv')
        project_name = main_path.name.replace("_merged.csv", "")
        
        # 3. Costruiamo i percorsi attesi per gli altri file
        methods_path = METHODS_DIR / f"{project_name}.vec"
        metadata_path = METADATA_DIR / f"{project_name}_meta.csv"
        output_path = OUTPUT_DIR / f"{project_name}_embedded.csv"

        print(f"\n>>> ELABORAZIONE PROGETTO: {project_name.upper()}")

        # 4. Verifica esistenza file necessari per evitare crash
        if not methods_path.exists():
            print(f"[SALTO] File embeddings mancante: {methods_path}")
            continue
        if not metadata_path.exists():
            print(f"[SALTO] File metadata mancante: {metadata_path}")
            continue

        # 5. Costruzione del comando da eseguire
        cmd = [
            sys.executable, ENRICH_SCRIPT,
            "--main", str(main_path),
            "--methods", str(methods_path),
            "--metadata", str(metadata_path),
            "--output", str(output_path)
        ]

        # 6. Esecuzione del comando
        try:
            # Usiamo subprocess per lanciare lo script e catturare l'output in tempo reale
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"[OK] Progetto {project_name} completato con successo.")
            else:
                print(f"[ERRORE] Lo script ha restituito un errore per {project_name}.")
        
        except Exception as e:
            print(f"[ERRORE CRITICO] Impossibile avviare lo script per {project_name}: {e}")

    print("\n" + "="*40)
    print("BATCH PROCESSING COMPLETATO")
    print("="*40)

if __name__ == "__main__":
    run_automation()