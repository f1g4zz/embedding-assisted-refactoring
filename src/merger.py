import pandas as pd
import argparse
from pathlib import Path

# Definizione delle colonne etichetta
LABELS = [
    "Extract Method", "Extract And Move Method", "Extract Variable",
    "Inline Variable", "Split Variable", "Parameterize Variable",
    "Merge Variable", "Replace Pipeline", "Invert Condition",
    "Merge Conditional Expresion"
]

def merge_labeled_files(input_dir, output_file):
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    if not input_path.is_dir():
        print(f"Errore: {input_dir} non è una directory valida.")
        return

    all_dataframes = []
    
    # Contatori per il riepilogo
    total_removed = 0
    total_kept = 0
    
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"Nessun file CSV trovato in {input_dir}")
        return

    print(f"Trovati {len(csv_files)} file. Inizio elaborazione...")

    for file in csv_files:
        if file.name == output_path.name:
            continue
            
        df = pd.read_csv(file)
        
        # Verifica che le colonne label esistano
        missing_labels = [l for l in LABELS if l not in df.columns]
        if missing_labels:
            print(f"Attenzione: Il file {file.name} non contiene le colonne etichetta. Salto.")
            continue

        # --- LOGICA DI FILTRAGGIO E CONTEGGIO ---
        # Identifichiamo le righe che hanno almeno un 1 (o valore != 0) nelle colonne LABELS
        has_labels_mask = (df[LABELS] != 0).any(axis=1)
        
        num_zeros = (~has_labels_mask).sum()  # Quanti verranno rimossi
        num_with_labels = has_labels_mask.sum() # Quanti verranno tenuti
        
        total_removed += num_zeros
        total_kept += num_with_labels
        
        # Filtriamo il dataframe tenendo solo i positivi
        filtered_df = df[has_labels_mask]
        
        print(f"File {file.name}: Rimossi {num_zeros} esempi vuoti, mantenuti {num_with_labels}.")
        
        all_dataframes.append(filtered_df)

    if not all_dataframes:
        print("Nessun dato valido (con etichette) da unire.")
        return

    # Unione dei soli dataframe filtrati
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Creazione cartella di output se non esiste
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Salvataggio
    final_df.to_csv(output_path, index=False)
    
    # --- REPORT FINALE ---
    print(f"\n" + "="*40)
    print(f"--- OPERAZIONE COMPLETATA ---")
    print(f"Esempi totali scartati (tutti 0): {total_removed}")
    print(f"Esempi totali salvati:           {total_kept}")
    print(f"Percentuale di righe utili:      {(total_kept/(total_kept + total_removed)*100):.2f}%")
    print(f"File creato: {output_path}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSV filtrando solo gli esempi con almeno una label.")
    parser.add_argument('--dir', required=True, help='Cartella contenente i file CSV')
    parser.add_argument('--out', default='merged_dataset_filtered.csv', help='Nome del file CSV finale')
    
    args = parser.parse_args()

    # Risoluzione percorsi
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    merge_labeled_files(resolve(args.dir), resolve(args.out))