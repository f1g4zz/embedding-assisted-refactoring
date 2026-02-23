import pandas as pd
import argparse
import sys
from pathlib import Path

def main(input_path, output_path):
    try:
        # 1. Carico il file CSV
        # Pandas accetta direttamente oggetti Path
        df = pd.read_csv(input_path)
        
        required_columns = {'class_name', 'refactoring'}
        if not required_columns.issubset(df.columns):
            print(f"Errore: Il file deve contenere le colonne {required_columns}")
            sys.exit(1)

        # 2. Analisi
        unique_classes_count = df['class_name'].nunique()
        print(f"Total unique classes: {unique_classes_count}")

        class_analysis = df.groupby('class_name')['refactoring'].agg(
            count='count', 
            types=lambda x: list(x.unique())
        ).reset_index()

        class_analysis.columns = ['Classe', 'Numero Refactoring', 'Tipi di Refactoring']
        class_analysis = class_analysis.sort_values(by='Numero Refactoring', ascending=False)

        # 3. Salvataggio
        class_analysis.to_csv(output_path, index=False)
        print(f"Analisi completata. Risultati salvati in: {output_path}")

        print("\nPrime 10 righe del risultato:")
        print(class_analysis.head(10))

    except FileNotFoundError:
        print(f"Errore: Il file '{input_path}' non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisi dei refactoring per classe da un file CSV.")
    parser.add_argument('-i', '--input', required=True, help="Percorso del file CSV di input")
    parser.add_argument('-o', '--output', required=True, help="Percorso del file CSV di output")

    args = parser.parse_args()
    
    # Definizione della BASE_DIR (3 livelli sopra la cartella dello script)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    def resolve_path(p):
        path = Path(p)
        # Se il percorso è assoluto lo tiene così, altrimenti lo attacca alla BASE_DIR
        return path if path.is_absolute() else BASE_DIR / path

    main(resolve_path(args.input), resolve_path(args.output))