import pandas as pd

# 1. Carico il file CSV
df = pd.read_csv('C:\Users\lanza\Downloads\papersEvolution\DesigniteJava\matches')

# 2. Calcolo il numero totale di classi uniche
unique_classes_count = df['class_name'].nunique()
print(f"Total unique classes: {unique_classes_count}")

# 3. Raggruppo per 'class_name'
# Per ogni classe:
# - Conto quante righe ci sono ('count')
# - Estraggo i nomi dei refactoring applicati senza duplicati (unique)
class_analysis = df.groupby('class_name')['refactoring'].agg(
    count='count', 
    types=lambda x: list(x.unique())
).reset_index()

# 4. Rinomino le colonne per chiarezza
class_analysis.columns = ['Classe', 'Numero Refactoring', 'Tipi di Refactoring']

# 5. Ordino i risultati in base al numero di refactoring (decrescente)
class_analysis = class_analysis.sort_values(by='Numero Refactoring', ascending=False)

# 6. Salvo il risultato in un nuovo file CSV
class_analysis.to_csv('class_refactoring_analysis.csv', index=False)

# 7. Visualizzo le prime righe per controllo
print(class_analysis.head(10))