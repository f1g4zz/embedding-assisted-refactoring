import pandas as pd

# 1. Carica il file CSV
# Sostituisci 'tuo_file.csv' con il nome del tuo file
df = pd.read_csv('titan-Class.csv')

# 2. Filtra le righe
# Teniamo solo dove MOVE_METHOD è >= 1
df_filtrato = df[df['RENAME_METHOD'] >= 1]

# 3. Salva il risultato in un nuovo CSV
df_filtrato.to_csv('risultato_filtrato.csv', index=False)

print(f"Operazione completata! Righe totali dopo il filtro: {len(df_filtrato)}")