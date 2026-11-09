import pandas as pd
t = pd.read_csv("results/episodios_edb.csv")
print(t[(t["tipo"] == "ST") & ~t["cambio_eje"]]["duracion_s"].describe())

print(t["tipo"].value_counts())