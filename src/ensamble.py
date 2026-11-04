""" ||||||||||||  8VO PASO: ENSAMBLE DEL PERFIL DE ST ||||||||||||

Ensamble del perfil de ST alineado al inicio del episodio.

Muestra cómo evoluciona el nivel de ST antes, durante y después de un
episodio. Cada episodio anotado se alinea en t = 0 (su inicio) y se le resta su
nivel basal (mediana del ST entre basal_ini_s y basal_fin_s). En cada instante
relativo se calculan la mediana y el IQR (p25-p75) entre episodios.

Se hace por separado para elevaciones (signo "+") y depresiones (signo "-"),
porque promediarlas juntas las cancela. Se conservan todas las derivaciones: un
mismo evento puede aparecer en varias a la vez y un registro aporta varios
episodios, así que los episodios no son independientes y el IQR describe la
variabilidad entre episodios.

Exclusiones (en este orden): derivación sin serie ST, basal incompleto (menos de
fraccion_valida_basal de datos o episodio antes de t_pre_s) y basal contaminado
(otro episodio anotado de la misma derivación se superpone con la ventana basal).

Uso:
    python src/ensamble.py [base]
Salidas:
    results/ensamble_<base>.csv                 mediana, p25, p75 y n por instante y signo
    results/ensamble_exclusiones_<base>.csv     episodios incluidos y excluidos
    results/figuras/ensamble_st_<base>.png
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rutas import CONFIG, RESULTS

NOMBRES_SIGNO = {"+": "Elevación de ST", "-": "Depresión de ST"}
COLORES_SIGNO = {"+": "#a04db5", "-": "#ff2323"}


def leer_parametros():
    """Devuelve el diccionario de parámetros de config/ensamble.json."""
    with open(CONFIG / "ensamble.json", encoding="utf-8") as f:
        return json.load(f)


def episodios_st(gt_df):
    """Episodios ST anotados, sin cambios de eje (que son cambios de posición,
    no isquemia)."""
    return gt_df[(gt_df["tipo"] == "ST") & ~gt_df["cambio_eje"]].reset_index(drop=True)


def perfil_episodio(t_s, st_mv, ep, otros, grilla_rel, p):
    """Perfil de ST de un episodio en la grilla relativa, menos su basal.

    Entradas: t_s y st_mv (serie de la derivación, t_s creciente), ep (fila del
    episodio), otros (episodios de la misma derivación), grilla_rel (s) y p.
    Salida: (curva en mV o None, motivo de exclusión o None). Los motivos son
    "basal incompleto" y "basal contaminado".
    """
    inicio = ep["inicio_s"]
    curva = np.interp(inicio + grilla_rel, t_s, st_mv, left=np.nan, right=np.nan)
    en_basal = (grilla_rel >= p["basal_ini_s"]) & (grilla_rel <= p["basal_fin_s"])
    if np.isfinite(curva[en_basal]).mean() < p["fraccion_valida_basal"]:
        return None, "basal incompleto"

    ini_basal, fin_basal = inicio + p["basal_ini_s"], inicio + p["basal_fin_s"]
    otros = otros.drop(ep.name)
    if ((otros["fin_s"] > ini_basal) & (otros["inicio_s"] < fin_basal)).any():
        return None, "basal contaminado"
    return curva - np.nanmedian(curva[en_basal]), None


def construir_ensamble(st_df, ref, p):
    """Arma el ensamble por signo.

    Entradas: st_df (serie ST: registro, derivacion, t_s, st_mv), ref (episodios
    ST anotados, de episodios_st) y p.
    Salidas: tabla del ensamble (signo, t_relativo_s, st_mediana_mv, st_p25_mv,
    st_p75_mv, n_episodios), tabla de exclusiones por signo y diccionario
    {signo: duraciones (s) de los episodios incluidos}. En cada instante, las
    estadísticas quedan en NaN si hay menos de n_min_por_instante episodios.
    """
    fs = 1.0 / np.median(np.diff(st_df.loc[st_df["registro"] == st_df["registro"].iloc[0], "t_s"]))
    grilla_rel = np.arange(-p["t_pre_s"], p["t_post_s"], 1.0 / fs)
    series = {k: (g["t_s"].to_numpy(), g["st_mv"].to_numpy())
              for k, g in st_df.sort_values("t_s").groupby(["registro", "derivacion"])}
    mismos = {k: g for k, g in ref.groupby(["registro", "derivacion"])}

    curvas = {"+": [], "-": []}
    duraciones = {"+": [], "-": []}
    conteo = {s: {"Episodios anotados": 0, "Sin serie ST": 0, "Basal incompleto": 0,
                  "Basal contaminado": 0} for s in curvas}
    for idx, ep in ref.iterrows():
        s = ep["signo"]
        conteo[s]["Episodios anotados"] += 1
        clave = (ep["registro"], ep["derivacion"])
        if clave not in series:
            conteo[s]["Sin serie ST"] += 1
            continue
        curva, motivo = perfil_episodio(*series[clave], ep, mismos[clave], grilla_rel, p)
        if curva is None:
            conteo[s][motivo.capitalize()] += 1
            continue
        curvas[s].append(curva)
        duraciones[s].append(ep["fin_s"] - ep["inicio_s"])

    filas = []
    for s, lista in curvas.items():
        if not lista:
            continue
        matriz = np.array(lista)
        n = np.isfinite(matriz).sum(axis=0)
        suficiente = n >= p["n_min_por_instante"]
        with np.errstate(all="ignore"):
            med, q25, q75 = (np.where(suficiente, f(matriz), np.nan) for f in (
                lambda m: np.nanmedian(m, axis=0),
                lambda m: np.nanpercentile(m, 25, axis=0),
                lambda m: np.nanpercentile(m, 75, axis=0)))
        filas.append(pd.DataFrame({"signo": s, "t_relativo_s": grilla_rel,
                                   "st_mediana_mv": med, "st_p25_mv": q25,
                                   "st_p75_mv": q75, "n_episodios": n}))
    ensamble = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(
        columns=["signo", "t_relativo_s", "st_mediana_mv", "st_p25_mv", "st_p75_mv", "n_episodios"])

    exclusiones = pd.DataFrame(conteo).T
    exclusiones["Episodios incluidos"] = [len(curvas[s]) for s in exclusiones.index]
    exclusiones.index = [NOMBRES_SIGNO[s] for s in exclusiones.index]
    return ensamble, exclusiones, duraciones


def figura_ensamble(ensamble, duraciones, ruta):
    """Figura con un panel por signo: mediana e IQR del cambio de ST (arriba) y
    cantidad de episodios en cada instante (abajo). La línea punteada vertical
    marca la duración mediana de los episodios incluidos."""
    signos = [s for s in ("+", "-") if s in set(ensamble["signo"])]
    fig, ejes = plt.subplots(2, len(signos), figsize=(6 * len(signos), 5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for j, s in enumerate(signos):
        e = ensamble[ensamble["signo"] == s]
        color = COLORES_SIGNO[s]
        ax, axn = ejes[0, j], ejes[1, j]
        ax.plot(e["t_relativo_s"], e["st_mediana_mv"], color=color, lw=2, label="Mediana")
        ax.fill_between(e["t_relativo_s"], e["st_p25_mv"], e["st_p75_mv"], color=color,
                        alpha=0.25, label="IQR (p25–p75)")
        ax.axvline(0, color="k", ls="--", alpha=0.7, label="Inicio anotado")
        ax.axvline(np.median(duraciones[s]), color="gray", ls=":",
                   label="Duración mediana")
        ax.set_title(NOMBRES_SIGNO[s] + " (n = " + str(len(duraciones[s])) + " episodios)")
        ax.set_ylabel("Cambio de ST (mV)")
        ax.grid(color="#e6e6e6")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        axn.plot(e["t_relativo_s"], e["n_episodios"], color=color)
        axn.set_ylabel("Episodios con dato")
        axn.set_xlabel("Tiempo relativo al inicio (s)")
        axn.grid(color="#e6e6e6")
        axn.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("\n>>>>>>>>>>>   Paso 8: ensamble de ST alineado al inicio   <<<<<<<<<<<<")
    base = sys.argv[1] if len(sys.argv) > 1 else "edb"
    p = leer_parametros()
    st_df = pd.read_csv(RESULTS / f"st_serie_{base}.csv.gz")
    gt_df = pd.read_csv(RESULTS / f"episodios_{base}.csv")
    st_df["registro"] = st_df["registro"].astype(str)
    gt_df["registro"] = gt_df["registro"].astype(str)

    ensamble, exclusiones, duraciones = construir_ensamble(st_df, episodios_st(gt_df), p)
    ensamble.to_csv(RESULTS / f"ensamble_{base}.csv", index=False, float_format="%.6g")
    exclusiones.to_csv(RESULTS / f"ensamble_exclusiones_{base}.csv")

    print("\n --------------------------------- Episodios incluidos y excluidos --------------------------------- ")
    print(exclusiones.to_string())
    print("\n --------- Duración de los episodios incluidos (s) --------- ")
    print(pd.DataFrame({NOMBRES_SIGNO[s]: pd.Series(d).describe() for s, d in duraciones.items()
                        if d}).T.to_string(float_format="%.0f"))

    (RESULTS / "figuras").mkdir(exist_ok=True)
    figura_ensamble(ensamble, duraciones, RESULTS / "figuras" / f"ensamble_st_{base}.png")
    print(f"\nFIGURA ensamble_st_{base}.png: mediana e IQR del cambio de ST alineado al inicio "
          "del episodio, por signo, con la cantidad de episodios en cada instante.")