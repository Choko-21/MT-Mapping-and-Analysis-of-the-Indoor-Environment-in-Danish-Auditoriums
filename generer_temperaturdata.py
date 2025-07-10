from pythermalcomfort.models import pmv_ppd_iso, set_tmp
import pythermalcomfort
from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.utilities import v_relative
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob
import unicodedata
import re
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

extract_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/spørgeskema - svar ved påske (KEA RUC CBS)/BOBOBOB"


def normalize_name(name):
    name = unicodedata.normalize("NFKD", name)
    name = "".join([c for c in name if not unicodedata.combining(c)])
    name = name.lower().replace("ø", "oe").replace("å", "aa").replace("æ", "ae")
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


svarfelt_mapping = {
    "temp_feeling": "t",
    "temp_feeling_accpetable": "t",
    "temp_feel_change": "t",
    "temp_prefer": "t",
    "temp_wearing": "t",
    "quality_freshness_t": "t",
    "quality_odor": "co2",
    "quality_freshness": "co2",
    "quality_acceptable": "co2",
    "quality_change_acceptable": "co2",
    "quality_odor_voc": "voc",
    "quality_freshness_voc": "voc",
    "quality_acceptable_voc": "voc",
    "quality_change_acceptable_voc": "voc",
    "light_level": "light",
    "light_acceptable": "light",
    "light_prefer": "light",
    "light_glare": "light",
    "noise_level": "noise",
    "noise_acceptable": "noise",
    "noise_dirturbing": "noise"
}

labels_path = os.path.join(os.path.dirname(extract_path), "labels__medtal.csv")
labels_raw = pd.read_csv(labels_path, header=None)
split_rows = labels_raw[0].str.split(";", expand=True)
split_rows.columns = ["variable", "værdi", "label"]
split_rows["label"] = split_rows["label"].str.strip('"')
split_rows = pd.concat([
    split_rows,
    split_rows[split_rows["variable"] == "quality_freshness"].assign(
        variable="quality_freshness_t")
])

# ➕ Tilføj labels til voc-felter ved at genbruge CO₂-felter
voc_fields = [
    "quality_odor_voc", "quality_freshness_voc",
    "quality_acceptable_voc", "quality_change_acceptable_voc"
]
for voc_field in voc_fields:
    source_field = voc_field.replace("_voc", "")
    if source_field in split_rows["variable"].values:
        voc_labels = split_rows[split_rows["variable"] == source_field].copy()
        voc_labels["variable"] = voc_field
        split_rows = pd.concat([split_rows, voc_labels], ignore_index=True)


# Indlæs spørgeskemasvar
svar_files = glob.glob(os.path.join(extract_path, "svar/*.csv"))
all_svar = []
for file in svar_files:
    try:
        df = pd.read_csv(file, sep=";", encoding="utf-8", low_memory=False)
    except:
        df = pd.read_csv(file, sep=";", encoding="ISO-8859-1", low_memory=False)
    df['filnavn'] = os.path.basename(file).replace("dataset_", "").replace(".csv", "")
    all_svar.append(df)

svar_df = pd.concat(all_svar, ignore_index=True)

# Konverter starttime til datetime – robust mod blandede formater


def parse_time_safe(value):
    for fmt in ("%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H.%M.%S"):
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    # Fallback med korrekt tolkning
    return pd.to_datetime(value, dayfirst=True, errors='coerce')


svar_df['tid'] = svar_df['starttime'].apply(parse_time_safe).dt.floor("min")

svar_df['lokale'] = svar_df['filnavn']
svar_df['lokale_norm'] = svar_df['lokale'].apply(normalize_name)

# 🧩 Mapping fra lokalnavn (fra sheet-navn) til svar-filer
belægning_navn_to_spørgeskema = {
    "CBS 033": "CBS DH.C.0.33",
    "KU A81": "KU 782-81",
    "KU A11": "KU 782-11",
    "KU HCØ": "KU HCØ 1",
    "RUC A25": "RUC 25.2-035",
    "RUC A01": "RUC 00.1-009",
    "CBS 202": "CBS SP202",
    "Ålborg selma 300": "AAU Selma 300",
    "Ålborg fib10": "AAU FIB 10",
    "Ålborg krogh": "AAU_Krogh3",
    "Århus 105": "AU 1482-105",
    "Århus 113": "AU 1441-113",
    "Emdrup": "Emdrup D169",
    "DTU 81": "DTU_B116_81",
    "DTU 83": "DTU_B116_83",
    "DTU 42": "DTU_b303a-42",
    "DTU 49": "DTU_b303a-49",
    "DTU A34": "DTU_B306_34",
    "KEA": "KEA",
}

# 🔄 Omvendt mapping: spørgsmålsfil → belægningsnavn
svar_df["belægning_favn"] = svar_df["filnavn"].map(
    {v: k for k, v in belægning_navn_to_spørgeskema.items()})
svar_df["belægning_favn_norm"] = svar_df["belægning_favn"].apply(
    lambda x: normalize_name(x) if pd.notna(x) else x
)


svar_df['join_tid'] = svar_df['tid'].astype(str)
svar_df['join_lokale'] = svar_df['lokale_norm']
svar_df["quality_freshness_t"] = svar_df["quality_freshness"]
svar_df["quality_odor_voc"] = svar_df["quality_odor"]
svar_df["quality_freshness_voc"] = svar_df["quality_freshness"]
svar_df["quality_acceptable_voc"] = svar_df["quality_acceptable"]
svar_df["quality_change_acceptable_voc"] = svar_df["quality_change_acceptable"]


print("Antal ugyldige svar tider:", svar_df['tid'].isna().sum())


sensor_to_room = {
    'TAIL-21': ['DTU_B306_34', 'DTU_B116_83', 'CBS DH.C.0.33', 'KU 782-11', 'Emdrup D169', 'AU 1441-113', 'AAU FIB 10'],
    'TAIL-22': ['DTU_B306_34', 'DTU_b303a-49', 'CBS DH.C.0.33', 'RUC 25.2-035', 'Emdrup D169', 'KU 782-81', 'AU 1441-113', 'AAU FIB 10', 'KU HCØ 1'],
    'TAIL-23': ['DTU_B306_34', 'DTU_B116_83', 'CBS DH.C.0.33', 'KU 782-11', 'Emdrup D169', 'AU 1441-113', 'AAU FIB 10', 'KU HCØ 1'],
    'TAIL-24': ['DTU_B306_34', 'DTU_B116_83', 'CBS DH.C.0.33', 'KU 782-81', 'Emdrup D169', 'AU 1441-113', 'AAU Selma 300'],
    'TAIL-25': ['DTU_B306_34', 'DTU_B116_83', 'DTU_b303a-42', 'KEA', 'RUC 00.1-009', 'CBS DH.C.0.33', 'KU 782-11', 'Emdrup D169', 'AAU Selma 300'],
    'TAIL-26': ['DTU_B306_34', 'DTU_b303a-42', 'KEA', 'RUC 25.2-035', 'CBS SP202', 'AAU Selma 300'],
    'TAIL-27': ['DTU_B306_34', 'RUC 00.1-009', 'CBS SP202', 'KU 782-11', 'AU 1482-105', 'AAU_Krogh3'],
    'TAIL-28': ['DTU_B306_34', 'DTU_b303a-49', 'KEA', 'RUC 00.1-009', 'CBS SP202', 'KU 782-81', 'DTU_B116_81', 'AU 1482-105', 'AAU_Krogh3'],
    'TAIL-29': ['DTU_B306_34', 'CBS SP202', 'KU 782-81', 'DTU_B116_81', 'AU 1482-105', 'AAU_Krogh3', 'KU HCØ 1'],
    'TAIL-30': ['DTU_B306_34', 'DTU_b303a-42', 'KEA', 'RUC 00.1-009', 'CBS SP202', 'KU 782-81', 'DTU_B116_81', 'AU 1482-105', 'AAU_Krogh3', 'KU HCØ 1']
}

tail_files = glob.glob(os.path.join(extract_path, "målinger/TAIL-*.csv"))
målinger = []
for file in tail_files:
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except:
        df = pd.read_csv(file, encoding='ISO-8859-1')
    df['sensor'] = os.path.basename(file).replace(".csv", "")

    # Robust konvertering af målingstid fra UNIX-tid
    def safe_unix_to_datetime(ts):
        try:
            return pd.to_datetime(ts, unit='s')
        except Exception:
            return pd.NaT

    df['tid'] = pd.to_datetime(df['ts'], unit='s', utc=True) \
        .dt.tz_convert('Europe/Copenhagen') \
        .dt.tz_localize(None) \
        .dt.floor('min')

    df['lokale_match'] = df['sensor'].map(sensor_to_room)
    df = df.explode('lokale_match')
    df['lokale_match_norm'] = df['lokale_match'].apply(normalize_name)
    df['join_tid'] = df['tid'].astype(str)
    df['join_lokale'] = df['lokale_match_norm']

    målinger.append(df)
måling_df_all = pd.concat(målinger, ignore_index=True)
print("Antal ugyldige måle tider:", måling_df_all['tid'].isna().sum())

# 🔍 Lav oversigt over unikke tider og matches mellem svar og målinger per lokale
records = []
for navn in svar_df["filnavn"].unique():
    svar_subset = svar_df[svar_df["filnavn"] == navn].dropna(subset=["tid"])
    måle_subset = måling_df_all[måling_df_all["lokale_match"] == navn].dropna(subset=["tid"])

    svar_tider = svar_subset["tid"].unique()
    måle_tider = måle_subset["tid"].unique()
    matches = set(svar_tider) & set(måle_tider)

    datoer = pd.to_datetime(svar_subset["tid"]).dt.date.unique()
    records.append({
        "filnavn": navn,
        "unik_tid_svar": len(svar_tider),
        "unik_tid_målinger": len(måle_tider),
        "match_tid": len(matches)
    })

match_df = pd.DataFrame(records)
print("\n🧾 Oversigt: Tidspunkter i svar og målinger per lokale\n")
print(match_df.sort_values("match_tid", ascending=False).to_string(index=False))

# Generér plots
for svarfelt, målefelt in svarfelt_mapping.items():

    svar_subset = svar_df[['starttime', 'lokale',
                           svarfelt, 'join_tid', 'join_lokale']].dropna()
    # Før data duplikeres: tæl unikke svar per kategori
    svar_kategorier = svar_subset[svarfelt].value_counts().sort_index()
    svar_pct = (svar_kategorier / svar_kategorier.sum()) * 100

    måling_df = måling_df_all[['tid', 'sensor',
                               målefelt, 'join_tid', 'join_lokale']].dropna()
    måling_df = måling_df.rename(columns={målefelt: 'værdi'})

    merged = pd.merge(svar_subset, måling_df, how='inner', on=['join_tid', 'join_lokale'])
    merged['måling_nr'] = merged.groupby(['starttime', 'lokale']).cumcount() + 1
    filtered = merged[merged['måling_nr'] <= 10]

    # Find labels
    label_df = split_rows[split_rows["variable"] == svarfelt].copy()
    label_df["værdi"] = pd.to_numeric(label_df["værdi"], errors="coerce")
    labels_dict = label_df.set_index("værdi")["label"].to_dict()

    # Smelt og konverter
    melted = filtered[[svarfelt, 'værdi']].copy().dropna()
    melted["værdi"] = pd.to_numeric(melted["værdi"], errors="coerce")  # vigtig for y-aksen

    # Lav label-kolonne
    melted[f"{svarfelt}_label"] = melted[svarfelt].map(labels_dict)
    melted[f"{svarfelt}_label"] = melted[f"{svarfelt}_label"].fillna(melted[svarfelt])
    melted[f"{svarfelt}_label"] = melted[f"{svarfelt}_label"].astype(str)

    # Bestem rækkefølge
    ordered_labels = [str(labels_dict[i]) if i in labels_dict else str(i)
                      for i in sorted(melted[svarfelt].dropna().unique())]
    melted[f"{svarfelt}_label"] = pd.Categorical(
        melted[f"{svarfelt}_label"], categories=ordered_labels, ordered=True
    )

    # Plot
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=melted, x=f"{svarfelt}_label",
                   y="værdi", inner="box", density_norm="count")
    plt.xlabel(svarfelt.replace("_", " ").capitalize())
    # Dynamisk y-akse label
    if målefelt == "t":
        y_label = "Temperature [°C]"
    elif målefelt == "co2":
        y_label = "CO₂ [ppm]"
    elif målefelt == "light":
        y_label = "Light [Lux]"
    elif målefelt == "noise":
        y_label = "Sound [dB]"
    elif målefelt == "voc":
        y_label = "TVOC [ppb]"
    else:
        y_label = "Målt værdi"

    plt.ylabel(y_label)
    # ➕ Lås y-aksen for plots
    if målefelt == "light":
        plt.ylim(top=1000)
    elif målefelt == "voc":
        plt.ylim(top=1)

    # Tilføj vandrette gridlinjer
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.title("")
    plt.xticks(rotation=30, ha="right")
    # ➕ Tilføj labels under hver kategori
    ax = plt.gca()
    for i, label in enumerate(ordered_labels):
        # count = svar_kategorier.get(i+1, 0)
        pct = svar_pct.get(i+1, 0)
        ax.text(i, 0.05, f"({pct:.0f}%)", ha="center",
                va="top", fontsize=8, transform=ax.get_xaxis_transform())

    # 🔽 TILFØJ HER: Tendenslinjer (median, Q1 og Q3 per kategori)
    grouped = melted.groupby(f"{svarfelt}_label", observed=False)["værdi"]
    x_positions = range(len(grouped))

    # Median (Q2)
    plt.plot(x_positions, grouped.median().values,
             color="red", linestyle="-", marker='o', linewidth=2, alpha=0.6)

    # Q1 og Q3
    plt.plot(x_positions, grouped.quantile(0.25).values,
             color="red", linestyle="--", linewidth=1.5, alpha=0.6)
    plt.plot(x_positions, grouped.quantile(0.75).values,
             color="red", linestyle="--", linewidth=1.5, alpha=0.6)
    plt.tight_layout()

    # Opret plots-mappe hvis den ikke findes
    plotsdone_dir = os.path.join(extract_path, "plots done")
    os.makedirs(plotsdone_dir, exist_ok=True)

    # Sæt sti for output plot
    output_path = os.path.join(plotsdone_dir, f"violinplot_{svarfelt}.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()


print("✔ Alle de første plots genereret og gemt.")


# 🔁 Ekstra plots: Densitet (antal personer / kapacitet) vs. quality_-svar

# 📥 Indlæs Excel med belægning
excel_path = os.path.join(extract_path, "målinger", "Antal personer done.xlsx")
sheet_dict = pd.read_excel(excel_path, sheet_name=None)

# 🧩 Mapping fra lokalnavn (fra sheet-navn) til kapacitet
filnavn_to_kapacitet = {
    'DTU A34': 194,
    'DTU 49': 80,
    'DTU 42': 250,
    'KEA': 51,
    'RUC A25': 120,
    'RUC A01': 250,
    'CBS 202': 205,
    'CBS 033': 194,
    'KU A11': 200,
    'KU A81': 293,
    'DTU 81': 275,
    'DTU 83': 100,
    'Emdrup': 238,
    'Århus 105': 197,
    'Århus 113': 81,
    'Ålborg krogh': 150,
    'Ålborg fib10': 111,
    'Ålborg selma 300': 115,
    'KU HCØ': 386
}


# 📊 Læs og saml alle sheets
sheets = []
for sheet_name, df in sheet_dict.items():
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()  # fjern mellemrum, lav lowercase
    if "ts" not in df or "personer" not in df:
        continue
    df = df[["ts", "personer"]].dropna()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["filnavn"] = sheet_name.strip()
    sheets.append(df)

personer_df = pd.concat(sheets)

# 📈 Interpolér én gang per lokale
interpolerede = []
for navn, gruppe in personer_df.groupby('filnavn'):
    gruppe = gruppe.drop_duplicates(subset='ts')
    gruppe = gruppe.infer_objects(copy=False)  # 🔧 gør kolonner numeriske hvor muligt
    g = gruppe.set_index('ts').infer_objects(copy=False).resample(
        '1min').interpolate('linear').reset_index()

    g["filnavn"] = navn
    interpolerede.append(g)

belægning_df = pd.concat(interpolerede, ignore_index=True)

# ➕ Beregn kapacitet og belægningsprocent
belægning_df["kapacitet"] = belægning_df["filnavn"].map(filnavn_to_kapacitet)
belægning_df["procent"] = (belægning_df["personer"] / belægning_df["kapacitet"]) * 100
belægning_df["tid"] = belægning_df["ts"].dt.floor("min")
belægning_df["join_lokale"] = belægning_df["filnavn"].apply(normalize_name)

# 🔍 Lav oversigt over unikke tider og matches per lokale
records = []
for navn in belægning_df["filnavn"].unique():
    navn_norm = normalize_name(navn)
    svar_subset = svar_df[svar_df["belægning_favn_norm"] == navn_norm].dropna(subset=["tid"])
    bel_subset = belægning_df[belægning_df["filnavn"] == navn].dropna(subset=["tid"])

    svar_tider = svar_subset["tid"].unique()
    bel_tider = bel_subset["tid"].unique()
    matches = set(svar_tider) & set(bel_tider)

    datoer = pd.to_datetime(svar_subset["tid"]).dt.date.unique()
    records.append({
        "filnavn": navn,
        "unik_tid_svar": len(svar_tider),
        "unik_tid_belægning": len(bel_tider),
        "match_tid": len(matches)
    })

match_df = pd.DataFrame(records)
print("\n🧾 Oversigt: Tidspunkter i svar og belægning per lokale\n")
print(match_df.sort_values("match_tid", ascending=False).to_string(index=False))


# 🎯 Generér plots: Belægningsprocent vs. quality_-svar
for svarfelt in svar_df.columns:
    if not svarfelt.startswith("quality_"):
        continue

    svar_subset = svar_df[["starttime", svarfelt, "tid", "join_lokale"]].dropna()
    merged = pd.merge(svar_subset, belægning_df[["tid", "join_lokale", "procent"]],
                      how="left", on=["tid", "join_lokale"])
    plotdata = merged[[svarfelt, "procent"]].dropna()

    # 🏷️ Find og brug labels
    label_df = split_rows[split_rows["variable"] == svarfelt].copy()
    label_df["værdi"] = pd.to_numeric(label_df["værdi"], errors="coerce")
    labels_dict = label_df.set_index("værdi")["label"].to_dict()
    plotdata[f"{svarfelt}_label"] = plotdata[svarfelt].map(
        labels_dict).fillna(plotdata[svarfelt]).astype(str)

    ordered_labels = [str(labels_dict[i]) if i in labels_dict else str(i)
                      for i in sorted(plotdata[svarfelt].dropna().unique())]
    plotdata[f"{svarfelt}_label"] = pd.Categorical(
        plotdata[f"{svarfelt}_label"], categories=ordered_labels, ordered=True
    )

    # 📊 Plot
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=plotdata, x=f"{svarfelt}_label",
                   y="procent", inner="box", density_norm="area")
    plt.ylabel("Occupancy [%]")
    plt.xlabel(svarfelt.replace("_", " ").capitalize())
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.title("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = os.path.join(plotsdone_dir, f"belægning_{svarfelt}.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()

print("✔ Alle plots genereret og gemt.")


# ----------------------
# CLO mapping
# ----------------------
clo_mapping = {
    1: 0.36,
    2: 0.57,
    3: 0.74,
    4: 1.15
}

svar_df["clo"] = svar_df["temp_wearing"].map(clo_mapping).fillna(0.57)

# ----------------------
# Merge svar + målinger
# ----------------------
merged_df = pd.merge(
    svar_df,
    måling_df_all,
    how="inner",
    on=["join_tid", "join_lokale"]
)

# Filtrer kun rækker med temp_feeling besvaret
merged_df = merged_df.dropna(subset=["temp_feeling", "t", "h"])

# ----------------------
# Beregn PMV og PPD
# ----------------------


def beregn_pmv(row):
    try:
        tdb = float(row["t"])
        tr = float(row["t"])  # samme som tdb
        rh = float(row["h"])
        clo = float(row["clo"])
        res = pmv_ppd_iso(
            tdb=tdb,
            tr=tr,
            vr=0.1,
            rh=rh,
            met=1.2,
            clo=clo
        )
        return pd.Series([res["pmv"], res["ppd"]])
    except Exception as e:
        print(f"Fejl ved beregning af PMV for række:\n{row}\nFejl: {e}")
        return pd.Series([np.nan, np.nan])


# Sørg for numeriske værdier
merged_df["t"] = pd.to_numeric(merged_df["t"], errors="coerce")
merged_df["h"] = pd.to_numeric(merged_df["h"], errors="coerce")
merged_df["clo"] = pd.to_numeric(merged_df["clo"], errors="coerce")


merged_df[["pmv", "ppd"]] = merged_df.apply(beregn_pmv, axis=1)

# ----------------------
# Plot PMV vs temp_feeling
# ----------------------

# Map labels fra dine labels-filer
label_df = split_rows[split_rows["variable"] == "temp_feeling"].copy()
label_df["værdi"] = pd.to_numeric(label_df["værdi"], errors="coerce")
labels_dict = label_df.set_index("værdi")["label"].to_dict()

merged_df["temp_feeling_label"] = merged_df["temp_feeling"].map(
    labels_dict).fillna(merged_df["temp_feeling"]).astype(str)

# Bevar rækkefølge
ordered_labels = [str(labels_dict[i]) if i in labels_dict else str(i)
                  for i in sorted(merged_df["temp_feeling"].dropna().unique())]

merged_df["temp_feeling_label"] = pd.Categorical(
    merged_df["temp_feeling_label"],
    categories=ordered_labels,
    ordered=True
)

plt.figure(figsize=(10, 5))
sns.violinplot(
    data=merged_df,
    x="temp_feeling_label",
    y="pmv",
    inner="quartile",
    density_norm="count"
)
plt.xlabel("Temp feeling (spørgeskema)")
plt.ylabel("PMV")
plt.title("PMV vs. Temp Feeling")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()


plt.figure(figsize=(10, 5))
sns.violinplot(
    data=merged_df,
    x="temp_feeling_label",
    y="ppd",
    inner="quartile",
    density_norm="count"
)
plt.xlabel("Temp feeling (spørgeskema)")
plt.ylabel("PPD [%]")
plt.title("PPD vs. Temp Feeling")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=30, ha="right")

# Tilføj komfortgrænse som horisontal linje
plt.axhline(10, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
plt.text(len(ordered_labels)-0.5, 12, "10 % grænse", color="red", fontsize=9, ha="right")

plt.tight_layout()

# Gem plot
plt.savefig(os.path.join(plotsdone_dir, "PPD_vs_temp_feeling.png"))
plt.show()
plt.close()

print("✔ PPD-plot gemt!")


# ----------------------
# Lav tabel pr universitet
# ----------------------

# Udtræk universitet fra belægning_favn
merged_df["universitet"] = merged_df["belægning_favn"].str.extract(r"(\w+)")

# Beregn gennemsnit
summary = merged_df.groupby("universitet").agg(
    antal_målinger=("pmv", "count"),
    gennemsnit_pmv=("pmv", "mean"),
    gennemsnit_ppd=("ppd", "mean")
).reset_index()

print("\n📝 Tabel pr universitet:")
print(summary)

# Gem CSV
summary_path = os.path.join(extract_path, "PMV_summary_per_universitet.csv")
summary.to_csv(summary_path, index=False, sep=";")
print(f"✔ Tabel gemt som {summary_path}")

# Gem også hele merged_df hvis du vil
full_path = os.path.join(extract_path, "PMV_merged_data.csv")
merged_df.to_csv(full_path, index=False, sep=";")
print(f"✔ Alle beregnede data gemt som {full_path}")


# Beregn PMV, PPD og SET

def beregn_termiske_parametre(row):
    try:
        tdb = float(row["t"])
        tr = float(row["t"])
        rh = float(row["h"])
        clo = float(row["clo"])
        met = 1.8

        pmv_res = pmv_ppd_iso(tdb=tdb, tr=tr, vr=0.1, rh=rh, met=met, clo=clo)
        set_res = set_tmp(tdb=tdb, tr=tr, v=0.1, rh=rh, met=met, clo=clo)

        return pd.Series([pmv_res["pmv"], pmv_res["ppd"], set_res["set"]])
    except Exception as e:
        print(f"Fejl ved termisk beregning:\n{row}\n{e}")
        return pd.Series([np.nan, np.nan, np.nan])


# Kør beregningen
merged_df[["pmv", "ppd", "set"]] = merged_df.apply(beregn_termiske_parametre, axis=1)

# 🎯 Mapping af spørgeskemasvar til labels
label_df = split_rows[split_rows["variable"] == "temp_feeling"].copy()
label_df["værdi"] = pd.to_numeric(label_df["værdi"], errors="coerce")
labels_dict = label_df.set_index("værdi")["label"].to_dict()

merged_df["temp_feeling_label"] = merged_df["temp_feeling"].map(
    labels_dict).fillna(merged_df["temp_feeling"]).astype(str)

ordered_labels = [str(labels_dict[i]) if i in labels_dict else str(i)
                  for i in sorted(merged_df["temp_feeling"].dropna().unique())]

merged_df["temp_feeling_label"] = pd.Categorical(
    merged_df["temp_feeling_label"],
    categories=ordered_labels,
    ordered=True
)

# 📊 Plot SET vs temp_feeling
plt.figure(figsize=(10, 5))
sns.violinplot(
    data=merged_df,
    x="temp_feeling_label",
    y="set",
    inner="box",
    density_norm="count"
)
plt.xlabel("Temp feeling (spørgeskema)")
plt.ylabel("SET [°C]")
plt.title("Standard Effective Temperature (SET) vs. Temp Feeling")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=30, ha="right")

# ➕ Tilføj komfortområde (24–26 °C)
plt.axhline(24, linestyle="--", color="green", alpha=0.5)
plt.axhline(26, linestyle="--", color="green", alpha=0.5)
plt.text(len(ordered_labels)-0.5, 24.1, "Komfortområde",
         color="green", ha="right", fontsize=8)

plt.tight_layout()
plt.show()

# Gem og vis
plt.savefig(os.path.join(plotsdone_dir, "SET_vs_temp_feeling.png"))
plt.close()

print("✔ SET-plot gemt!")


lokale_navn_ændringer = {
    "Emdrup": "AU E",
    "Århus 105": "AU 105",
    "Århus 113": "AU 113",
    "Ålborg fib10": "AAU fib",
    "Ålborg krogh": "AAU krogh",
    "Ålborg selma 300": "AAU selma"
}

merged_df["belægning_favn"] = merged_df["belægning_favn"].replace(lokale_navn_ændringer)


lokale_order = [
    "KEA",
    "KU A11", "KU A81", "KU HCØ",
    "DTU A34", "DTU 42", "DTU 49", "DTU 81", "DTU 83",
    "CBS 033", "CBS 202",
    "RUC A01", "RUC A25",
    "AU E", "AU 105", "AU 113",
    "AAU fib", "AAU krogh", "AAU selma"
]


# 📊 Tabel over SET per lokale
set_summary = merged_df.groupby("belægning_favn").agg(
    antal_målinger=("set", "count"),
    gennemsnitlig_SET=("set", "mean"),
    median_SET=("set", "median"),
    std_SET=("set", "std"),
    min_SET=("set", "min"),
    max_SET=("set", "max")
).reset_index()

# Sortér evt. efter gennemsnitlig SET
set_summary = set_summary.sort_values("gennemsnitlig_SET")

# Vis i konsol
print("\n📝 SET-værdier per lokale:")
print(set_summary.to_string(index=False))


# ➕ Fjern NaN og forbered data
plotdata = merged_df.dropna(subset=["set", "belægning_favn"]).copy()
plotdata["belægning_favn"] = plotdata["belægning_favn"].astype(str)

# 🎻 Plot
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=plotdata,
    x="belægning_favn",
    y="set",
    order=lokale_order,
    inner="quartile",
    density_norm="width"
)
plt.xlabel("Room")
plt.ylabel("SET [°C]")
plt.title("Standard Effective Temperature (SET) per room")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha="right")
plt.axhspan(22, 26, color='blue', alpha=0.2, zorder=0)
plt.tight_layout()
plt.show()


plotdata = merged_df.dropna(subset=["pmv", "belægning_favn"]).copy()


plotdata = merged_df.dropna(subset=["pmv", "belægning_favn"]).copy()

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=plotdata,
    x="belægning_favn",
    y="pmv",
    order=lokale_order,
    palette="Set2",
    whis=[2.5, 97.5],
    showfliers=False
)
plt.xlabel("Room")
plt.ylabel("PMV")
plt.title("Boxplot: PMV per Room")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha="right")

# Komfortområde
plt.axhspan(-0.2, 0.2, color='blue', alpha=0.2, zorder=0)
plt.axhspan(-0.5, 0.5, color='blue', alpha=0.2, zorder=0)
plt.axhspan(-0.7, 0.7, color='blue', alpha=0.2, zorder=0)

plt.tight_layout()

plt.show()



plotdata = merged_df.dropna(subset=["ppd", "belægning_favn"]).copy()

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=plotdata,
    x="belægning_favn",
    y="ppd",
    order=lokale_order,
    palette="Set2",
    whis=[2.5, 97.5],
    showfliers=False  # Vis evt. outliers
)
plt.xlabel("Room")
plt.ylabel("PPD [%]")
plt.ylim(0, 50)

plt.title("Boxplot: PPD % per Room ")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha="right")

# Komfortgrænse ved 10 %

plt.axhspan(0, 6, color='blue', alpha=0.2, zorder=0)
plt.axhspan(0, 10, color='blue', alpha=0.2, zorder=0)
plt.axhspan(0, 15, color='blue', alpha=0.2, zorder=0)


plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
