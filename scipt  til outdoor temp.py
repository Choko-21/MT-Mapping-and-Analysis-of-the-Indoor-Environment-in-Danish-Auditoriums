import pandas as pd
import os
import glob

# === 1. Opsætning ===
data_dir = r"C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo ved påske/Ude_Temp"
output_dir = r"C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data"

if not os.path.exists(data_dir):
    raise ValueError(f"Stien findes ikke: {data_dir}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Find alle relevante filtyper
file_paths = glob.glob(os.path.join(data_dir, "*.xlsx"))
file_paths += glob.glob(os.path.join(data_dir, "*.csv"))
file_paths += glob.glob(os.path.join(data_dir, "*.CSV"))

df_list = []

# === 2. Indlæs og forbered hver fil ===
for path in file_paths:
    filename = os.path.basename(path)
    location = ''.join(filter(str.isalpha, filename.split("_")[0]))

    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path, sep=";", decimal=",", usecols=[
                             "DateTime", "Middel"], parse_dates=["DateTime"])
        else:
            df = pd.read_excel(
                path, usecols=["DateTime", "Middel"], parse_dates=["DateTime"])
        df["Lokation"] = location
        df_list.append(df)
    except Exception as e:
        print(f"Fejl ved {filename}: {e}")

# === 3. Tjek at vi har data ===
if not df_list:
    raise ValueError("Ingen gyldige datafiler fundet.")

# === 4. Saml og interpolér ===
combined_df = pd.concat(df_list, ignore_index=True)
combined_df["DateTime"] = pd.to_datetime(combined_df["DateTime"])
combined_df = combined_df.drop_duplicates(subset=["DateTime", "Lokation"])
combined_df = combined_df.set_index("DateTime")

resampled_list = []
for location, group in combined_df.groupby("Lokation"):
    group = group.sort_index()
    group_resampled = group.resample("5min").asfreq()
    group_resampled["Middel"] = group_resampled["Middel"].interpolate(
        method="linear", limit_area="inside")
    group_resampled["Middel"] = group_resampled["Middel"].round(2)
    group_resampled["Lokation"] = location
    resampled_list.append(group_resampled)

final_df = pd.concat(resampled_list).reset_index()

# === 5. Omdøb kolonner ===
final_df = final_df.rename(columns={"DateTime": "Tid", "Middel": "temp"})

# === 6. Gem som CSV med punktum som decimalseparator ===
final_df.to_csv(os.path.join(output_dir, "Outdoor_temp_data_interp.csv"),
                index=False, sep=";", decimal=".")

print("Fil gemt i:", output_dir)


# === 7. Beregn dagligt gennemsnit og maksimum ===
final_df["date"] = final_df["Tid"].dt.date

daily_stats = (
    final_df
    .groupby(["date", "Lokation"])
    .agg(
        mean_out_temp=("temp", "mean"),
        max_out_temp=("temp", "max")
    )
    .reset_index()
)

# Rund værdierne pænt af
daily_stats["mean_out_temp"] = daily_stats["mean_out_temp"].round(2)
daily_stats["max_out_temp"] = daily_stats["max_out_temp"].round(2)

# Gem som nyt sheet i en Excel-fil
output_path = os.path.join(output_dir, "Outdoor_temp_daily_stats.xlsx")
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    final_df.to_excel(writer, index=False, sheet_name="Interpolated Data")
    daily_stats.to_excel(writer, index=False, sheet_name="Daily Stats")

print("Filer gemt i:", output_dir)
