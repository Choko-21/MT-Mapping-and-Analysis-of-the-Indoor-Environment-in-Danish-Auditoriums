import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# === 1 Filstier ===# 
# === Indstillinger for filstier ===
sensor_ids = range(21, 31)
base_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Kalibrerings data/DTU kalibering/Tail-"
file_suffix = ".xlsx"

# === Outputmappe til plots ===
output_dir = "output_kalibrerings_plots"
os.makedirs(output_dir, exist_ok=True)



# === Load reference sensor data ===
ref_df = pd.read_excel("C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Kalibrerings data/DTU kalibering/calibrerings forsøg data.xlsx")  # <- Udskift med din sti
ref_df.rename(columns={"Ts": "Timestamp", "temp": "ReferenceTemp"}, inplace=True)

# Konverter timestamp fra format dd/mm/yyyy HH.MM
ref_df['Timestamp'] = pd.to_datetime(ref_df['Timestamp'], format="%d/%m/%Y %H.%M", errors='coerce')

# (valgfrit) Lokaliser til samme tidszone
ref_df['Timestamp'] = ref_df['Timestamp'].dt.tz_localize('Europe/Copenhagen', nonexistent='shift_forward')



# === 2 Indlæs og klargøring af data ===
# === Indlæs og saml data ===
dataframes = []
for sid in sensor_ids:
    file_path = f"{base_path}{sid}{file_suffix}"
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        df['Sensor'] = f"Sensor-{sid}"
        dataframes.append(df)
    else:
        print(f"Fil ikke fundet: {file_path}")

combined_df = pd.concat(dataframes, ignore_index=True)

# === Konverter timestamp ===
combined_df['Timestamp'] = pd.to_datetime(combined_df['ts'], unit='s', errors='coerce')

# === 3 Usikkerheder ===
# === CO₂-målinger og usikkerhed ===
co2_col = [col for col in combined_df.columns if 'co2' in col.lower()][0]
combined_df['CO2'] = pd.to_numeric(combined_df[co2_col], errors='coerce')


def co2_uncertainty(co2_val):
    if pd.isna(co2_val):
        return np.nan
    return 75 if co2_val < 1001 else 40 + 0.05 * co2_val


# === Usikkerhedsopslag ===
uncertainty_lookup = {
    'CO2': lambda df: df['CO2'].apply(co2_uncertainty),
    'Temperature': lambda df: pd.Series(0.5, index=df.index),
    'Humidity': lambda df: pd.Series(2, index=df.index),
    'VOC': lambda df: df['VOC'] * 0.15,
    'Light': lambda df: df['Light'] * 0.10,
    'Noise': lambda df: pd.Series(2, index=df.index)
}

# === 4 Tidsserieplots ===
colors = plt.cm.tab20.colors  # Definer farver til plotting
parametre = {
    'co2': ('CO₂ [ppm]', 'CO2'),
    't': ('Temperature [°C]', 'Temperature'),
    'h': ('Relative humidity [%]', 'Humidity'),
    'voc': ('VOC [ppm]', 'VOC'),
    'light': ('Light [lux]', 'Light'),
    'noise': ('Noise [dB]', 'Noise')  # sidste parameter
}

for col_key, (ylabel, label) in parametre.items():
    if col_key in combined_df.columns:
        combined_df[label] = pd.to_numeric(combined_df[col_key], errors='coerce')
        if label == 'Humidity':
            combined_df[label] = combined_df[label] / 1  # Konverter til %
        if label == 'Temperature':
            combined_df[label] = combined_df[label] / 1     # Konverter til °C
        plot_df_param = combined_df[['Timestamp', label, 'Sensor']].dropna().copy()

        # Tilføj usikkerhed baseret på opslag
        if label in uncertainty_lookup:
            plot_df_param['Err'] = uncertainty_lookup[label](combined_df)
        else:
            plot_df_param['Err'] = np.nan

        # Tidsserieplot
        plt.figure(figsize=(14, 6))
        for i, (sensor, group) in enumerate(plot_df_param.groupby('Sensor')):
            color = colors[i % len(colors)]
            x = group['Timestamp'].values
            y = group[label].values
            yerr = group['Err'].values
            plt.plot(x, y, label=sensor, color=color, alpha=0.8)
            plt.fill_between(x, y - yerr, y + yerr, color='gold', alpha=0.2)
        
        if label == "Temperature":
            # Reference: plot as dashed black line
            ref_plot = ref_df.dropna(subset=['Timestamp', 'ReferenceTemp'])
            plt.plot(ref_plot['Timestamp'], ref_plot['ReferenceTemp'], linestyle='--',
             color='black', label='Reference sensor')    
        
        
        plt.title(f"{label} over time")
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label}_timeseries.png"), dpi=300)
        plt.show()

        # === 5 Violinplots ===
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=plot_df_param, x='Sensor', y=label, inner="box")
        plt.title(f"Distribution of {label} measurements per sensor")
        plt.ylabel(ylabel)
        plt.xlabel("Sensor")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label}_violinplot.png"), dpi=300)
        plt.show()

        # === 6 Konsistenskontrol ===
        # === 6.1 Max-min med usikkerhed ===
        plot_df_param['Timestamp_min'] = plot_df_param['Timestamp'].dt.floor('min')
        pivot = plot_df_param.pivot(index='Timestamp_min', columns='Sensor', values=label)
        max_series = pivot.max(axis=1)
        min_series = pivot.min(axis=1)
        err_series = plot_df_param.groupby('Timestamp_min')['Err'].mean()

        plt.figure(figsize=(14, 4))
        plt.plot(max_series.index, max_series, label='Max value', color='red')
        plt.plot(min_series.index, min_series, label='Min value', color='blue')
        plt.fill_between(max_series.index, max_series - err_series, max_series + err_series,
                         color='gold', alpha=0.2, label='Uncertainty on max')
        plt.fill_between(min_series.index, min_series - err_series, min_series + err_series,
                         color='gold', alpha=0.2, label='Uncertainty on min')

        plt.title(f'{label}: Max/Min with uncertainty')
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label}_minmax_with_uncertainty.png"), dpi=300)
        plt.show()

        pivot = plot_df_param.pivot(index='Timestamp', columns='Sensor', values=label)
        mean_series = pivot.mean(axis=1)
        deviations = pivot.subtract(mean_series, axis=0).abs()
        bounds = plot_df_param.groupby('Timestamp')['Err'].mean()
        outside_bounds = deviations.gt(bounds, axis=0)
        frac_outside = outside_bounds.sum(axis=1)

        plt.figure(figsize=(14, 4))
        plt.plot(frac_outside.index, frac_outside, color='red',
                 label='Measurements outside uncertainty')
        # plt.ylim(0, 1)  # Fjernet for at vise reel sensorantal
        plt.ylabel('Number of sensors')
        plt.xlabel('Time')
        plt.title(f'Consistency check: {label}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label}_consistency_check.png"), dpi=300)
        plt.show()
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=plot_df_param, x='Sensor', y=label, inner="box")
        plt.title(f"Distribution of {label} measurements per sensor")
        plt.ylabel(ylabel)
        plt.xlabel("Sensor")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label}_violinplot.png"), dpi=300)
        plt.show()
