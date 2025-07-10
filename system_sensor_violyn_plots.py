import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd
import folium
from datetime import datetime
import pytz
from folium.plugins import MarkerCluster
import os


import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score






output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)


# === Safe datetime parser funktion ===
def parse_datetime_safe(series, fmt='%d/%m/%Y %H.%M', tz='Europe/Copenhagen'):
    dt_naive = pd.to_datetime(series, format=fmt, errors='coerce')
    dt_safe = dt_naive.dt.tz_localize(tz, nonexistent='shift_forward', ambiguous='NaT')
    return dt_safe


# === 1. Indlæs sensor-data (Atmo sensorer) ===

sensor_ids = range(21, 31)
base_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data/TAIL-"
file_suffix = ".csv"

sensor_frames = []
for sensor_id in sensor_ids:
    path = f"{base_path}{sensor_id}{file_suffix}"

    # Robust fil-check
    if not os.path.exists(path):
        print(f"FIL MANGLER: {path}")
        continue

    try:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
        df = df[pd.to_numeric(df['ts'], errors='coerce').notna()]
        df['ts'] = pd.to_datetime(df['ts'], unit='s', origin='unix', utc=True).dt.tz_convert('Europe/Copenhagen')
        df['sensor_id'] = f"TAIL-{sensor_id}"

        for col in df.columns:
            if col not in ['ts', 'sensor_id']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        sensor_frames.append(df)

    except Exception as e:
        print(f"Fejl ved indlæsning af TAIL-{sensor_id}: {e}")
        continue

# === Indlæs Airthings sensorer ===

def load_other_atmo_sensor(path, sensor_id):
    df_raw = pd.read_excel(path)
    df_raw['Ts'] = pd.to_datetime(df_raw['Ts'], dayfirst=True, errors='coerce')
    df = df_raw.groupby('Ts').agg('max').reset_index()
    df.rename(columns={'Ts': 'ts', 'VOC ppb': 'voc'}, inplace=True)
    df['sensor_id'] = sensor_id

    for col in ['co2', 'pm25', 'pm1', 'h', 'temp', 'voc', 'p']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['ts'] = df['ts'].dt.tz_localize('Europe/Copenhagen', nonexistent='shift_forward')
    return df

# Indlæs de to Airthings sensorer
other_sensor_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data/AirThings_1.xlsx"
other_df_1 = load_other_atmo_sensor(other_sensor_path, sensor_id="TAIL-Airthings1")

other_sensor_path_2 = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data/AirThings_2.xlsx"
other_df_2 = load_other_atmo_sensor(other_sensor_path_2, sensor_id="TAIL-Airthings2")

sensor_frames.append(other_df_1)
sensor_frames.append(other_df_2)

# Kombiner alle Atmo/Airthings sensorer
sensor_df = pd.concat(sensor_frames, ignore_index=True)


# === Indlæs System data (med multi-kolonne support) ===

system_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data/System data.xlsx"
system_data = pd.read_excel(system_path, sheet_name=None)

system_frames = []

for sheet_name, df in system_data.items():
    df = df.copy()
    try:
        room_name, param_raw = sheet_name.rsplit(' ', 1)
    except ValueError:
        
        continue

    df.columns = df.columns.str.strip().str.lower()

    if 'tid' not in df.columns:
        continue

    df.rename(columns={'tid': 'ts'}, inplace=True)
    df['ts'] = parse_datetime_safe(df['ts'])  # Brug safe datetime parser

    if param_raw.strip().lower() == 'co2':
        co2_cols = [col for col in df.columns if 'co2' in col]
        for col in co2_cols:
            df_sub = df[['ts', col]].dropna().copy()
            df_sub.rename(columns={col: 'value'}, inplace=True)
            df_sub['room'] = room_name.strip()
            df_sub['parameter'] = 'CO₂ [ppm]'
            system_frames.append(df_sub[['ts', 'value', 'room', 'parameter']])
    
    elif param_raw.strip().lower() == 'temp':
        temp_cols = [col for col in df.columns if 'temp' in col]
        for col in temp_cols:
            df_sub = df[['ts', col]].dropna().copy()
            df_sub.rename(columns={col: 'value'}, inplace=True)
            df_sub['room'] = room_name.strip()
            df_sub['parameter'] = 'Temperature [°C]'
            system_frames.append(df_sub[['ts', 'value', 'room', 'parameter']])
    
    system_df = pd.concat(system_frames, ignore_index=True)


# === Indlæs udetemperatur data ===

ude_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data/Outdoor_temperature.xlsx"
ude_df = pd.read_excel(ude_path)
ude_df.rename(columns={'Tid': 'ts', 'temp': 't', 'Lokation': 'lokation'}, inplace=True)
ude_df['ts'] = parse_datetime_safe(ude_df['ts'])



# Kombiner alt
sensor_df = pd.concat(sensor_frames, ignore_index=True)

def calc_absolute_humidity(temp_c, rh_percent):
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))  # mættet damptryk [hPa]
    ah = (es * rh_percent * 2.1674) / (273.15 + temp_c)
    return ah

# Beregn absolut fugtighed, hvis både 't' og 'h' er tilgængelig
sensor_df['absolute_h'] = calc_absolute_humidity(sensor_df['t'], sensor_df['h'])


lokale_to_ude = {
    'KEA': 'KEA',
    'KU A11': 'KU',
    'KU A81': 'KU',
    'KU HCØ': 'KU',
    'DTU A34': 'DTU',
    'DTU 42': 'DTU',
    'DTU 49': 'DTU',
    'DTU 81': 'DTU',
    'DTU 83': 'DTU',
    'CBS 033': 'CBS',
    'CBS 202': 'CBS',
    'RUC A01': 'RUC',
    'RUC A25': 'RUC',
    'AU E': 'AU E',
    'AU 105': 'AU',
    'AU 113': 'AU',
    'AAU fib': 'AAU',
    'AAU krogh': 'AAU',
    'AAU selma': 'AAU'
}


# === 2. Definer lokationsopsætning ===
# Her kan du styre hvilke sensorer og tidsrum der skal bruges for hvert lokale
# Du kan nu angive flere tidsintervaller pr. lokale
lokaleopsætning = [
    {
        "room": "KEA",
        "coords": (55.694430, 12.551002),
        "sensors": [
            {"id": "TAIL-25", "placering": "siddeplads"},
            {"id": "TAIL-26", "placering": "siddeplads"},
            {"id": "TAIL-30", "placering": "siddeplads"},
            {"id": "TAIL-28", "placering": "væg"},
        ],
        "periods": [["2025-04-03 11:55", "2025-04-03 17:05"]],
        "periodsnat": [["2025-04-03 8:00", "2025-04-03 12:00"]]
    },
    {
        "room": "KU A11",
        "coords": (55.682293, 12.543871),
        "sensors": [
            {"id": "TAIL-23", "placering": "siddeplads"},
            {"id": "TAIL-25", "placering": "siddeplads"},
            {"id": "TAIL-27", "placering": "siddeplads"},
            {"id": "TAIL-21", "placering": "væg"},
        ],
        "periods": [
            ["2025-04-22 08:45", "2025-04-22 13:15"],
            ["2025-04-23 08:00", "2025-04-23 12:05"]],
        "periodsnat": [["2025-04-23 00:00", "2025-04-23 8:00"]]
    },
    {
        "room": "KU A81",
        "coords": (55.682241, 12.543050),
        "sensors": [
            {"id": "TAIL-28", "placering": "siddeplads"},
            {"id": "TAIL-29", "placering": "siddeplads"},
            {"id": "TAIL-30", "placering": "siddeplads"},
            {"id": "TAIL-22", "placering": "væg"},
            {"id": "TAIL-24", "placering": "væg"},
        ],
        "periods": [
            ["2025-04-22 07:45", "2025-04-22 15:00"],
            ["2025-04-23 08:55", "2025-04-23 14:00"]],
        "periodsnat": [["2025-04-23 00:00", "2025-04-23 8:00"]]
    },
    {
        "room": "KU HCØ",
        "coords": (55.701200, 12.561218),
        "sensors": [
            {"id": "TAIL-22", "placering": "siddeplads"},
            {"id": "TAIL-23", "placering": "siddeplads"},
            {"id": "TAIL-30", "placering": "siddeplads"},
            {"id": "TAIL-29", "placering": "væg"},
        ],
        "periods": [
            ["2025-05-14 09:00", "2025-05-14 12:00"],
            ["2025-05-14 13:00", "2025-05-14 15:00"],
            ["2025-05-15 09:00", "2025-05-15 12:05"]],
        "periodsnat": [["2025-05-15 00:00", "2025-05-15 8:00"]]
    },
    {
        "room": "DTU A34",
        "coords": (55.784965, 12.518372),
        "sensors": [
            {"id": "TAIL-23", "placering": "siddeplads"},
            {"id": "TAIL-24", "placering": "siddeplads"},
            {"id": "TAIL-25", "placering": "siddeplads"},
            {"id": "TAIL-29", "placering": "siddeplads"},
            {"id": "TAIL-30", "placering": "siddeplads"},
            {"id": "TAIL-Airthings2", "placering": "siddeplads"},
            {"id": "TAIL-21", "placering": "væg"},
            {"id": "TAIL-22", "placering": "væg"},
            {"id": "TAIL-Airthings1", "placering": "væg"},
        ],
        "periods": [
            ["2025-03-17 12:50", "2025-03-17 15:05"],
            ["2025-03-18 08:00", "2025-03-18 12:00"],
            ["2025-03-18 12:50", "2025-03-18 15:15"],
            ["2025-03-19 07:55", "2025-03-19 09:50"],
            ["2025-03-19 12:45", "2025-03-19 14:20"],
            ["2025-03-20 08:00", "2025-03-20 11:40"],
            ["2025-03-20 13:00", "2025-03-20 15:15"],
            ["2025-03-21 08:00", "2025-03-21 10:30"],
            ["2025-03-24 13:00", "2025-03-24 16:00"],
            ["2025-03-25 08:00", "2025-03-25 12:00"],
            ["2025-03-25 13:00", "2025-03-25 15:40"],
            ["2025-03-26 07:50", "2025-03-26 09:50"],
            ["2025-03-26 13:00", "2025-03-26 14:40"],
            ["2025-03-27 12:50", "2025-03-27 15:30"],
            ["2025-03-28 12:50", "2025-03-28 14:15"]],
        "periodsnat": [["2025-03-25 00:00", "2025-03-25 8:00"]]
    },
    {
        "room": "DTU 42",
        "coords": (55.785359, 12.520083),
        "sensors": [
            {"id": "TAIL-25", "placering": "væg"},
            {"id": "TAIL-26", "placering": "siddeplads"},
            {"id": "TAIL-30", "placering": "siddeplads"},
        ],
        "periods": [
            ["2025-03-31 08:00", "2025-03-31 12:00"],
            ["2025-03-31 13:00", "2025-03-31 15:00"],
            ["2025-04-01 08:00", "2025-04-01 12:00"],
            ["2025-04-01 13:00", "2025-04-01 15:00"],
            ["2025-04-02 10:00", "2025-04-02 14:30"]],
        "periodsnat": [["2025-04-02 00:00", "2025-04-02 8:00"]]
    },
    {
        "room": "DTU 49",
        "coords": (55.785077, 12.519597),
        "sensors": [
            {"id": "TAIL-22", "placering": "væg"},
            {"id": "TAIL-28", "placering": "siddeplads"},
        ],
        "periods": [
            ["2025-03-31 09:30", "2025-03-31 16:00"],
            ["2025-04-01 09:00", "2025-04-01 11:00"],
            ["2025-04-01 12:50", "2025-04-01 16:00"],
            ["2025-04-02 09:00", "2025-04-02 12:00"],
            ["2025-04-02 13:00", "2025-04-02 15:00"]],
        "periodsnat": [["2025-04-02 00:00", "2025-04-02 8:00"]]
    },
    {
        "room": "DTU 81",
        "coords": (55.789326, 12.525098),
        "sensors": [
            {"id": "TAIL-28", "placering": "siddeplads"},
            {"id": "TAIL-29", "placering": "siddeplads"},
            {"id": "TAIL-30", "placering": "siddeplads"},
        ],
        "periods": [["2025-04-25 13:00", "2025-04-25 15:00"]],
        "periodsnat": [["2025-04-25 00:00", "2025-04-25 8:00"]]
    },
    {
        "room": "DTU 83",
        "coords": (55.789560, 12.525295),
        "sensors": [
            {"id": "TAIL-23", "placering": "siddeplads"},
            {"id": "TAIL-24", "placering": "siddeplads"},
            {"id": "TAIL-25", "placering": "siddeplads"},
            {"id": "TAIL-21", "placering": "væg"},
        ],
        "periods": [
            ["2025-04-24 09:00", "2025-04-24 11:30"],
            ["2025-04-24 13:00", "2025-04-24 15:00"]],
        "periodsnat": [["2025-04-24 00:00", "2025-04-24 8:00"]]
    },
    {
        "room": "CBS 033",
        "coords": (55.683414, 12.515411),
        "sensors": [
            {"id": "TAIL-21", "placering": "siddeplads"},
            {"id": "TAIL-22", "placering": "siddeplads"},
            {"id": "TAIL-23", "placering": "siddeplads"},
            {"id": "TAIL-24", "placering": "siddeplads"},
            {"id": "TAIL-25", "placering": "væg"},
        ],
        "periods": [
            ["2025-04-10 09:00", "2025-04-10 17:15"],
            ["2025-04-11 12:30", "2025-04-11 15:15"]],
        "periodsnat": [["2025-04-11 00:00", "2025-04-11 8:00"]]
    },
    {
        "room": "CBS 202",
        "coords": (55.681765, 12.530562),
        "sensors": [
            {"id": "TAIL-27", "placering": "siddeplads"},
            {"id": "TAIL-28", "placering": "siddeplads"},
            {"id": "TAIL-29", "placering": "siddeplads"},
            {"id": "TAIL-30", "placering": "siddeplads"},
            {"id": "TAIL-26", "placering": "væg"},
        ],
        "periods": [
            ["2025-04-10 11:30", "2025-04-10 15:00"],
            ["2025-04-11 10:30", "2025-04-11 16:00"]],
        "periodsnat": [["2025-04-11 00:00", "2025-04-11 8:00"]]
    },
    {
        "room": "RUC A01",
        "coords": (55.653668, 12.138726),
        "sensors": [
            {"id": "TAIL-25", "placering": "siddeplads"},
            {"id": "TAIL-28", "placering": "siddeplads"},
            {"id": "TAIL-30", "placering": "siddeplads"},
            {"id": "TAIL-27", "placering": "væg"},
        ],
        "periods": [
            ["2025-04-07 08:15", "2025-04-07 14:15"],
            ["2025-04-08 08:15", "2025-04-08 12:00"],
            ["2025-04-09 10:45", "2025-04-09 11:45"]],
        "periodsnat": [["2025-04-09 00:00", "2025-04-09 8:00"]]
    },
    {
        "room": "RUC A25",
        "coords": (55.651999, 12.134783),
        "sensors": [
            {"id": "TAIL-22", "placering": "siddeplads"},
            {"id": "TAIL-26", "placering": "væg"},
        ],
        "periods": [
            ["2025-04-07 09:15", "2025-04-07 11:45"],
            ["2025-04-08 09:15", "2025-04-08 12:00"],
            ["2025-04-09 09:15", "2025-04-09 10:15"]],
        "periodsnat": [["2025-04-09 00:00", "2025-04-09 8:00"]]
    },
    {
        "room": "AU E",
        "coords": (55.720818, 12.542272),
        "sensors": [
            {"id": "TAIL-21", "placering": "væg"},
            {"id": "TAIL-22", "placering": "væg"},
            {"id": "TAIL-23", "placering": "siddeplads"},
            {"id": "TAIL-24", "placering": "siddeplads"},
            {"id": "TAIL-25", "placering": "siddeplads"},
        ],
        "periods": [
            ["2025-04-28 09:15", "2025-04-28 12:00"],
            ["2025-04-29 11:15", "2025-04-29 14:00"]],
        "periodsnat": [["2025-04-29 00:00", "2025-04-29 8:00"]]
    },
    {
        "room": "AU 105",
        "coords": (56.172820, 10.205703),
        "sensors": [
            {"id": "TAIL-27", "placering": "siddeplads"},
            {"id": "TAIL-28", "placering": "siddeplads"},
            {"id": "TAIL-29", "placering": "siddeplads"},
            {"id": "TAIL-30", "placering": "væg"},
        ],
        "periods": [
            ["2025-05-05 12:15", "2025-05-05 16:00"],
            ["2025-05-06 08:15", "2025-05-06 09:45"],
            ["2025-05-06 11:15", "2025-05-06 16:00"]],
        "periodsnat": [["2025-05-06 00:00", "2025-05-06 8:00"]]
    },
    {
        "room": "AU 113",
        "coords": (56.171828, 10.203418),
        "sensors": [
            {"id": "TAIL-23", "placering": "siddeplads"},
            {"id": "TAIL-24", "placering": "siddeplads"},
            {"id": "TAIL-21", "placering": "væg"},
            {"id": "TAIL-22", "placering": "væg"},
        ],
        "periods": [
            ["2025-05-05 10:15", "2025-05-05 15:50"],
            ["2025-05-06 14:15", "2025-05-06 18:00"]],
        "periodsnat": [["2025-05-06 00:00", "2025-05-06 8:00"]]
    },
    {
        "room": "AAU fib",
        "coords": (57.017614, 9.976512),
        "sensors": [
            {"id": "TAIL-22", "placering": "siddeplads"},
            {"id": "TAIL-23", "placering": "siddeplads"},
            {"id": "TAIL-21", "placering": "væg"},
        ],
        "periods": [
            ["2025-05-07 10:15", "2025-05-07 12:00"],
            ["2025-05-08 08:15", "2025-05-08 11:45"]],
        "periodsnat": [["2025-05-08 00:00", "2025-05-08 8:00"]]
    },
    {
        "room": "AAU krogh",
        "coords": (57.014187, 9.982140),
        "sensors": [
            {"id": "TAIL-29", "placering": "siddeplads"},
            {"id": "TAIL-30", "placering": "siddeplads"},
            {"id": "TAIL-28", "placering": "væg"},
            {"id": "TAIL-27", "placering": "væg"},
        ],
        "periods": [
            ["2025-05-07 09:00", "2025-05-07 12:00"],
            ["2025-05-07 13:00", "2025-05-07 16:00"],
            ["2025-05-08 08:15", "2025-05-08 11:15"]],
        "periodsnat": [["2025-05-08 00:00", "2025-05-08 8:00"]]
    },
    {
        "room": "AAU selma",
        "coords": (57.012099, 9.991720),
        "sensors": [
            {"id": "TAIL-24", "placering": "væg"},
            {"id": "TAIL-25", "placering": "siddeplads"},
            {"id": "TAIL-26", "placering": "væg"},
        ],
        "periods": [
            ["2025-05-07 10:00", "2025-05-07 11:15"],
            ["2025-05-07 14:15", "2025-05-07 16:30"],
            ["2025-05-08 12:15", "2025-05-08 14:00"]],
        "periodsnat": [["2025-05-08 00:00", "2025-05-08 8:00"]]
    }
]



# === 3. Beregn statistik per lokale ===
avg_data = []
for entry in lokaleopsætning:
    room = entry['room']
    lat, lon = entry['coords']
    sensors = entry['sensors']
    periods = entry['periods']

    df_total = pd.DataFrame()
    for start_str, end_str in periods:
        start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
        end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')

        sensor_ids = [s['id'] for s in sensors]
        df_sub = sensor_df[(sensor_df['sensor_id'].isin(sensor_ids)) &
                           (sensor_df['ts'] >= start) & (sensor_df['ts'] <= end)]
        df_total = pd.concat([df_total, df_sub])

    if not df_total.empty:
        co2 = df_total['co2'].mean()
        temp = df_total['t'].mean()
        noise = df_total['noise'].mean()
        light = df_total['light'].mean()
    else:
        co2 = temp = noise = light = None

    avg_data.append({
        'room': room,
        'lat': lat,
        'lon': lon,
        'co2': co2,
        't': temp,
        'noise': noise,
        'light': light
    })

avg_df = pd.DataFrame(avg_data)


# Udtræk dato og beregn dagligt gennemsnit
sensor_df['date'] = sensor_df['ts'].dt.date

# Ny liste til daglig statistik
daily_stats = []

for entry in lokaleopsætning:
    room = entry['room']
    lat, lon = entry['coords']
    sensors = entry['sensors']
    periods = entry['periods']

    df_total = pd.DataFrame()
    for start_str, end_str in periods:
        start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
        end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')
        sensor_ids = [s['id'] for s in sensors]
        df_sub = sensor_df[(sensor_df['sensor_id'].isin(sensor_ids)) &
                           (sensor_df['ts'] >= start) & (sensor_df['ts'] <= end)]
        df_sub = df_sub.copy()
        df_sub['room'] = room
        df_sub['lat'] = lat
        df_sub['lon'] = lon
        df_total = pd.concat([df_total, df_sub])

    if not df_total.empty:
        grouped = df_total.groupby(['room', 'date']).agg(
            co2=('co2', 'mean'),
            t=('t', 'mean'),
            noise=('noise', 'mean'),
            light=('light', 'mean'),
            lat=('lat', 'first'),
            lon=('lon', 'first')
        ).reset_index()
        daily_stats.append(grouped)

# Saml alle daglige målinger
daily_df = pd.concat(daily_stats, ignore_index=True)


# === 6A. Beregn statistik og kort per universitet ===
uni_coords = {
    "KU": (55.688578, 12.549380),
    "DTU": (55.786857, 12.521689),
    "KEA": (55.694430, 12.551002),
    "CBS": (55.682590, 12.522987),
    "RUC": (55.652834, 12.136755),
    "AU E": (55.720818, 12.542272),
    "AU": (56.172324, 10.204561),
    "AAU": (57.014633, 9.983457)
}


# === 6B. Generér statistiktabel per lokale og universitet ===

stat_rows = []
limits = {
    'co2': 1000,
    't': (20, 24),
    'noise': 55,
    'light': 300
}

for entry in lokaleopsætning:
    room = entry['room']
    sensors = entry['sensors']
    periods = entry['periods']
    df_total = pd.DataFrame()
    for start_str, end_str in periods:
        start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
        end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')
        sensor_ids = [s['id'] for s in sensors]
        df_sub = sensor_df[(sensor_df['sensor_id'].isin(sensor_ids)) &
                           (sensor_df['ts'] >= start) & (sensor_df['ts'] <= end)]
        df_total = pd.concat([df_total, df_sub])

    for param in ['co2', 't', 'noise', 'light']:
        vals = df_total[param].dropna()
        if vals.empty:
            continue
        mean = vals.mean()
        std = vals.std()
        min_val = vals.min()
        max_val = vals.max()
        ci_low, ci_high = norm.interval(0.95, loc=mean, scale=std)
        if isinstance(limits[param], tuple):
            outliers = ((vals < limits[param][0]) | (vals > limits[param][1])).sum()
        else:
            outliers = (vals > limits[param]).sum()

        stat_rows.append({
            'room': room,
            'parameter': param,
            'n': len(vals),
            'mean': round(mean, 2),
            'std': round(std, 2),
            'min': round(min_val, 2),
            'max': round(max_val, 2),
            'range': round(max_val - min_val, 2),
            '95% CI lower': round(ci_low, 2),
            '95% CI upper': round(ci_high, 2),
            'outliers': int(outliers)
        })

room_stats_df = pd.DataFrame(stat_rows)
room_stats_df.to_csv(os.path.join(output_dir, "statistik_per_lokale.csv"), index=False)

# === 7. Boksplot med outliers ===

room_stats_df = pd.read_csv(os.path.join(output_dir, "statistik_per_lokale.csv"))

for param in ['co2', 't', 'noise', 'light']:
    # Udtræk alle målinger for bokspottet – ikke kun gennemsnit
    box_data = []
    for entry in lokaleopsætning:
        room = entry['room']
        sensors = entry['sensors']
        periods = entry['periods']
        df_total = pd.DataFrame()
        for start_str, end_str in periods:
            start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
            end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')
            sensor_ids = [s['id'] for s in sensors]
        df_sub = sensor_df[(sensor_df['sensor_id'].isin(sensor_ids)) &
                           (sensor_df['ts'] >= start) & (sensor_df['ts'] <= end)]
        df_total = pd.concat([df_total, df_sub])

        if param in df_total.columns:
            for val in df_total[param].dropna():
                box_data.append({'room': room, 'value': val})

    box_df = pd.DataFrame(box_data)

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=box_df, x="room", y="value", palette="Set2", showfliers=True)
    plt.title(f"Boksplot af {param.upper()} pr. lokale")
    plt.ylabel(param.upper())
    plt.xlabel("Room")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boksplot_{param}_lokaler.png"), dpi=300)
    plt.show()

# === Statistik pr. lokale og sensor ===
sensor_stats = []

for entry in lokaleopsætning:
    room = entry['room']
    sensors = entry['sensors']
    periods = entry['periods']

    for sensor_dict in sensors:
        sensor_id = sensor_dict['id']
        df_total = pd.DataFrame()
        for start_str, end_str in periods:
            start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
            end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')
            df_sub = sensor_df[(sensor_df['sensor_id'] == sensor_id) &
                               (sensor_df['ts'] >= start) & (sensor_df['ts'] <= end)]
            df_total = pd.concat([df_total, df_sub])

        for param in ['co2', 't', 'noise', 'light']:
            vals = df_total[param].dropna()
            if vals.empty:
                continue
            mean = vals.mean()
            std = vals.std()
            min_val = vals.min()
            max_val = vals.max()
            ci_low, ci_high = norm.interval(0.95, loc=mean, scale=std)
            if isinstance(limits[param], tuple):
                outliers = ((vals < limits[param][0]) | (vals > limits[param][1])).sum()
            else:
                outliers = (vals > limits[param]).sum()

            sensor_stats.append({
                'room': room,
                'sensor': sensor_id.replace('TAIL-', 'Tail-'),
                'parameter': param,
                'n': len(vals),
                'mean': round(mean, 2),
                'std': round(std, 2),
                'min': round(min_val, 2),
                'max': round(max_val, 2),
                'range': round(max_val - min_val, 2),
                '95% CI low': round(ci_low, 2),
                '95% CI upr': round(ci_high, 2),
                'outliers': int(outliers)
            })

sensor_stats_df = pd.DataFrame(sensor_stats)
sensor_stats_df.to_csv(os.path.join(
    output_dir, "statistik_per_sensor_per_lokale.csv"), index=False)

# === Statistik pr. universitet baseret på al rådata ===
universitet_data = {}


# Beregn statistik ud fra al data for hvert universitet
uni_stat_rows = []

for uni, df in universitet_data.items():
    for param in ['co2', 't', 'noise', 'light']:
        vals = df[param].dropna()
        if vals.empty:
            continue
        mean = vals.mean()
        std = vals.std()
        min_val = vals.min()
        max_val = vals.max()
        ci_low, ci_high = norm.interval(0.95, loc=mean, scale=std)
        if isinstance(limits[param], tuple):
            outliers = ((vals < limits[param][0]) | (vals > limits[param][1])).sum()
        else:
            outliers = (vals > limits[param]).sum()

        uni_stat_rows.append({
            'university': uni,
            'parameter': param,
            'lokaler': avg_df[avg_df['university'] == uni]['room'].nunique(),
            'n': len(vals),
            'mean': round(mean, 2),
            'std': round(std, 2),
            'min': round(min_val, 2),
            'max': round(max_val, 2),
            'range': round(max_val - min_val, 2),
            '95% CI low': round(ci_low, 2),
            '95% CI upr': round(ci_high, 2),
            'outliers': int(outliers)
        })

uni_all_data_df = pd.DataFrame(uni_stat_rows)
uni_all_data_df.to_csv(os.path.join(
    output_dir, "statistik_per_universitet.csv"), index=False)

# === 8. Statistik pr. sensor pr. dag pr. lokale ===
daily_sensor_stats = []
day_period_entries = []

for entry in lokaleopsætning:
    room = entry['room']
    sensors = entry['sensors']
    periods = entry['periods']
    university = entry['university'] if 'university' in entry else ''

    for sensor_id in sensors:
        for start_str, end_str in periods:
            start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
            end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')
            df_sub = sensor_df[(sensor_df['sensor_id'] == sensor_id) &
                               (sensor_df['ts'] >= start) & (sensor_df['ts'] <= end)].copy()

            if df_sub.empty:
                continue

            df_sub['date'] = df_sub['ts'].dt.date

            grouped = df_sub.groupby('date')

            for date, group in grouped:
                for param in ['co2', 't', 'noise', 'light']:
                    vals = group[param].dropna()
                    if vals.empty:
                        continue
                    mean = vals.mean()
                    std = vals.std()
                    min_val = vals.min()
                    max_val = vals.max()
                    ci_low, ci_high = norm.interval(0.95, loc=mean, scale=std)
                    if isinstance(limits[param], tuple):
                        outliers = ((vals < limits[param][0]) |
                                    (vals > limits[param][1])).sum()
                    else:
                        outliers = (vals > limits[param]).sum()

                    daily_sensor_stats.append({
                        'room': room,
                        'date': date.strftime('%Y-%m-%d'),
                        'sensor_id': sensor_id,
                        'parameter': param,
                        'n': len(vals),
                        'mean': round(mean, 2),
                        'std': round(std, 2),
                        'min': round(min_val, 2),
                        'max': round(max_val, 2),
                        'range': round(max_val - min_val, 2),
                        '95% CI lower': round(ci_low, 2),
                        '95% CI upper': round(ci_high, 2),
                        'outliers': int(outliers)
                    })

            day_period_entries.append({
                'room': room,
                'date': start.strftime('%Y-%m-%d'),
                'start': start.strftime('%H:%M'),
                'end': end.strftime('%H:%M')
            })

sensor_day_df = pd.DataFrame(daily_sensor_stats)
sensor_day_df.to_csv(os.path.join(
    output_dir, "statistik_per_sensor_per_dag.csv"), index=False)

# === 9. Gem lokalets daglige perioder separat ===
period_df = pd.DataFrame(day_period_entries).drop_duplicates()
period_df.to_csv(os.path.join(output_dir, "lokale_dato_periode.csv"), index=False)






def normalize_parameter_name(param):
    if param in ['CO₂', 'CO2', 'co2']:
        return 'CO2'
    elif param in ['t', 'temp', 'Temperatur']:
        return 'Temperatur'
    elif param in ['noise', 'Lyd']:
        return 'Lyd'
    elif param in ['light', 'Lys']:
        return 'Lys'
    else:
        return param





# === 10. Split violinplot: Væg vs. siddeplads ===
violin_data = []


# Først Atmo-sensorer
for entry in lokaleopsætning:
    room = entry['room']
    periods = entry['periods']
    for sensor in entry['sensors']:
        sensor_id = sensor['id']
        placering = sensor['placering']
        df_total = pd.DataFrame()
        for start_str, end_str in periods:
            start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
            end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')
            df_sub = sensor_df[(sensor_df['sensor_id'] == sensor_id) &
                                (sensor_df['ts'] >= start) & (sensor_df['ts'] <= end)]
            df_total = pd.concat([df_total, df_sub])

        for _, row in df_total.iterrows():
            if pd.notna(row.get('t')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'Temperature [°C]', 'value': row['t'] })
            if pd.notna(row.get('co2')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'CO₂ [ppm]', 'value': row['co2'] })
            if pd.notna(row.get('noise')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'Sound [dB]', 'value': row['noise'] })
            if pd.notna(row.get('light')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'Light [lux]', 'value': row['light'] })
            if pd.notna(row.get('h')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'Relative Humidity [%]', 'value': row['h'] })
            if pd.notna(row.get('absolute_h')):
                violin_data.append({ 'room': room,'placering': placering,'parameter': 'Absolute Humidity [g/m³]','value': row['absolute_h'] })     
            if pd.notna(row.get('pm1')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'PM1 [µg/m³]', 'value': row['pm1'] })
            if pd.notna(row.get('pm25')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'PM2.5 [µg/m³]', 'value': row['pm25'] })
            if pd.notna(row.get('pm4')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'PM4 [µg/m³]', 'value': row['pm4'] })
            if pd.notna(row.get('pm10')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'PM10 [µg/m³]', 'value': row['pm10'] })
            if pd.notna(row.get('p')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'Pressure [hPa]', 'value': row['p'] })
            if pd.notna(row.get('voc_index')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'TVOC Index [-]', 'value': row['voc_index'] })
            if pd.notna(row.get('nox_index')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'NOX Index [-]', 'value': row['nox_index'] })
            if pd.notna(row.get('ch2o')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'CH₂O [ppm]', 'value': row['ch2o'] })
            if pd.notna(row.get('voc')):
                violin_data.append({'room': room, 'placering': placering, 'parameter': 'TVOC [ppm]', 'value': row['voc'] })
            


violin_data_system = []
# Dernæst system-data ind i væg (kun hvis flag aktiveret)
for entry in lokaleopsætning:
    room = entry['room']
    periods = entry['periods']
    df_room_system = system_df[system_df['room'] == room]

    for start_str, end_str in periods:
        start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
        end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')
        df_period = df_room_system[(df_room_system['ts'] >= start) & (df_room_system['ts'] <= end)]

        for _, row in df_period.iterrows():
            if row['parameter'] == 'Temperature [°C]' and pd.notna(row['value']):
                violin_data_system.append({
                    'room': room,
                    'placering': 'væg',
                    'parameter': 'Temperature [°C]',
                    'value': row['value'],
                    'source': 'system'
                })
            if row['parameter'] == 'CO₂ [ppm]' and pd.notna(row['value']):
                violin_data_system.append({
                    'room': room,
                    'placering': 'væg',
                    'parameter': 'CO₂ [ppm]',
                    'value': row['value'],
                    'source': 'system'
                })


# === Byg dataframe ===
violin_df = pd.DataFrame(violin_data)
df_system = pd.DataFrame(violin_data_system)

# Oversæt placering
violin_df['Placement'] = violin_df['placering'].replace({
    'siddeplads': 'Seating',
    'væg': 'Wall'
})



# Ensure split violin plot does not crash due to missing hue levels
split_ready = []
for param in violin_df['parameter'].unique():
    sub_df = violin_df[violin_df['parameter'] == param]
    for room in sub_df['room'].unique():
        room_data = sub_df[sub_df['room'] == room]
        placements = room_data['Placement'].unique()
        if 'Seating' not in placements:
            split_ready.append({'room': room, 'Placement': 'Seating', 'parameter': param, 'value': np.nan})
        if 'Wall' not in placements:
            split_ready.append({'room': room, 'Placement': 'Wall', 'parameter': param, 'value': np.nan})

if split_ready:
    violin_df = pd.concat([violin_df, pd.DataFrame(split_ready)], ignore_index=True)

# === Plot ===
for param in violin_df['parameter'].unique():
    plt.figure(figsize=(14, 6))
    sns.violinplot(
        data=violin_df[violin_df['parameter'] == param],
        x="room", y="value", hue="Placement", palette="Set2",
        inner="quartile", split=True 
    )
    
    # Optional: add comfort temperature bands
    if param.startswith("Temp"):
        plt.axhspan(19, 25, color='blue', alpha=0.2, zorder=0)
        plt.axhspan(20, 24, color='blue', alpha=0.2, zorder=0)
        plt.axhspan(21, 23, color='blue', alpha=0.2, zorder=0)
    
    # Force fixed y-axis for illuminance
    if param.startswith("Light"):
        plt.ylim(0, 750)  # Set fixed range for illuminance
        plt.axhspan(300, 500, color='blue', alpha=0.2, zorder=0)
    
    if param.startswith("CO"):
        plt.axhspan(250, 950, color='blue', alpha=0.2, zorder=0)
        plt.axhspan(250, 1200, color='blue', alpha=0.2, zorder=0)
        plt.axhspan(250, 1750, color='blue', alpha=0.2, zorder=0)
        plt.ylim(300, 1800)
        
    if param.startswith("Relative Humidity"):
        plt.axhspan(30, 50, color='blue', alpha=0.2, zorder=0)
        plt.axhspan(25, 60, color='blue', alpha=0.2, zorder=0)
        plt.axhspan(20, 70, color='blue', alpha=0.2, zorder=0)
    if param.startswith("Absolute"):
        plt.axhspan(6, 12, color='blue', alpha=0.2, zorder=0)
    if param == "TVOC [ppm]":
        plt.axhspan(0, 0.2, color='blue', alpha=0.2, zorder=0)
        plt.axhspan(0, 0.6, color='blue', alpha=0.2, zorder=0)
        plt.ylim(0, 1)
    if param.startswith("PM"):
        plt.ylim(0, 15)
    
    if param.startswith("CH"):
        plt.axhline(0.08, linestyle="--", color="red", linewidth=1.5, alpha=0.7, label="WHO guideline")
        plt.legend(loc="upper right")
    
        
    
    plt.title(f"Split Violin Plot of {param} by Room (Wall vs. Seating)")
    plt.ylabel(f"{param}")
    plt.xlabel("Room")
    plt.legend(loc='upper right') 
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


    







def lav_split_violin_data(param_navn, param_kolonne, source_label_sensor, placering_sensor):
    """
    Hjælpefunktion der genererer split-data til violinplot
    """
    data = []

    for entry in lokaleopsætning:
        room = entry['room']
        periods = entry['periods']

        sensor_ids = [s['id'] for s in entry['sensors'] if s['placering'] == placering_sensor]
        df_sensor = pd.DataFrame()

        for start_str, end_str in periods:
            start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
            end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')
            df_sub = sensor_df[
                (sensor_df['sensor_id'].isin(sensor_ids)) &
                (sensor_df['ts'] >= start) & (sensor_df['ts'] <= end)
            ]
            df_sensor = pd.concat([df_sensor, df_sub], ignore_index=True)

        for _, row in df_sensor.iterrows():
            if pd.notna(row.get(param_kolonne)):
                data.append({'room': room, 'Source': source_label_sensor, 'value': row[param_kolonne]})

        df_sys = df_system[
            (df_system['room'] == room) &
            (df_system['parameter'] == param_navn)
        ]
        for _, row in df_sys.iterrows():
            if pd.notna(row['value']):
                data.append({'room': room, 'Source': 'System', 'value': row['value']})

    df = pd.DataFrame(data)

    # Sikr split virker
    split_ready = []
    for room in df['room'].unique():
        room_data = df[df['room'] == room]
        levels = room_data['Source'].unique()
        if source_label_sensor not in levels:
            split_ready.append({'room': room, 'Source': source_label_sensor, 'value': np.nan})
        if 'System' not in levels:
            split_ready.append({'room': room, 'Source': 'System', 'value': np.nan})

    if split_ready:
        df = pd.concat([df, pd.DataFrame(split_ready)], ignore_index=True)

    return df


def plot_split_violin(df, parameter, ylabel, bands=None, ylim=None):
    """
    Plot split violin med valgfri komfortzoner
    """
    plt.figure(figsize=(18, 6))
    sns.violinplot(data=df, x='room', y='value', hue='Source',
                   palette='Set2', inner='quartile', split=True)

    if bands:
        for band in bands:
            plt.axhspan(*band, color='blue', alpha=0.2, zorder=0)
    if ylim:
        plt.ylim(*ylim)

    plt.title(f"Split Violin: {parameter} – {df['Source'].unique()[0]} vs. System")
    plt.ylabel(ylabel)
    plt.xlabel("Room")
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# === 1. CO₂ – væg vs system
df_co2_wall = lav_split_violin_data("CO₂ [ppm]", "co2", "Wall", "væg")
plot_split_violin(df_co2_wall, "CO₂", "CO₂ [ppm]", bands=[(200, 950), (200, 1200), (200, 1750)], ylim=(200, 1800))

# === 2. Temperatur – væg vs system
df_temp_wall = lav_split_violin_data("Temperature [°C]", "t", "Wall", "væg")
plot_split_violin(df_temp_wall, "Temperature", "Temperature [°C]", bands=[(19, 25), (20, 24), (21, 23)], ylim=None)

# === 3. CO₂ – siddeplads vs system
df_co2_seating = lav_split_violin_data("CO₂ [ppm]", "co2", "Seating", "siddeplads")
plot_split_violin(df_co2_seating, "CO₂", "CO₂ [ppm]", bands=[(200, 950), (200, 1200), (200, 1750)], ylim=(200, 1800))

# === 4. Temperatur – siddeplads vs system
df_temp_seating = lav_split_violin_data("Temperature [°C]", "t", "Seating", "siddeplads")
plot_split_violin(df_temp_seating, "Temperature", "Temperature [°C]", bands=[(19, 25), (20, 24), (21, 23)], ylim=None)








sanity_temp_udedoors = []

for entry in lokaleopsætning:
    room = entry['room']
    periods = entry['periods']

    # Find tilhørende ude-lokation
    ude_loc = lokale_to_ude.get(room, None)
    if ude_loc is None:
        continue

    # --- Wall sensor temperatur ---
    vaeg_sensor_ids = [s['id'] for s in entry['sensors'] if s['placering'] == 'væg']
    df_vaeg = pd.DataFrame()

    for start_str, end_str in periods:
        start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
        end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')
        df_sub = sensor_df[(sensor_df['sensor_id'].isin(vaeg_sensor_ids)) &
                            (sensor_df['ts'] >= start) & (sensor_df['ts'] <= end)]
        df_vaeg = pd.concat([df_vaeg, df_sub])

    for _, row in df_vaeg.iterrows():
        if pd.notna(row.get('t')):
            sanity_temp_udedoors.append({'room': room, 'Sourse': 'Wall sensor', 'value': row['t']})

    # --- Ude temperatur ---
    df_ude = ude_df[ude_df['lokation'] == ude_loc]

    for start_str, end_str in periods:
        start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
        end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')
        df_sub = df_ude[(df_ude['ts'] >= start) & (df_ude['ts'] <= end)]
        for _, row in df_sub.iterrows():
            if pd.notna(row['t']):
                sanity_temp_udedoors.append({'room': room, 'Sourse': 'Outside', 'value': row['t']})

sanity_temp_udedoors_df = pd.DataFrame(sanity_temp_udedoors)

# Sikr dummy-data så split kører
split_ready = []
for room in sanity_temp_udedoors_df['room'].unique():
    room_data = sanity_temp_udedoors_df[sanity_temp_udedoors_df['room'] == room]
    Sourse_levels = room_data['Sourse'].unique()
    if 'Wall sensor' not in Sourse_levels:
        split_ready.append({'room': room, 'Sourse': 'Wall sensor', 'value': np.nan})
    if 'Outside' not in Sourse_levels:
        split_ready.append({'room': room, 'Sourse': 'Outside', 'value': np.nan})

if split_ready:
    sanity_temp_udedoors_df = pd.concat([sanity_temp_udedoors_df, pd.DataFrame(split_ready)], ignore_index=True)

# Plot
plt.figure(figsize=(18, 6))
sns.violinplot(data=sanity_temp_udedoors_df, x='room', y='value', hue='Sourse',
               palette='Set2', inner="quartile", split=True)
plt.title("Split Violin: Temperature - Wall sensor vs. Outside")
plt.axhspan(19, 25, color='blue', alpha=0.2, zorder=0)
plt.axhspan(20, 24, color='blue', alpha=0.2, zorder=0)
plt.axhspan(21, 23, color='blue', alpha=0.2, zorder=0)
plt.ylabel("Temperature [°C]")
plt.xlabel("Room")
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
















#Lyd nivaue om natten 

# Filtrer til nattetimer og kun 'seating'-placeringer
night_data_seating = []

for entry in lokaleopsætning:
    room = entry['room']
    periods = entry['periodsnat']

    for sensor in entry['sensors']:
        sensor_id = sensor['id']
        placering = sensor['placering']

        if placering != 'siddeplads':
            continue  # kun seating

        df_total = pd.DataFrame()

        # Filtrér målinger inden for hver måleperiode
        for start_str, end_str in periods:
            start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
            end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')

            df_sub = sensor_df[
                (sensor_df['sensor_id'] == sensor_id) &
                (sensor_df['ts'] >= start) &
                (sensor_df['ts'] <= end)
            ]

            # Filtrér kun målinger mellem 07:00 og 08:00
            df_sub = df_sub[df_sub['ts'].dt.hour.between(7, 12)]

            df_total = pd.concat([df_total, df_sub], ignore_index=True)

        for _, row in df_total.iterrows():
            if pd.notna(row.get('noise')):
                night_data_seating.append({
                    'room': room,
                    'value': row['noise'],
                    'ts': row['ts'],
                    'sensor_id': sensor_id
                })

# Lav DataFrame
night_df = pd.DataFrame(night_data_seating)

# Sikrer at alle rum (med seating) optræder – også hvis de kun har 1 eller ingen målinger
seating_rooms = [entry['room'] for entry in lokaleopsætning if any(s['placering'] == 'siddeplads' for s in entry['sensors'])]

for room in seating_rooms:
    if room not in night_df['room'].unique():
        night_df = pd.concat([night_df, pd.DataFrame([{
            'room': room,
            'value': np.nan,
            'ts': pd.NaT,
            'sensor_id': None
        }])], ignore_index=True)



# Sanity check
if night_df.empty:
    print("⚠️ Ingen seating-lydmålinger i nattetimerne.")
else:
    # Plot uden split og hue
    plt.figure(figsize=(14, 6))
    sns.violinplot(data=night_df, x='room', y='value', palette='Set2', inner='box')
    plt.title("Violin Plot of Night-time Sound Levels (07:00–08:00, Seating only)")
    plt.ylabel("Sound level [dB(A)]")
    plt.xlabel("Room")
    plt.ylim(25, 60)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    











split_noise_data = []

for entry in lokaleopsætning:
    room = entry['room']
    sensors = entry['sensors']

    for sensor in sensors:
        sensor_id = sensor['id']
        placering = sensor['placering']

        # Gennemgå periods = med personer
        for start_str, end_str in entry.get('periods', []):
            start = pd.to_datetime(start_str).tz_localize("Europe/Copenhagen")
            end = pd.to_datetime(end_str).tz_localize("Europe/Copenhagen")

            df_sub = sensor_df[
                (sensor_df['sensor_id'] == sensor_id) &
                (sensor_df['ts'] >= start) &
                (sensor_df['ts'] <= end) &
                (sensor_df['noise'].notna())
            ]

            for _, row in df_sub.iterrows():
                split_noise_data.append({
                    'room': room,
                    'Placement': 'With people',
                    'value': row['noise']
                })

        # Gennemgå periodsnat = uden personer
        for start_str, end_str in entry.get('periodsnat', []):
            start = pd.to_datetime(start_str).tz_localize("Europe/Copenhagen")
            end = pd.to_datetime(end_str).tz_localize("Europe/Copenhagen")

            df_sub = sensor_df[
                (sensor_df['sensor_id'] == sensor_id) &
                (sensor_df['ts'] >= start) &
                (sensor_df['ts'] <= end) &
                (sensor_df['ts'].dt.hour.between(7, 12)) &
                (sensor_df['noise'].notna())
            ]

            for _, row in df_sub.iterrows():
                split_noise_data.append({
                    'room': room,
                    'Placement': 'Without people',
                    'value': row['noise']
                })

noise_split_df = pd.DataFrame(split_noise_data)

# Sikrer at begge hue-værdier er til stede for alle rum
split_ready = []
for room in noise_split_df['room'].unique():
    placements = noise_split_df[noise_split_df['room'] == room]['Placement'].unique()
    if 'With people' not in placements:
        split_ready.append({'room': room, 'Placement': 'With people', 'value': np.nan})
    if 'Without people' not in placements:
        split_ready.append({'room': room, 'Placement': 'Without people', 'value': np.nan})

if split_ready:
    noise_split_df = pd.concat([noise_split_df, pd.DataFrame(split_ready)], ignore_index=True)



plt.figure(figsize=(16, 6))
sns.violinplot(
    data=noise_split_df,
    x="room", y="value", hue="Placement",
    split=True, palette="Set2", inner="box"
)
plt.axhline(55, color='red', linestyle='--', label='Recommended Max (55 dB)')
plt.title("Split Violin Plot: Sound Levels With vs. Without People")
plt.ylabel("Sound level [dB(A)]")
plt.xlabel("Room")
plt.xticks(rotation=45, ha='right')
plt.ylim(25, 80)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.legend()
plt.show()





split_noise_data = []

for entry in lokaleopsætning:
    room = entry['room']
    sensors = entry['sensors']

    for sensor in sensors:
        sensor_id = sensor['id']

        # Saml "with people" data
        with_people_vals = []

        for start_str, end_str in entry.get('periods', []):
            start = pd.to_datetime(start_str).tz_localize("Europe/Copenhagen")
            end = pd.to_datetime(end_str).tz_localize("Europe/Copenhagen")

            df_sub = sensor_df[
                (sensor_df['sensor_id'] == sensor_id) &
                (sensor_df['ts'] >= start) & (sensor_df['ts'] <= end) &
                (sensor_df['noise'].notna())
            ]

            with_people_vals.extend(df_sub['noise'].tolist())

        if not with_people_vals:
            continue

        # Lav 10% fraktil
        q10 = np.percentile(with_people_vals, 10)

        for val in with_people_vals:
            split_noise_data.append({
                'room': room,
                'Placement': 'With people',
                'value': val
            })
            if val <= q10:
                split_noise_data.append({
                    'room': room,
                    'Placement': 'Lowest 10% fractile',
                    'value': val
                })

noise_split_df = pd.DataFrame(split_noise_data)


plt.figure(figsize=(16, 6))
sns.violinplot(
    data=noise_split_df,
    x="room", y="value", hue="Placement",
    split=True, palette="Set2", inner="quartile"
)
plt.axhline(55, color='red', linestyle='--', label='Recommended Max (55 dB)')
plt.axhline(30, color='red', linestyle='--', label='Recommended background (30 dB)')
plt.title("Sound levels: With people vs. lowest 10% fractile")
plt.ylabel("Sound level [dB(A)]")
plt.xlabel("Room")
plt.xticks(rotation=45, ha='right')
plt.ylim(25, 80)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

































cdf_data = []

for entry in lokaleopsætning:
    if entry['room'] == 'DTU A34':
        continue
    room = entry['room']
    sensors = [s['id'] for s in entry['sensors']]
    periods = entry['periods']

    df_total = pd.DataFrame()

    for start_str, end_str in periods:
        start = pd.to_datetime(start_str).tz_localize("Europe/Copenhagen")
        end = pd.to_datetime(end_str).tz_localize("Europe/Copenhagen")

        df_sub = sensor_df[
            (sensor_df['sensor_id'].isin(sensors)) &
            (sensor_df['ts'] >= start) &
            (sensor_df['ts'] <= end) &
            (sensor_df['t'].notna())
        ]

        df_total = pd.concat([df_total, df_sub])

    if df_total.empty:
        continue

    # Rund til minut og fjern dubletter (sensor_id + minut)
    df_total['minute'] = df_total['ts'].dt.floor('T')
    df_unique = df_total.drop_duplicates(subset=['sensor_id', 'minute'])

    # Sorter og kumulér
    temps = df_unique['t'].sort_values().reset_index(drop=True)
    cum_minutes = np.arange(1, len(temps) + 1)

    for temp, minutes in zip(temps, cum_minutes):
        cdf_data.append({'room': room, 'temperature': temp, 'minutes': minutes})

cdf_df = pd.DataFrame(cdf_data)


rooms = cdf_df['room'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(rooms)))

plt.figure(figsize=(14, 6))
for i, room in enumerate(rooms):
    df_room = cdf_df[cdf_df['room'] == room]
    plt.plot(df_room['minutes'], df_room['temperature'],
             label=room, color=colors[i], linewidth=2)

plt.xlabel("Akkumulerede minutter")
plt.ylabel("Temperatur [°C]")
plt.title("Kumulativ temperaturfordeling pr. lokale (baseret på måleperioder)")
plt.axhline(25, color='red', linestyle='--', label="25 °C grænse")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Lokale")
plt.tight_layout()
plt.show()






cdf_data = []

for entry in lokaleopsætning:
    room = entry['room']
    sensors = [s['id'] for s in entry['sensors']]
    periods = entry['periods']

    df_total = pd.DataFrame()

    for start_str, end_str in periods:
        start = pd.to_datetime(start_str).tz_localize("Europe/Copenhagen")
        end = pd.to_datetime(end_str).tz_localize("Europe/Copenhagen")

        df_sub = sensor_df[
            (sensor_df['sensor_id'].isin(sensors)) &
            (sensor_df['ts'] >= start) &
            (sensor_df['ts'] <= end) &
            (sensor_df['t'].notna())
        ]

        df_total = pd.concat([df_total, df_sub])

    if df_total.empty:
        continue

    # Rund til minut og fjern dubletter (sensor_id + minut)
    df_total['minute'] = df_total['ts'].dt.floor('T')
    df_unique = df_total.drop_duplicates(subset=['sensor_id', 'minute'])

    # Sorter og kumulér
    temps = df_unique[df_unique['t'] >= 25]['t'].sort_values().reset_index(drop=True)
    cum_minutes = np.arange(1, len(temps) + 1)

    for temp, minutes in zip(temps, cum_minutes):
        cdf_data.append({'room': room, 'temperature': temp, 'minutes': minutes})

cdf_df = pd.DataFrame(cdf_data)

plt.figure(figsize=(12, 6))
for room in cdf_df['room'].unique():
    df_room = cdf_df[cdf_df['room'] == room]
    plt.plot(df_room['minutes'], df_room['temperature'], label=room, linewidth=2)

plt.xlabel("Akkumulerede minutter over 25 °C")
plt.ylabel("Temperatur [°C]")
plt.title("Kumulativ fordeling af temperaturer > 25 °C (per lokale)")
plt.axhline(26, color='orange', linestyle='--', label="26 °C grænse (100 timer)")
plt.axhline(27, color='red', linestyle='--', label="27 °C grænse (25 timer)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Lokale")
plt.tight_layout()
plt.show()








cdf_data = []

for entry in lokaleopsætning:
    room = entry['room']
    sensors = [s['id'] for s in entry['sensors']]
    periods = entry['periods']

    df_total = pd.DataFrame()

    for start_str, end_str in periods:
        start = pd.to_datetime(start_str).tz_localize("Europe/Copenhagen")
        end = pd.to_datetime(end_str).tz_localize("Europe/Copenhagen")

        df_sub = sensor_df[
            (sensor_df['sensor_id'].isin(sensors)) &
            (sensor_df['ts'] >= start) &
            (sensor_df['ts'] <= end) &
            (sensor_df['t'].notna())
        ]

        df_total = pd.concat([df_total, df_sub])

    if df_total.empty:
        continue

    # Rund til minut og fjern dubletter (sensor_id + minut)
    df_total['minute'] = df_total['ts'].dt.floor('T')
    df_unique = df_total.drop_duplicates(subset=['sensor_id', 'minute'])

    # Sorter og kumulér
    temps = df_unique[df_unique['t'] >= 25]['t'].sort_values().reset_index(drop=True)
    cum_percent = (np.arange(1, len(temps) + 1) / len(temps)) * 100


    for temp, pct in zip(temps, cum_percent):
        cdf_data.append({'room': room, 'temperature': temp, 'percent': pct})

cdf_df = pd.DataFrame(cdf_data)

plt.figure(figsize=(12, 6))

for room in cdf_df['room'].unique():
    df_room = cdf_df[cdf_df['room'] == room]
    plt.plot(df_room['percent'], df_room['temperature'], label=room, linewidth=2)

plt.xlabel("Andel af måletid [%]")
plt.ylabel("Temperatur [°C]")
plt.title("Andel af tid med temperatur over 25 °C (pr. lokale)")
plt.axhline(26, color='orange', linestyle='--', label="26 °C grænse (100 timer)")
plt.axhline(27, color='red', linestyle='--', label="27 °C grænse (25 timer)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Lokale", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()







from tqdm import tqdm  # valgfrit for visning

scatter_data = []

for entry in tqdm(lokaleopsætning):
    room = entry['room']
    periods = entry['periods']

    # systemdata for dette rum og temperaturmålinger
    df_sys = system_df[
        (system_df['room'] == room) &
        (system_df['parameter'] == 'Temperature [°C]')
    ].copy()

    if df_sys.empty:
        continue

    df_sys['minute'] = df_sys['ts'].dt.floor('T')

    # Vægsensorer for dette lokale
    wall_sensors = [s['id'] for s in entry['sensors'] if s['placering'] == 'væg']
    df_wall = pd.DataFrame()

    for start_str, end_str in periods:
        start = pd.to_datetime(start_str).tz_localize('Europe/Copenhagen')
        end = pd.to_datetime(end_str).tz_localize('Europe/Copenhagen')

        df_sub = sensor_df[
            (sensor_df['sensor_id'].isin(wall_sensors)) &
            (sensor_df['ts'] >= start) &
            (sensor_df['ts'] <= end)
        ]

        df_wall = pd.concat([df_wall, df_sub])

    if df_wall.empty:
        continue

    df_wall['minute'] = df_wall['ts'].dt.floor('T')
    df_wall = df_wall.dropna(subset=['t'])

    # Gennemsnit per minut fra vægsensorer (hvis flere)
    wall_avg = df_wall.groupby('minute')['t'].mean().reset_index(name='wall_temp')

    # Systemdata per minut
    sys_avg = df_sys.dropna(subset=['value']).groupby('minute')['value'].mean().reset_index(name='system_temp')

    # Merge begge datasæt
    merged = pd.merge(wall_avg, sys_avg, on='minute', how='inner')
    merged['room'] = room

    scatter_data.append(merged)
    
scatter_df = pd.concat(scatter_data, ignore_index=True)

# Plot


plt.figure(figsize=(12, 6))
sns.scatterplot(data=scatter_df, x='wall_temp', y='system_temp', hue='room', palette='Set2', alpha=0.6)
plt.xlabel("Vægsensor temperatur [°C]")
plt.ylabel("System temperatur [°C]")
plt.title("System vs. Vægsensor temperatur (per lokale, i perioder med aktivitet)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Lokale', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




rooms = scatter_df['room'].unique()
n = len(rooms)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True, sharey=True)
axes = axes.flatten()

for i, room in enumerate(rooms):
    ax = axes[i]
    df_room = scatter_df[scatter_df['room'] == room]

    if len(df_room) < 2:
        ax.set_visible(False)
        continue

    # Regressionsplot
    sns.regplot(data=df_room, x='wall_temp', y='system_temp', ax=ax,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    # Tilføj x=y-linje
    lims = [17, 27]
    ax.plot(lims, lims, '--', color='grey', alpha=0.6, label='x = y')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Beregn R²
    X = df_room['wall_temp'].values.reshape(-1, 1)
    y = df_room['system_temp'].values
    model = LinearRegression().fit(X, y)
    r2 = r2_score(y, model.predict(X))

    ax.set_title(f"{room}\nR² = {r2:.2f}")
    ax.set_xlabel("Wall sensor [°C]")
    ax.set_ylabel("System [°C]")
    ax.grid(True, linestyle='--', alpha=0.8)
    if i % cols != 0:
        ax.set_ylabel("")
    if i < (rows - 1) * cols:
        ax.set_xlabel("")
    

# Fjern overskydende tomme plots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("System vs. wall temperature – with regression line and R²", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()








def plot_temp_share_with_broken_axis(summary_df):
    df = summary_df.set_index('room')[['% over 25°C', '% over 26°C', '% over 27°C']]
    colors = ['orange', 'red', 'darkred']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8),
    gridspec_kw={'height_ratios': [1, 3]})

    # Øvre (outlier zoom)
    df.plot(kind='bar', ax=ax1, color=colors, width=0.8)
    ax1.set_ylim(50, 100)
    ax1.set_yticks(np.arange(50, 105, 10))  # eller 2 hvis du vil have tættere
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(bottom=False, labelbottom=False)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(['% over 25°C', '% over 26°C', '% over 27°C'],title='Temperature threshold',loc='upper right')
    

    # Nedre (normal zoom)
    df.plot(kind='bar', ax=ax2, color=colors, width=0.8)
    ax2.get_legend().remove()
    ax2.set_ylim(0, 10)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel("Room")
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylabel("Percentage of measurement time [%]")



    # Broken axis streg-tegning
    d = .005  # snitstørrelse
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # venstre skrå streg
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # højre skrå streg

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    plt.suptitle("Share of time with temperatures exceeding 25 °C, 26 °C and 27 °C", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    
    plot_temp_share_with_broken_axis(summary_df)

    