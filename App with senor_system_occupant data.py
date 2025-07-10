import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, dash_table
import datetime
import os
import warnings

warnings.filterwarnings("ignore", message=".*to_pydatetime is deprecated.*")


import datetime

def filter_by_hours(df, time_col, hours):
    if df.empty:
        return df
    return df[(df[time_col].dt.time >= datetime.time(hours[0])) &
              (df[time_col].dt.time <= datetime.time(hours[1], 59))]



# === Indstillinger for filstier ===
sensor_ids = range(21, 31)
base_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data/TAIL-"
file_suffix = ".csv"
person_xlsx = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data/Antal personer done.xlsx"
system_data_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data/System data.xlsx"

# === Load alle sensorfiler ===
sensor_frames = []
for sensor_id in sensor_ids:
    path = f"{base_path}{sensor_id}{file_suffix}"
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
        df = df[pd.to_numeric(df['ts'], errors='coerce').notna()]
        df['ts'] = pd.to_datetime(pd.to_numeric(df['ts'], errors='coerce'), unit='s', errors='coerce', utc=True).dt.tz_convert('Europe/Copenhagen')
        df = df.dropna(subset=['ts'])
        df['sensor_id'] = f"TAIL-{sensor_id}"
        for col in df.columns:
            if col not in ['ts', 'sensor_id']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        sensor_frames.append(df)

def load_other_atmo_sensor(path, sensor_id):
    df_raw = pd.read_excel(path)
    df_raw['Ts'] = pd.to_datetime(df_raw['Ts'], dayfirst=True, errors='coerce')
    
    # Kombiner målingerne pr. tidspunkt (fjerner NaN via max())
    df = df_raw.groupby('Ts').agg('max').reset_index()

    # Omformater til samme format som sensor_df
    df.rename(columns={'Ts': 'ts', 'VOC ppb': 'voc'}, inplace=True)
    df['sensor_id'] = sensor_id

    # Sikrer numeriske værdier
    for col in ['co2', 'pm25', 'pm1', 'h', 'temp', 'voc', 'p']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Tilføj timezone (ligesom de andre sensorer har)
    df['ts'] = df['ts'].dt.tz_localize('Europe/Copenhagen')
    
    return df        
        
# === Indlæs også den nye sensor ===
other_sensor_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data/AirThings_1.xlsx"
other_df_1 = load_other_atmo_sensor(other_sensor_path, sensor_id="TAIL-Airthings1")
# === Indlæs anden ekstra sensor ===
other_sensor_path_2 = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/Atmo fuld data/AirThings_2.xlsx"
other_df_2 = load_other_atmo_sensor(other_sensor_path_2, sensor_id="TAIL-Airthings2")

sensor_frames.append(other_df_1)
sensor_frames.append(other_df_2)

# Kombiner alt
sensor_df = pd.concat(sensor_frames, ignore_index=True)


# === Load persondata ===
person_data_all = pd.read_excel(person_xlsx, sheet_name=None)
for sheet, df in person_data_all.items():
    df.columns = df.columns.str.strip().str.lower()
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce').dt.tz_localize('Europe/Copenhagen', ambiguous='NaT', nonexistent='shift_forward')

# === Hjælpefunktioner ===
def clean_with_limits(df, param):
    limits = {
        'co2': (300, 5000), 't': (10, 50), 'h': (10, 90), 'light': (0, 10000), 'voc_index': (0, 500),
        'noise': (20, 100), 'pm1': (0, 100), 'pm25': (0, 100), 'pm4': (0, 100), 'pm10': (0, 100),
        'ch2o': (0, 1), 'voc': (0, 50), 'nox_index': (0, 500), 'abs_h': (0, 80), 'p': (950, 1050)
    }
    if param not in limits:
        return df
    low, high = limits[param]
    return df[(df[param] >= low) & (df[param] <= high)]

def remove_spikes_median(df, col, window=5, factor=3, min_threshold=0.5, return_removed=False):
    if col not in df.columns or len(df) < window:
        return df.copy(), pd.DataFrame() if return_removed else df.copy()
    smoothed = df[col].rolling(window=window, center=True).median()
    diff = (df[col] - smoothed).abs()
    median_diff = diff.median()
    std_diff = df[col].std()
    threshold = max(factor * median_diff, 0.5 * std_diff, min_threshold)
    mask = diff < threshold
    cleaned_df = df[mask]
    removed_df = df[~mask]
    if return_removed:
        return cleaned_df, removed_df
    else:
        return cleaned_df
        
    
def load_system_data(path, room_name=None, param_filter=None):
    xls = pd.ExcelFile(path)
    all_data = []

    for sheet in xls.sheet_names:
        sheet_lower = sheet.lower().replace(" ", "")
        room_match = room_name.lower().replace(" ", "") if room_name else ""
        param_match = param_filter.lower() if param_filter else ""

        if room_match not in sheet_lower:
            continue

        try:
            df = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue

        df['lokale'] = room_name if room_name else sheet
        df.columns = df.columns.str.strip().str.lower()

        if 'tid' not in df.columns:
            continue

        df['tid'] = pd.to_datetime(df['tid'], errors='coerce')
        df = df.dropna(subset=['tid'])
        df['tid'] = df['tid'].dt.tz_localize('Europe/Copenhagen')

        # Nu understøtter vi både enkel og dobbelt kolonner:
        for col_key, label in [('co2', 'co2'), ('temp', 'temperatur')]:
            if param_filter and not col_key.startswith(param_filter.lower()):
                continue

            # Først enkelt kolonne (fx 'co2', 'temp')
            match = [c for c in df.columns if c == col_key]
            for col in match:
                series = pd.to_numeric(df[col], errors='coerce')
                if series.dropna().empty:
                    continue

                temp = df[['tid']].copy()
                temp['parameter'] = label
                temp['værdi'] = series
                temp['lokale'] = room_name if room_name else sheet
                all_data.append(temp[['tid', 'parameter', 'værdi', 'lokale']])

            # Dernæst to kolonner (fx 'co2 1', 'co2 2' / 'temp 1', 'temp 2')
            for i in [1, 2]:
                multi_col = f"{col_key} {i}"
                if multi_col in df.columns:
                    series = pd.to_numeric(df[multi_col], errors='coerce')
                    if series.dropna().empty:
                        continue

                    temp = df[['tid']].copy()
                    temp['parameter'] = f"{label} {i}"
                    temp['værdi'] = series
                    temp['lokale'] = room_name if room_name else sheet
                    all_data.append(temp[['tid', 'parameter', 'værdi', 'lokale']])

    if not all_data:
        return pd.DataFrame(columns=['tid', 'parameter', 'værdi', 'lokale'])

    return pd.concat(all_data, ignore_index=True)

def param_match(system_param, selected_param):
    selected_param = selected_param.lower()
    system_param = system_param.lower()

    if selected_param in ["co2"] and "co2" in system_param:
        return True
    if selected_param in ["t", "temp", "temperature", "temperatur"] and "temp" in system_param:
        return True

    return False


app = Dash(__name__)

available_parameters = [col for col in sensor_df.columns if col not in ['ts', 'sensor_id']]

param_names = {
    'co2': 'CO₂ [ppm]', 'tvoc': 'TVOC [ppb]', 'pm25': 'PM2.5 [µg/m³]',
    'pm1': 'PM1 [µg/m³]', 'pm10': 'PM10 [µg/m³]', 'temperature': 'Temperature [°C]',
    'humidity': 'Humidity [%]', 'pressure': 'Pressure [hPa]'
}

sensor_to_room = {
    'TAIL-21': ['DTU A34', 'CBS 033', 'KU A11', 'DTU 83', 'Emdrup', 'Århus 113', 'Ålborg fib10'],
    'TAIL-22': ['DTU A34', 'DTU 49', 'CBS 033', 'RUC A25', 'Emdrup','KU A81', 'Århus 113', 'Ålborg fib10','KU HCØ'],
    'TAIL-23': ['DTU A34', 'CBS 033', 'KU A11', 'DTU 83','Emdrup', 'Århus 113', 'Ålborg fib10','KU HCØ'],
    'TAIL-24': ['DTU A34', 'CBS 033', 'KU A81', 'DTU 83', 'Emdrup', 'Århus 113', 'Ålborg selma 300'],
    'TAIL-25': ['DTU A34', 'DTU 42', 'KEA', 'RUC A01', 'CBS 033', 'KU A11', 'DTU 83', 'Emdrup', 'Ålborg selma 300'],
    'TAIL-26': ['DTU A34', 'DTU 42', 'KEA', 'RUC A25', 'CBS 202', 'Ålborg selma 300'],
    'TAIL-27': ['DTU A34', 'RUC A01', 'CBS 202', 'KU A11', 'Århus 105', 'Ålborg krog'],
    'TAIL-28': ['DTU A34', 'DTU 49', 'KEA', 'RUC A01', 'CBS 202', 'KU A81', 'DTU 81', 'Århus 105', 'Ålborg krog'],
    'TAIL-29': ['DTU A34', 'CBS 202', 'KU A81', 'DTU 81', 'Århus 105', 'Ålborg krog','KU HCØ'],
    'TAIL-30': ['DTU A34', 'DTU 42', 'KEA', 'RUC A01', 'CBS 202', 'KU A81', 'DTU 81', 'Århus 105', 'Ålborg krog','KU HCØ'],
    'TAIL-Airthings1': ['DTU A34'],
    'TAIL-Airthings2': ['DTU A34']
}

app.layout = html.Div([
    html.H1("Sensor og persondata visualisering"),
    html.Label("Vælg lokale:"),
    dcc.Dropdown(id='room-dropdown',
                 options=[{'label': r, 'value': r} for r in sorted(set(v for rooms in sensor_to_room.values() for v in rooms))],
                 value='KEA'),
    html.Label("Vælg sensorer:"),
    dcc.Dropdown(id='sensor-dropdown', multi=True),
    html.Label("Vælg parameter:"),
    dcc.Dropdown(id='parameter-dropdown', options=[{'label': p, 'value': p} for p in available_parameters], value='co2'),
    html.Label("Vælg dato-interval:"),
    dcc.DatePickerRange(id='date-picker'),
    html.Label("Vælg timer på dagen:"),
    dcc.RangeSlider(id='hour-slider', min=0, max=23, value=[7, 17], step=1, marks={i: f"{i}" for i in range(24)}),
    dcc.Graph(id='sensor-plot'),
    html.Label("Show amount of people?"),
    dcc.Checklist(id='person-toggle', options=[{'label': 'Show amount of people', 'value': 'show'}], value=[]),
    html.Label("Vis systemdata?"),
    dcc.Checklist(id='system-toggle', options=[{'label': 'Vis systemdata', 'value': 'show_system'}],value=['show_system']),  # Systemdata vises som standard),
    html.H3("Gennemsnitsværdier for valgte tidsinterval"),
    dash_table.DataTable(id='summary-table',
        columns=[
            {"name": "Sensor", "id": "sensor"},
        {"name": "CO2 [ppm]", "id": "co2"},
        {"name": "Temperature [°C]", "id": "temperature"},
        {"name": "Noise", "id": "noise"},
        {"name": "Light", "id": "light"},
        {"name": "Antal målinger", "id": "count"},
        {"name": "Avg persons", "id": "person_avg"},
        {"name": "Max persons", "id": "person_max"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center"})
])

@app.callback(
    [Output('sensor-dropdown', 'options'),
     Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date')],
    Input('room-dropdown', 'value')
)
def update_sensor_options(room):
    options = [{'label': s, 'value': s} for s, rooms in sensor_to_room.items() if room in rooms]
    room_periods = {
        'DTU A34': ('2025-03-17', '2025-03-28'),
        'DTU 49': ('2025-03-31', '2025-04-02'),
        'DTU 42': ('2025-03-31', '2025-04-02'),
        'KEA': ('2025-04-03', '2025-04-03'),
        'RUC A25': ('2025-04-07', '2025-04-09'),
        'RUC A01': ('2025-04-07', '2025-04-09'),
        'CBS 202': ('2025-04-10', '2025-04-11'),
        'CBS 033': ('2025-04-10', '2025-04-11'),
        'KU A11': ('2025-04-22', '2025-04-23'),
        'KU A81': ('2025-04-22', '2025-04-23'),
        'DTU 83': ('2025-04-24', '2025-04-24'),
        'DTU 81': ('2025-04-25', '2025-04-25'),
        'Emdrup': ('2025-04-28', '2025-04-29'),
        'Århus 105': ('2025-05-05', '2025-05-06'),
        'Århus 113': ('2025-05-05', '2025-05-06'),
        'Ålborg krog': ('2025-05-07', '2025-05-08'),
        'Ålborg fib10': ('2025-05-07', '2025-05-08'),
        'Ålborg selma 300': ('2025-05-07', '2025-05-08'),
        'KU HCØ': ('2025-05-14', '2025-05-15')
    }
    if room in room_periods:
        start_str, end_str = room_periods[room]
        return options, start_str + "T00:00:00", end_str + "T23:59:59"
    return options, "2025-03-17T00:00:00", "2025-05-15T23:59:59"


@app.callback(
    Output('sensor-plot', 'figure'),
    Output('summary-table', 'data'),
    [
        Input('sensor-dropdown', 'value'),
        Input('parameter-dropdown', 'value'),
        Input('date-picker', 'start_date'),
        Input('date-picker', 'end_date'),
        Input('room-dropdown', 'value'),
        Input('person-toggle', 'value'),
        Input('system-toggle', 'value'),
        Input('hour-slider', 'value')
    ]
)
def update_plot(sensor_ids, parameter, start_date, end_date, room, show_persons, show_system, hours):
    fig = go.Figure()
    rows = []

    if not sensor_ids:
        return fig, []

    start_ts = pd.to_datetime(start_date).tz_localize('Europe/Copenhagen')
    end_ts = pd.to_datetime(end_date).tz_localize('Europe/Copenhagen')

    # Persondata statistik
    person_avg = "-"
    person_max = "-"
    df_person = person_data_all.get(room)
    if df_person is not None:
        df_person.columns = df_person.columns.str.strip().str.lower()
        if 'ts' in df_person.columns and 'personer' in df_person.columns:
            df_person = df_person.dropna(subset=['ts', 'personer'])
            dfp = df_person.copy()
            dfp = dfp[(dfp['ts'] >= start_ts) & (dfp['ts'] <= end_ts)]
            h_start, h_end = int(hours[0]), int(hours[1])
            dfp = filter_by_hours(dfp, 'ts', hours)
            if not dfp.empty:
                person_avg = round(dfp['personer'].mean(), 1)
                person_max = int(dfp['personer'].max())
    else:
        dfp = pd.DataFrame(columns=['ts', 'personer'])

    # Plot personer hvis valgt
    if 'show' in show_persons and not dfp.empty:
        fig.add_trace(go.Scatter(x=dfp['ts'], y=dfp['personer'], name="Personer", yaxis="y2", line=dict(dash='dot')))

    for i, sensor in enumerate(sensor_ids):
        sub = sensor_df[sensor_df['sensor_id'] == sensor]
        sub = sub[(sub['ts'] >= start_ts) & (sub['ts'] <= end_ts)]
        sub = filter_by_hours(sub, 'ts', hours)

        if parameter not in sub.columns:
            continue

        sub = sub.dropna(subset=[parameter]).sort_values('ts')
        sub = clean_with_limits(sub, parameter)
        sub, removed = remove_spikes_median(sub, parameter, window=5, factor=3, return_removed=True)

        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub['ts'], y=sub[parameter], name=sensor))

        if not removed.empty:
            fig.add_trace(go.Scatter(x=removed['ts'], y=removed[parameter], mode='markers',
                                      name=f"{sensor} spikes", marker=dict(color='red', size=6, symbol='x')))

        values = {}
        for param in ['co2', 't', 'noise', 'light']:
            if param in sub:
                temp_df = sub[['ts', param]].dropna()
                temp_df = clean_with_limits(temp_df, param)
                temp_df, _ = remove_spikes_median(temp_df, param, return_removed=True)
                values[param] = temp_df[param].mean()
                values[f'{param}_count'] = len(temp_df)
            else:
                values[param] = None
                values[f'{param}_count'] = 0

        rows.append({
            "sensor": sensor,
            "co2": round(values['co2'], 1) if values['co2'] is not None else "-",
            "temperature": round(values['t'], 1) if values['t'] is not None else "-",
            "noise": round(values['noise'], 1) if values['noise'] is not None else "-",
            "light": round(values['light'], 1) if values['light'] is not None else "-",
            "count": max(values[f'{p}_count'] for p in ['co2', 't', 'noise', 'light']),
            "person_avg": person_avg,
            "person_max": person_max
        })

        # Beregn korrelation for denne sensor
        if not dfp.empty and not sub.empty:
            merged = pd.merge_asof(
                sub[['ts', parameter]].sort_values('ts'),
                dfp[['ts', 'personer']].sort_values('ts'),
                on='ts',
                direction='nearest',
                tolerance=pd.Timedelta('2min')
            )
            merged = merged.dropna(subset=[parameter, 'personer'])
            if not merged.empty:
                corr = merged[parameter].corr(merged['personer'])
                fig.add_annotation(
                    text=f"{sensor} corr: {corr:.2f}",
                    xref="paper", yref="paper",
                    x=0.01, y=0.95 - 0.05 * i,
                    showarrow=False
                )

    # === SYSTEM DATA håndtering ===
    if 'show_system' in show_system:
        df_sys = load_system_data(system_data_path, room_name=room)
        df_sys = df_sys.dropna(subset=['værdi'])
        df_sys = df_sys[(df_sys['tid'] >= start_ts) & (df_sys['tid'] <= end_ts)]
        df_sys = filter_by_hours(df_sys, 'tid', hours)

        # Filtrer kun systemdata som matcher valgt parameter
        relevant_system_params = [
            p for p in df_sys['parameter'].unique()
            if param_match(p, parameter)
        ]

        for param_name in relevant_system_params:
            group = df_sys[df_sys['parameter'] == param_name]
            if group.empty:
                continue

            fig.add_trace(go.Scatter(
                x=group['tid'],
                y=group['værdi'],
                name=f"SYSTEM: {param_name}",
                mode='lines',
                line=dict(width=3,color='black')
            ))

            sys_avg = group['værdi'].mean()

            rows.append({
                "sensor": f"System ({param_name})",
                "co2": round(sys_avg, 1) if "co2" in param_name.lower() else "-",
                "temperature": round(sys_avg, 1) if "temp" in param_name.lower() else "-",
                "noise": "-", "light": "-",
                "count": len(group),
                "person_avg": person_avg,
                "person_max": person_max
            })

    fig.update_layout(
        title=f"{parameter} over tid",
        template='plotly_white',
        xaxis_title="Tid",
        yaxis_title=parameter,
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='black'),
        yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Personer'),
        legend=dict(orientation="h")
    )

    return fig, rows



if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    