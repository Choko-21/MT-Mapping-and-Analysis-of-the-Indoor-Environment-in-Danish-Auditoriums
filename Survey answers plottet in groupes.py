import pandas as pd
import os
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
import plotly.colors as pc
import matplotlib.colors as mc
import colorsys

# Funktion der gør en farve mørkere (bruges til Normal distribution)


def darken_color(color, amount=0.2):
    try:
        c = mc.cnames.get(color, color)
        r, g, b = mc.to_rgb(c)
        return f"rgb({int(r*(1-amount)*255)}, {int(g*(1-amount)*255)}, {int(b*(1-amount)*255)})"
    except:
        return color


# === 1. DATAINDSLÆSNING ===

# === DAILY DATA (OCCUPANCY AND TEMP) ===
# Indlæs daglig MAX belægning og MAX udetemperatur per lokale og dato
path_daily = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/daily_data.xlsx"
df_daily = pd.read_excel(path_daily)
df_daily['dato'] = pd.to_datetime(df_daily['dato'], dayfirst=True)
df_daily['dato'] = df_daily['dato'].dt.date
df_daily['occupancy_group'] = np.where(
    df_daily['occupancy'] < 0.50, 'Less than 50%', '50% or more')
df_daily['temp_group'] = np.where(
    df_daily['outdoor_temp'] < 14, 'Lower then 14°C', '14°C or higher')
folder_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/spørgeskema - svar ved påske (KEA RUC CBS)/Dataset/"
labels_path = "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/spørgeskema - svar ved påske (KEA RUC CBS)/BOBOBOB/labels__medtal.csv"

all_files = [f for f in os.listdir(folder_path) if f.startswith(
    "dataset_") and f.endswith(".csv")]
df_list = []
for filename in all_files:
    filepath = os.path.join(folder_path, filename)
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    df['filnavn'] = os.path.splitext(filename)[0]
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)


# === TÆLLING AF STATOOVERALL ===
overall_counts = df.groupby('filnavn').agg(
    statoverall_2=('statoverall_2', 'sum'),
    statoverall_3=('statoverall_3', 'sum'),
    statoverall_4=('statoverall_4', 'sum')
).reset_index()

# Beregn total række
total_row = pd.DataFrame({
    'filnavn': ['Total'],
    'statoverall_2': [overall_counts['statoverall_2'].sum()],
    'statoverall_3': [overall_counts['statoverall_3'].sum()],
    'statoverall_4': [overall_counts['statoverall_4'].sum()]
})

# Kombinér
overall_counts = pd.concat([overall_counts, total_row], ignore_index=True)

# Gem som CSV
overall_counts.to_csv(
    "C:/Users/Martin/OneDrive - Danmarks Tekniske Universitet/MT-Optimizing ventilation in classrooms/Data fra forsøg/spørgeskema_overview.csv",
    sep=';', index=False, encoding='utf-8'
)

print("✅ CSV-fil gemt!")


df = df[(df['statoverall_4'] == 1)]
df['respondent_id'] = df.index + 1

# === LOKALEMETADATA ===
lokale_metadata = {
    "dataset_RUC_00_1_009": ("RUC", 1993, 250, 352, "In the suburb", "more than 60%", "large", "Under-seat air supply", "below xx"),
    "dataset_RUC_25_2_035": ("RUC", 2001, 120, 105, "In the suburb", "less occupied", "small", "Under-seat air supply", "above"),
    "dataset_CBS_SP202":    ("CBS", 2000, 205, 239, "In the city", "more than 60%", "large", "Under-seat air supply", "below xx"),
    "dataset_CBS_DH_C_0_33": ("CBS", 1988, 194, 40.9, "In the city", "less occupied", "small", "Under-seat air supply", "above"),
    "dataset_KEA":          ("KEA", 1900, 51, 147, "In the city", "more than 60%", "small", "Other air supply types", "below xx"),
    "dataset_DTU_B116_81":  ("DTU", 1974, 275, 305.7, "In the suburb", "more than 60%", "large", "Under-seat air supply", "below xx"),
    "dataset_DTU_B116_83":  ("DTU", 1974, 100, 159.9, "In the suburb", "less occupied", "small", "Under-seat air supply", "above"),
    "dataset_DTU_b303a_42": ("DTU", 1974, 250, 267.1, "In the suburb", "more than 60%", "large", "Under-seat air supply", "below xx"),
    "dataset_DTU_b303a_49": ("DTU", 1974, 80, 169.2, "In the suburb", "less occupied", "small", "Under-seat air supply", "above"),
    "dataset_DTU_B306_34":  ("DTU", 1965, 194, 193.7, "In the suburb", "more than 60%", "large", "Under-seat air supply", "below xx"),
    "dataset_AAU_FIB_10":   ("AAU", 2002, 111, 108.1, "In the suburb", "less occupied", "small", "Other air supply types", "above"),
    "dataset_AAU_Krogh3":   ("AAU", 1996, 150, 195.6, "In the suburb", "more than 60%", "large", "Under-seat air supply", "below xx"),
    "dataset_AAU_Selma_300": ("AAU", 1990, 115, 227.1, "In the suburb", "less occupied", "small", "Other air supply types", "above"),
    "dataset_AU_1441_113":  ("AU", 2000, 81, 114, "In the city", "more than 60%", "large", "Under-seat air supply", "below xx"),
    "dataset_AU_1482_105":  ("AU", 2004, 197, 146, "In the city", "less occupied", "small", "Under-seat air supply", "above"),
    "dataset_Emdrup_D169":  ("AU", 1968, 238, 236.6, "In the city", "more than 60%", "large", "Under-seat air supply", "below xx"),
    "dataset_KU_782_11":    ("KU", 2016, 200, 198.1, "In the city", "less occupied", "small", "Other air supply types", "above"),
    "dataset_KU_782_81":    ("KU", 1970, 293, 276, "In the city", "more than 60%", "large", "Under-seat air supply", "below xx"),
    "dataset_KU_HCØ_1":     ("KU", 1959, 386, 416.75, "In the city", "less occupied", "small", "Other air supply types", "above")
}


meta_df = pd.DataFrame.from_dict(
    lokale_metadata,
    orient='index',
    columns=['universitet', 'build_year', 'kapacitet', 'areal', 'location_group',
             'occupation_group', 'size_group', 'ventilation_group', 'outdoor_temp_group']
).reset_index().rename(columns={'index': 'lokale_navn'})

# Tilføj kolonne til hoveddata og merge
# Tilføj dato til df og merge med daglig data

df['lokale_navn'] = df['filnavn'].str.replace(r"[ .-]", "_", regex=True)
df = df.merge(meta_df, how='left', on='lokale_navn')
if 'starttime' in df.columns:
    df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
else:
    df['starttime'] = pd.Timestamp('2025-01-01')
df['date'] = df['starttime'].dt.date
df['filnavn_clean'] = df['filnavn'].str.replace(r"[ .-]", "_", regex=True)
df_daily['lokale_clean'] = df_daily['lokale'].str.replace(r"[ .-]", "_", regex=True)
df = df.merge(df_daily, how='left', left_on=[
              'filnavn_clean', 'date'], right_on=['lokale_clean', 'dato'])

# Fjern respondenter uden merge med df_daily
before_rows = df.shape[0]

df = df[df['occupancy_group'].notna() & df['temp_group'].notna()]

after_rows = df.shape[0]
print(f"✅ Fjernet {before_rows - after_rows} respondenter uden merge med df_daily.")


# Find rækker uden merge til df_daily
missing_daily = df[
    df['occupancy_group'].isna() | df['temp_group'].isna()
]

print(f"\n🔍 Antal respondenter uden merge i df_daily: {missing_daily.shape[0]}")

if not missing_daily.empty:
    print("\nDe første rækker uden merge i df_daily:")
    print(missing_daily[[
        'filnavn', 'starttime'
    ]].head(10))

    # Tæl hvor mange der mangler per filnavn
    tab = missing_daily.groupby('filnavn') \
        .agg(antal=('respondent_id', 'nunique')) \
        .reset_index() \
        .sort_values('antal', ascending=False)

    print("\nDatasæt med respondenter der mangler merge med df_daily:")
    print(tab.to_string(index=False))
else:
    print("✅ Ingen respondenter mangler merge med df_daily.")


print("\n🔎 Samlet overview efter merge med df_daily:")
print("Total antal svar:", df.shape[0])
print("Antal med manglende occupancy:", df['occupancy'].isna().sum())
print("Antal med manglende outdoor_temp:", df['outdoor_temp'].isna().sum())
print("Antal med manglende occupancy_group:", df['occupancy_group'].isna().sum())
print("Unikke datoer i df:", df['date'].nunique())
print("Unikke datoer i df_daily:", df_daily['dato'].nunique())


# Debug: se om merge virker
print("Antal rækker før merge:", df.shape[0])
print("Antal rækker UDEN metadata efter merge:", df['universitet'].isna().sum())
print("Unikke filnavne uden match:", df[df['universitet'].isna()]['filnavn'].unique())

# Ekstra grupper
df['build_year_group'] = np.where(df['build_year'] < 1988, 'Before 1988', '1988 or later')
df['kapacitet_group'] = np.where(df['kapacitet'] < 194, 'Less then 194', '194 or more')
df['areal_group'] = np.where(df['areal'] < 196, 'Less then 196', '196 or larger')


# === LABELS ===
labels_df = pd.read_csv(labels_path, sep=';', header=None, names=['raw'])
labels_df[['variable', 'value', 'label']] = labels_df['raw'].str.split(';', expand=True)
labels_df['value'] = pd.to_numeric(labels_df['value'], errors='coerce')
labels_df['clean_label'] = labels_df['label'].str.replace(
    r'^\d+\s*', '', regex=True).str.replace('"', '')
labels_df = labels_df.sort_values(['variable', 'value'])

spm_cols = [col for col in labels_df['variable'].unique() if col in df.columns]

# Tilføj ekstra grupper til df_long
extra_groups = ['occupancy_group', 'temp_group', 'build_year', 'kapacitet',
                'areal', 'build_year_group', 'kapacitet_group', 'areal_group',
                'universitet', 'location_group', 'ventilation_group']


# Melt med ekstra grupper
df_long = df[spm_cols + ['filnavn', 'respondent_id', 'starttime'] + extra_groups].melt(
    id_vars=['filnavn', 'respondent_id', 'starttime'] + extra_groups,
    var_name='variable',
    value_name='value'
)
df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
df_long = df_long.merge(labels_df[['variable', 'value', 'clean_label']], how='left', on=[
                        'variable', 'value'])


# === CHECK FOR MISMATCH BEFORE DROPPING NA ===
# Step 2: Tjek for mismatch mellem df_long og labels_df
df_long_check = df[spm_cols + ['filnavn', 'respondent_id', 'starttime'] + extra_groups] \
    .melt(
        id_vars=['filnavn', 'respondent_id', 'starttime'] + extra_groups,
        var_name='variable',
        value_name='value'
)

df_long_check['value'] = pd.to_numeric(df_long_check['value'], errors='coerce')

df_long_check = df_long_check.merge(
    labels_df[['variable', 'value', 'clean_label']],
    how='left',
    on=['variable', 'value']
)

# behold kun rows hvor label mangler
mismatch_rows = df_long_check[
    df_long_check['clean_label'].isna() &
    df_long_check['value'].notna()
]

if mismatch_rows.empty:
    print("\n✅ Der er ingen besvarelser med værdier der mangler i labels_df.")
else:
    print(f"\n⚠️ Antal rows med værdier der ikke findes i labels_df: {mismatch_rows.shape[0]}")
    print("\nEksempel på de første 10 rows med mismatch:")
    print(mismatch_rows[['filnavn', 'respondent_id', 'variable', 'value']].head(10))

    # Lidt aggregering
    tab = mismatch_rows.groupby(['filnavn', 'variable']) \
        .agg(antal_observationer=('respondent_id', 'count')) \
        .reset_index() \
        .sort_values(['filnavn', 'antal_observationer'], ascending=[True, False])

    print("\nOversigt over mismatch fordelt på datasæt og variabel:")
    print(tab.to_string(index=False))


df_long = df_long.rename(columns={'clean_label': 'label'})
df_long = df_long.dropna(subset=['label'])


# === CHECK FOR RESPONDENTER UDEN GYLDIGE SVAR ===
respondenter_i_df = set(df['respondent_id'])
respondenter_i_long = set(df_long['respondent_id'])
uden_svar_ids = respondenter_i_df - respondenter_i_long

print(f"\n🕵️‍♂️ Antal respondenter med gennemført besvarelse men ingen gyldige spørgsmålssvar: {len(uden_svar_ids)}")

if len(uden_svar_ids) > 0:
    tab = df[df['respondent_id'].isin(uden_svar_ids)] \
        .groupby('filnavn') \
        .agg(antal_respondenter=('respondent_id', 'nunique')) \
        .reset_index() \
        .sort_values('antal_respondenter', ascending=False)

    print("\nDatasæt med respondenter uden gyldige svar:")
    print(tab.to_string(index=False))
else:
    print("✅ Alle respondenter har mindst ét gyldigt spørgsmålssvar.")


# === 2. LAYOUT ===
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Label("Vælg lokalekarakteristik til inspektion"),
    dcc.Dropdown(
        id='meta_inspect_dropdown',
        options=[
            {'label': 'Universitet', 'value': 'universitet'},
            {'label': 'Byggeår', 'value': 'build_year_group'},
            {'label': 'Kapacitet', 'value': 'kapacitet_group'},
            {'label': 'Areal', 'value': 'areal_group'},
            {'label': 'Lokation', 'value': 'location_group'},
            {'label': 'Ventilationstype', 'value': 'ventilation_group'},
            {'label': 'Occupancy', 'value': 'occupancy_group'},
            {'label': 'Outdoor temp (cold/warm)', 'value': 'temp_group'}
        ],
        value='universitet',
        style={'margin-bottom': '20px'}
    ),
    html.H1("Spørgeskema-sammenligning"),



    html.Label("Svarvariabel"),
    dcc.Dropdown(id='target_var'),

    html.Label("Lokation"),
    dcc.Dropdown(id='location_filter', multi=True),

    html.Label("Dato-interval"),
    dcc.DatePickerRange(id='date_range'),

    html.Label("Vis Normal distributionskurve"),
    dcc.Checklist(id='show_normal',
                  options=[{'label': 'Tilføj Normal distribution', 'value': 'show'}],
                  value=[], labelStyle={'display': 'inline-block', 'margin-right': '10px'}),

    html.Label("Søjlediagram-type"),
    dcc.RadioItems(
        id='bar_mode',
        options=[
            {'label': 'Side-by-side', 'value': 'group'},
            {'label': 'Stacked', 'value': 'stack'}
        ],
        value='group',
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),


    html.Label("Y-akse visning"),
    dcc.RadioItems(id='y_axis_mode',
                   options=[
                       {'label': 'Number of responses', 'value': 'Number of responses'},
                       {'label': 'Percentage', 'value': 'Percentage'}
                   ],
                   value='Percentage', labelStyle={'display': 'inline-block', 'margin-right': '10px'}),

    dcc.Graph(id='comparison_plot'),

    html.H3("Tabel over fordelinger"),
    dash_table.DataTable(
        id='summary_table',
        columns=[], data=[],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={'fontWeight': 'bold'},
    )
])

# === 3. DROPDOWNS ===


@app.callback(
    Output('target_var', 'options'),
    Output('target_var', 'value'),
    Output('location_filter', 'options'),
    Output('location_filter', 'value'),
    Output('date_range', 'start_date'),
    Output('date_range', 'end_date'),
    Input('comparison_plot', 'id')
)
def setup_target_and_location(_):
    vars_ = df_long['variable'].unique()
    locations = sorted(df_long['filnavn'].unique())
    start_date = df_long['starttime'].min().date()
    end_date = df_long['starttime'].max().date()
    return (
        [{'label': v, 'value': v} for v in vars_], vars_[0],
        [{'label': l, 'value': l} for l in locations], locations,
        start_date, end_date
    )


print("\n⏳ Dash default dato-interval:")
print("Start:", df_long['starttime'].min())
print("Slut :", df_long['starttime'].max())


@app.callback(
    Output('comparison_plot', 'figure'),
    Output('summary_table', 'columns'),
    Output('summary_table', 'data'),
    Input('meta_inspect_dropdown', 'value'),
    Input('target_var', 'value'),
    Input('location_filter', 'value'),
    Input('date_range', 'start_date'),
    Input('date_range', 'end_date'),
    Input('show_normal', 'value'),
    Input('y_axis_mode', 'value'),
    Input('bar_mode', 'value')
)
def update_metadata_plot(group_var, target_var, location_filter, start_date, end_date, show_normal, y_axis_mode, bar_mode):
    return update_plot_and_table(group_var, target_var, location_filter, start_date, end_date, show_normal, y_axis_mode, bar_mode)


def update_plot_and_table(group_var, target_var, location_filter, start_date, end_date, show_normal, y_axis_mode, bar_mode):
    try:
        df_filtered = df_long.copy()

        # Filtrér lokationer
        if not location_filter or "Alle" in location_filter:
            pass  # behold alt
        else:
            df_filtered = df_filtered[df_filtered['filnavn'].isin(location_filter)]

        # Filtrér på dato
        import datetime

        # Læg en dag til slutdatoen
        end_date_plus1 = pd.to_datetime(end_date) + pd.Timedelta(days=1)

        df_filtered = df_filtered[
            (df_filtered['starttime'] >= pd.to_datetime(start_date)) &
            (df_filtered['starttime'] < end_date_plus1)
        ]

        # === CHECK FOR RESPONDENTER DER FILTRERES VÆK I DATO-FILTER ===
        all_resp_ids = set(df_long['respondent_id'])
        filtered_resp_ids = set(df_filtered['respondent_id'])

        missing_resp_ids = all_resp_ids - filtered_resp_ids

        # === UDEN GRUPPEOPDELING ===
        if group_var == 'none':
            # Find alle respondenter
            all_respondents = df['respondent_id'].unique()

            # Udtræk svar på target_var
            svar_df = df_long[df_long['variable'] == target_var][['respondent_id', 'label']]

            # Merge så alle respondenter er med – selv dem uden svar
            svar_df_full = pd.DataFrame({'respondent_id': all_respondents}).merge(
                svar_df, on='respondent_id', how='left'
            )

            # Hvis nogen ikke har svaret, sæt label til "No response"
            svar_df_full['label'] = svar_df_full['label'].fillna('No response')

            # Tæl forekomster
            df_target = svar_df_full.groupby('label').size().reset_index(name='n')
            df_target['Percentage'] = round(df_target['n'] / df_target['n'].sum() * 100, 1)
            df_target['Number of responses'] = df_target['n']
            df_target['y_val'] = df_target[y_axis_mode]

            # Lav label-orden inkl. "No response"
            label_order = list(labels_df[labels_df['variable'] ==
                               target_var]['clean_label'].unique())
            if "No response" not in label_order:
                label_order.append("No response")

            label_to_num = {label: i + 1 for i, label in enumerate(label_order)}
            df_target['x'] = df_target['label'].map(label_to_num)

            # Plot
            fig = px.bar(df_target, x='x', y='y_val', text='n')
            fig.update_traces(textposition='outside')
            fig.update_layout(
                title='',
                width=900,
                height=400,
                legend_title_text='',
                legend=dict(orientation='v', yanchor='top', y=0.99, xanchor='right', x=0.99),
                xaxis_title=target_var,
                yaxis_title=y_axis_mode
            )

            fig.update_xaxes(
                tickmode='array',
                tickvals=list(label_to_num.values()),
                ticktext=list(label_to_num.keys())
            )

            # Normal distribution
            if 'show' in show_normal:
                values = np.repeat(df_target['x'], df_target['n'])
                mu, sigma = norm.fit(values)
                x_vals = np.linspace(min(label_to_num.values()),
                                     max(label_to_num.values()), 200)
                y_vals = norm.pdf(x_vals, mu, sigma)
                max_height = df_target['y_val'].max()
                y_scaled = y_vals / y_vals.max() * (0.95 * max_height)

                fig.add_scatter(x=x_vals, y=y_scaled, mode='lines',
                                showlegend=False, name='Normal distribution',
                                line=dict(dash='dot', color='black'))

            # Tabel
            df_table = df_target.rename(columns={'label': 'Svar'})[
                ["Svar", "Number of responses", "Percentage"]
            ]

            total_n = df_table["Number of responses"].sum()
            df_table.loc["Total"] = ["Total", total_n, 100.0]

            table_data = df_table.reset_index(drop=True).to_dict('records')

            table_columns = [
                {"name": "Svar", "id": "Svar"},
                {"name": "Number of responses", "id": "Number of responses"},
                {"name": "Percentage", "id": "Percentage"}
            ]

            return fig, table_columns, table_data

        # === MED GRUPPEOPDELING ===
        else:
            if group_var in df_filtered.columns and group_var not in ['variable']:
                df_target_only = df_filtered[df_filtered['variable']
                                             == target_var][['respondent_id', 'label']]
                df_pivot = df_target_only.rename(columns={'label': target_var})
                group_info = df_filtered.drop_duplicates(
                    'respondent_id')[['respondent_id', group_var]]
                df_pivot = df_pivot.merge(group_info, on='respondent_id', how='left')
                df_pivot = df_pivot.dropna()
            else:
                df_pivot = df_filtered[df_filtered['variable'].isin([group_var, target_var])] \
                    .pivot_table(index='respondent_id', columns='variable', values='label', aggfunc='first') \
                    .dropna()

            df_grouped = df_pivot.groupby(
                [group_var, target_var]).size().reset_index(name='n')

            # Tilføj 0'er for kombinationer uden svar
            if group_var in labels_df['variable'].unique():
                all_group_labels = labels_df[labels_df['variable']
                                             == group_var]['clean_label'].unique()
            else:
                all_group_labels = df_pivot[group_var].dropna().unique()

            if target_var in labels_df['variable'].unique():
                all_target_labels = labels_df[labels_df['variable']
                                              == target_var]['clean_label'].unique()
            else:
                all_target_labels = df_pivot[target_var].dropna().unique()

            all_combinations = pd.MultiIndex.from_product(
                [all_group_labels, all_target_labels],
                names=[group_var, target_var]
            ).to_frame(index=False)

            df_grouped = all_combinations.merge(
                df_grouped, on=[group_var, target_var], how='left').fillna({'n': 0})
            df_grouped['n'] = df_grouped['n'].astype(int)
            df_grouped['Percentage'] = df_grouped.groupby(group_var)['n'].transform(
                lambda x: round(x / x.sum() * 100, 1))
            df_grouped['Number of responses'] = df_grouped['n']
            df_grouped['y_val'] = df_grouped[y_axis_mode]

            label_order = list(labels_df[labels_df['variable'] ==
                               target_var]['clean_label'].unique())
            label_to_num = {label: i + 1 for i, label in enumerate(label_order)}
            df_grouped['x'] = df_grouped[target_var].map(label_to_num)

            fig = px.bar(df_grouped, x='x', y='y_val',
                         color=group_var, barmode=bar_mode, text='n')
            fig.update_traces(textposition='outside')

            max_y = df_grouped['y_val'].max()
            fig.update_yaxes(range=[0, max_y * 1.1])

            fig.update_layout(
                title='',
                width=900,
                height=400,
                legend_title_text='', legend=dict(orientation='v', yanchor='top', y=0.99, xanchor='left', x=0.01),
                xaxis_title=target_var,
                yaxis_title=y_axis_mode
            )

            fig.update_xaxes(
                tickmode='array',
                tickvals=list(label_to_num.values()),
                ticktext=list(label_to_num.keys())
            )

            # Normal distribution med samme farve som bar – mørkere
            if 'show' in show_normal:
                bar_colors = {
                    trace.name: trace.marker.color for trace in fig.data if trace.type == 'bar'}
                for group in df_grouped[group_var].unique():
                    sub = df_grouped[df_grouped[group_var] == group]
                    if sub['n'].sum() == 0:
                        continue
                    values = np.repeat(sub['x'], sub['n'])
                    mu, sigma = norm.fit(values)
                    x_vals = np.linspace(min(label_to_num.values()),
                                         max(label_to_num.values()), 200)
                    y_vals = norm.pdf(x_vals, mu, sigma)
                    max_height = sub['y_val'].max()
                    y_scaled = y_vals / y_vals.max() * (0.95 * max_height)
                    darker_color = darken_color(bar_colors.get(group, 'gray'), 0.3)

                    fig.add_scatter(
                        x=x_vals,
                        y=y_scaled,
                        mode='lines',
                        showlegend=False, name=f'Normal distribution ({group})',
                        line=dict(dash='dot', color=darker_color)
                    )

            # Tabel
            df_table = df_grouped.rename(columns={group_var: "Gruppe", target_var: "Svar"})[
                ["Gruppe", "Svar", "Number of responses", "Percentage"]
            ].sort_values(by=["Gruppe", "Svar"])

            # Tilføj total-række per gruppe
            total_row = pd.DataFrame({
                "Gruppe": ["Total"],
                "Svar": [""],
                "Number of responses": [df_table["Number of responses"].sum()],
                "Percentage": [100.0]
            })

            df_table = pd.concat([df_table, total_row], ignore_index=True)
            table_data = df_table.to_dict('records')

            table_columns = [
                {"name": "Gruppe", "id": "Gruppe"},
                {"name": "Svar", "id": "Svar"},
                {"name": "Number of responses", "id": "Number of responses"},
                {"name": "Percentage", "id": "Percentage"}
            ]
            return fig, table_columns, table_data

    except Exception as e:
        print("⚠️ Fejl i update_plot_and_table:", e)
        return go.Figure(), [], []


if __name__ == "__main__":
    app.run(debug=True)


# Mappe til plots
save_folder = r"C:\Users\Martin\OneDrive - Danmarks Tekniske Universitet\MT-Optimizing ventilation in classrooms\svar_plots_grupper"
os.makedirs(save_folder, exist_ok=True)

# Variabler du vil ekskludere
exclude_vars = [
    'statoverall_1',
    'statoverall_2',
    'statoverall_3',
    'statoverall_4',
    'statoverall_5',
    's_1',
    'language',
    'draft_acceptable',
    'draft_air_movement',
    'sex',
    'light_prefer',
    'placing_in_room'
]

# Korte navne
group_var_names = {
    'universitet': 'uni',
    'build_year_group': 'year',
    'kapacitet_group': 'kapa',
    'areal_group': 'area',
    'location_group': 'loc',
    'ventilation_group': 'ven',
    'occupancy_group': 'occ',
    'temp_group': 'out'
}

# Hent alle variable (spørgsmål)
all_vars = sorted([v for v in df_long['variable'].unique() if v not in exclude_vars])

# Gruppemuligheder
group_vars = list(group_var_names.keys())

# Loop igennem kombinationer
for group_var in group_vars:
    for target_var in all_vars:

        if group_var == target_var:
            continue

        try:
            fig, table_columns, table_data = update_plot_and_table(
                group_var=group_var,
                target_var=target_var,
                location_filter=['Alle'],
                start_date=df_long['starttime'].min().date(),
                end_date=df_long['starttime'].max().date(),
                show_normal=[],
                y_axis_mode='Percentage',
                bar_mode='group'
            )

            # Lav kort filnavn
            safe_group = group_var_names.get(group_var, group_var)
            filename = f"{safe_group}__{target_var}.png"
            save_path = os.path.join(save_folder, filename)

            # GEM MED FAST BREDDE/HØJDE:
            fig.write_image(
                save_path,
                width=900,
                height=400,
                scale=2
            )

        except Exception as e:
            print(f"⚠️ Fejl ved plot {group_var} / {target_var}: {e}")


print("✔ Alle plots genereret og gemt.")
