import pandas as pd
import os
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import numpy as np
from scipy.stats import norm
import plotly.colors as pc
import matplotlib.colors as mc
import colorsys

# Funktion der gør en farve mørkere (bruges til normalfordeling)


def darken_color(color, amount=0.2):
    try:
        c = mc.cnames.get(color, color)
        r, g, b = mc.to_rgb(c)
        return f"rgb({int(r*(1-amount)*255)}, {int(g*(1-amount)*255)}, {int(b*(1-amount)*255)})"
    except:
        return color


# === 1. DATAINDSLÆSNING ===
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

# Fjern besvarelser der ikke er fuldført (statoverall_4 != 1)
df = df[(df['statoverall_4'] == 1)]

df['respondent_id'] = df.index + 1

labels_df = pd.read_csv(labels_path, sep=';', header=None, names=['raw'])
labels_df[['variable', 'value', 'label']] = labels_df['raw'].str.split(';', expand=True)
labels_df['value'] = pd.to_numeric(labels_df['value'], errors='coerce')
labels_df['clean_label'] = labels_df['label'].str.replace(
    r'^\d+\s*', '', regex=True).str.replace('"', '')
labels_df = labels_df.sort_values(['variable', 'value'])

spm_cols = [col for col in labels_df['variable'].unique() if col in df.columns]

if 'starttime' in df.columns:
    df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
else:
    df['starttime'] = pd.Timestamp('2025-01-01')

df_long = df[spm_cols + ['filnavn', 'respondent_id', 'starttime']].melt(
    id_vars=['filnavn', 'respondent_id', 'starttime'],
    var_name='variable',
    value_name='value'
)
df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
df_long = df_long.merge(labels_df[['variable', 'value', 'clean_label']], how='left', on=[
                        'variable', 'value'])
df_long = df_long.rename(columns={'clean_label': 'label'})
df_long = df_long.dropna(subset=['label'])

# === 2. LAYOUT ===
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Spørgeskema-sammenligning"),

    html.Label("Grupperingsvariabel"),
    dcc.Dropdown(id='group_var'),

    html.Label("Svarvariabel"),
    dcc.Dropdown(id='target_var'),

    html.Label("Lokation"),
    dcc.Dropdown(id='location_filter', multi=True),

    html.Label("Dato-interval"),
    dcc.DatePickerRange(id='date_range'),

    html.Label("Vis normalfordelingskurve"),
    dcc.Checklist(id='show_normal',
                  options=[{'label': 'Tilføj normalfordeling', 'value': 'show'}],
                  value=[], labelStyle={'display': 'inline-block', 'margin-right': '10px'}),


    html.Label("Y-akse visning"),
    html.Label("Søjlediagram-type"),
    dcc.RadioItems(
        id='barmode',
        options=[
            {'label': 'Group', 'value': 'group'},
            {'label': 'Stack', 'value': 'stack'}
        ],
        value='group',   # default
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),

    html.Label("Legend placering"),
    dcc.RadioItems(
        id='legend_position',
        options=[
            {'label': 'Venstre', 'value': 'left'},
            {'label': 'Højre', 'value': 'right'}
        ],
        value='right',
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),
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
    Output('group_var', 'options'),
    Output('group_var', 'value'),
    Output('target_var', 'options'),
    Output('target_var', 'value'),
    Output('location_filter', 'options'),
    Output('location_filter', 'value'),
    Output('date_range', 'start_date'),
    Output('date_range', 'end_date'),
    Input('comparison_plot', 'id')
)
def setup_dropdowns(_):
    vars_ = df_long['variable'].unique()
    locations = ['Alle'] + sorted(df_long['filnavn'].unique())
    start_date = df_long['starttime'].min().date()
    end_date = df_long['starttime'].max().date()
    group_options = [{'label': 'Ingen gruppeopdeling', 'value': 'none'}] + \
        [{'label': v, 'value': v} for v in vars_]
    return (
        group_options, 'none',
        [{'label': v, 'value': v} for v in vars_], vars_[0],
        [{'label': l, 'value': l} for l in locations], sorted(df_long['filnavn'].unique()),
        start_date, end_date
    )


@app.callback(
    Output('comparison_plot', 'figure'),
    Output('summary_table', 'columns'),
    Output('summary_table', 'data'),
    Input('group_var', 'value'),
    Input('target_var', 'value'),
    Input('location_filter', 'value'),
    Input('date_range', 'start_date'),
    Input('date_range', 'end_date'),
    Input('show_normal', 'value'),
    Input('y_axis_mode', 'value'),
    Input('legend_position', 'value'),
    Input('barmode', 'value')
)
def update_plot_and_table(group_var, target_var, location_filter, start_date, end_date, show_normal, y_axis_mode, legend_position, barmode):
    try:
        df_filtered = df_long.copy()

        # Filtrér lokationer
        if not location_filter or "Alle" in location_filter:
            pass  # behold alt
        else:
            df_filtered = df_filtered[df_filtered['filnavn'].isin(location_filter)]

        # Læg en dag til slutdatoen
        end_date_plus1 = pd.to_datetime(end_date) + pd.Timedelta(days=1)

        df_filtered = df_filtered[
            (df_filtered['starttime'] >= pd.to_datetime(start_date)) &
            (df_filtered['starttime'] < end_date_plus1)
        ]

        # === UDEN GRUPPEOPDELING ===
        if group_var == 'none':
            df_target = df_filtered[df_filtered['variable'] == target_var]
            df_target = df_target.groupby('label').size().reset_index(name='n')
            df_target['Percentage'] = round(df_target['n'] / df_target['n'].sum() * 100, 1)
            df_target['Number of responses'] = df_target['n']
            df_target['y_val'] = df_target[y_axis_mode]

            # Opret numerisk x-akse
            label_order = list(labels_df[labels_df['variable'] ==
                               target_var]['clean_label'].unique())
            label_to_num = {label: i + 1 for i, label in enumerate(label_order)}
            df_target['x'] = df_target['label'].map(label_to_num)

            # Plot
            fig = px.bar(df_target, x='x', y='y_val', text='n')
            fig.update_traces(textposition='outside')
            # Justér y-akse range med 10% luft over højeste søjle
            max_height = df_target['y_val'].max()
            fig.update_yaxes(range=[0, max_height * 1.10])

            fig.update_layout(
                title='',
                legend_title_text='',
                legend=dict(bgcolor='rgba(255,255,255,0.6)', orientation='v',
                            yanchor='top', y=0.99,
                            xanchor='left' if legend_position == 'left' else 'right',
                            x=0.01 if legend_position == 'left' else 0.99),
                xaxis_title='',
                yaxis_title=y_axis_mode
            )
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(label_to_num.values()),
                ticktext=list(label_to_num.keys())
            )

            # Normalfordeling
            if 'show' in show_normal:
                values = np.repeat(df_target['x'], df_target['n'])
                mu, sigma = norm.fit(values)
                x_vals = np.linspace(min(label_to_num.values()),
                                     max(label_to_num.values()), 200)
                y_vals = norm.pdf(x_vals, mu, sigma)
                max_height = df_target['y_val'].max()
                y_scaled = y_vals / y_vals.max() * (0.95 * max_height)

                fig.add_scatter(x=x_vals, y=y_scaled, mode='lines',
                                name='Normalfordeling', line=dict(dash='dot', color='black'))

            # Tabel
            df_table = df_target.rename(columns={'label': 'Svar'})[
                ["Svar", "Number of responses", "Percentage"]
            ]

            # Beregn totaler
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
            df_pivot = df_filtered[df_filtered['variable'].isin([group_var, target_var])] \
                .pivot_table(index='respondent_id', columns='variable', values='label', aggfunc='first') \
                .dropna()

            df_grouped = df_pivot.groupby(
                [group_var, target_var]).size().reset_index(name='n')

            # Tilføj 0'er for kombinationer uden svar
            all_group_labels = labels_df[labels_df['variable']
                                         == group_var]['clean_label'].unique()
            all_target_labels = labels_df[labels_df['variable']
                                          == target_var]['clean_label'].unique()
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
                         color=group_var, barmode=barmode, text='n')
            fig.update_traces(textposition='outside')

            if barmode == 'stack':
                total_heights = df_grouped.groupby('x')['y_val'].sum()
                max_height = total_heights.max()
            else:
                max_height = df_grouped['y_val'].max()

            fig.update_yaxes(range=[0, max_height * 1.1])

            fig.update_layout(
                title='',
                legend_title_text='',
                legend=dict(bgcolor='rgba(255,255,255,0.6)', orientation='v',
                            yanchor='top', y=0.99,
                            xanchor='left' if legend_position == 'left' else 'right',
                            x=0.01 if legend_position == 'left' else 0.99),
                xaxis_title='',
                yaxis_title=y_axis_mode
            )
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(label_to_num.values()),
                ticktext=list(label_to_num.keys())
            )

            # Normalfordeling med samme farve som bar – mørkere
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
                        showlegend=False, name=f'Normalfordeling ({group})',

                        line=dict(dash='dot', color=darker_color)
                    )

            # Tabel
            df_table = df_grouped.rename(columns={group_var: "Gruppe", target_var: "Svar"})[
                ["Gruppe", "Svar", "Number of responses", "Percentage"]
            ]

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
        return px.Figure(), [], []


if __name__ == "__main__":
    app.run(debug=True)
