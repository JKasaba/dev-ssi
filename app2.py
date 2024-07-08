import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from dash import dash_table
from ssi_module import calculate_ssi
from planck_module import planck
from daylight_module import daylight
from cct_module import cct_mccamy
import io
import base64

# Constants
CORRECTION_FACTOR_DAYLIGHT = 14388 / 14380
CORRECTION_FACTOR_ILLUM_A = 14350 / 14388
MIN_WAVELENGTH = 300
MAX_WAVELENGTH = 830
DEFAULT_DAYLIGHT_CCT = 5000
DEFAULT_BLACKBODY_CCT = 3200
MIN_DAYLIGHT_CCT = 4000
MAX_DAYLIGHT_CCT = 25000
MIN_BLACKBODY_CCT = 1000
MAX_BLACKBODY_CCT = 10000
wavelengths = np.linspace(300, 830, 530)

CCT_MAPPING = {
    'D50': 5000 * CORRECTION_FACTOR_DAYLIGHT,
    'D55': 5500 * CORRECTION_FACTOR_DAYLIGHT,
    'D65': 6500 * CORRECTION_FACTOR_DAYLIGHT,
    'D75': 7500 * CORRECTION_FACTOR_DAYLIGHT,
    'Custom_Blackbody': 3200,
    'HMI': 5606,
    'A': 2855.542,
    'Xenon': 5159,
    'Warm LED': 3133,
    'Cool LED': 5300,
    'F1': 6425,
    'F2': 4224,
    'F3': 2447,
    'F4': 2939,
    'F5': 6342,
    'F6': 4148,
    'F7': 6489,
    'F8': 4995,
    'F9': 4147,
    'F10': 4988,
    'F11': 4001,
    'F12': 3002
}

# Function to interpolate and normalize spectra
def interpolate_and_normalize(spec):
    wavelengths = np.arange(MIN_WAVELENGTH, MAX_WAVELENGTH + 1)
    spec_resample = np.interp(wavelengths, spec['wavelength'], spec['intensity'])
    spec_resample /= spec_resample[np.where(wavelengths == 560)]
    return pd.DataFrame({'wavelength': wavelengths, 'intensity': spec_resample})

# Load test spectra from the provided text file
file_path_test = 'testSources_test.csv'
file_path_ref = 'daylighttestsources.csv'
test_spectra_df = pd.read_csv(file_path_test)
ref_spectra_df = pd.read_csv(file_path_ref)

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        return pd.read_excel(io.BytesIO(decoded))
    else:
        return pd.DataFrame()

# Interpolate and normalize each test spectrum
warm_led_spec = interpolate_and_normalize(test_spectra_df[['wavelength', 'Warm LED']].rename(columns={'Warm LED': 'intensity'}))
cool_led_spec = interpolate_and_normalize(test_spectra_df[['wavelength', 'Cool LED']].rename(columns={'Cool LED': 'intensity'}))
hmi_spec = interpolate_and_normalize(test_spectra_df[['wavelength', 'HMI']].rename(columns={'HMI': 'intensity'}))
xenon_spec = interpolate_and_normalize(test_spectra_df[['wavelength', 'Xenon']].rename(columns={'Xenon': 'intensity'}))
D50_spec = interpolate_and_normalize(ref_spectra_df[['wavelength', 'D50']].rename(columns={'D50': 'intensity'}))
D55_spec = interpolate_and_normalize(ref_spectra_df[['wavelength', 'D55']].rename(columns={'D55': 'intensity'}))
D65_spec = interpolate_and_normalize(ref_spectra_df[['wavelength', 'D65']].rename(columns={'D65': 'intensity'}))
D75_spec = interpolate_and_normalize(ref_spectra_df[['wavelength', 'D75']].rename(columns={'D75': 'intensity'}))
custom_spec_test = interpolate_and_normalize(test_spectra_df[['wavelength', 'Custom']].rename(columns={'Custom': 'intensity'}))
fluorescent_specs = {}
for i in range(1, 13):
    name = f'F{i}'
    fluorescent_specs[name] = interpolate_and_normalize(test_spectra_df[['wavelength', name]].rename(columns={name: 'intensity'}))

# Initialize Dash app with callback exceptions suppressed
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
server = app.server

# Define layout
app.layout = dbc.Container(fluid=True, children=[
    dbc.NavbarSimple(
        brand="SSI Calculator",
        brand_href="#",
        color="dark",
        dark=True,
        fluid=True,
        style={'backgroundColor': '#000000'}
    ),
    dcc.Store(id='stored-cct-value'),
    dcc.Store(id='stored-custom-spec'),

    dbc.Tabs([
        dbc.Tab(label="Calculations", children=[
            dbc.Row([
                dbc.Col(width=4, children=[  # Half the screen width for settings, reference, and spectral data
                    dbc.Card([
                        dbc.CardHeader("Test Spectrum", style={'backgroundColor': '#000000', 'color': '#BA9E5E'}),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='testChoice',
                                options=[
                                    {'label': 'Warm LED', 'value': 'Warm LED'},
                                    {'label': 'Cool LED', 'value': 'Cool LED'},
                                    {'label': 'HMI', 'value': 'HMI'},
                                    {'label': 'Xenon', 'value': 'Xenon'}
                                ] +[{'label': f'F{i}', 'value': f'F{i}'} for i in range(1, 13)] + [{'label': 'Custom', 'value': 'Custom'}],
                                value='Warm LED'
                            ),
                            # html.Div(id='customTestSpecInputs')
                        ])
                    ],  style={'margin-top': '20px'}),
                    dbc.Card([
                        dbc.CardHeader("Upload CSV or Excel", style={'backgroundColor': '#000000', 'color': '#BA9E5E'}),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                            html.Div(id='output-data-upload')
                        ])
                    ], id='upload-card', style={'margin-top': '20px', 'display': 'none'}),
                    dbc.Card([
                        dbc.CardHeader("Reference Spectrum", style={'backgroundColor': '#000000', 'color': '#BA9E5E'}),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='refChoice',
                                options=[
                                    {'label': 'Default', 'value': 'Default'},
                                    {'label': 'Blackbody: A', 'value': 'A'},
                                    {'label': 'Blackbody: Custom CCT', 'value': 'Custom_Blackbody'},
                                    {'label': 'Daylight: D50', 'value': 'D50'},
                                    {'label': 'Daylight: D55', 'value': 'D55'},
                                    {'label': 'Daylight: D65', 'value': 'D65'},
                                    {'label': 'Daylight: D75', 'value': 'D75'},
                                    {'label': 'Daylight: Custom CCT', 'value': 'Custom_Daylight'}
                                ],
                                value='Default'
                            ),
                            html.Div(id='refSpecInputs', style={'margin-top': '20px'}),
                            dbc.Row([
                                dbc.Label("CCT"),
                                dbc.Input(
                                    type="number",
                                    id="refCct",
                                    value=4000,
                                    min=MIN_BLACKBODY_CCT,
                                    max=MAX_BLACKBODY_CCT,
                                    step=1,
                                    debounce=True
                                ),
                            ], id='customCctInput', style={'margin-top': '20px', 'display': 'none'}),
                            dbc.Button("Submit CCT", id="submit-cct", color="primary", className="mr-2", style={'margin-top': '10px', 'display': 'none'}), 
                            html.Div(id='warning-message', style={'color': 'red', 'margin-top': '10px'})
                        ])
                    ], style={'margin-top': '20px'}),
                    
                    dbc.Card([
                        dbc.CardHeader("Spectral Data", style={'backgroundColor': '#000000', 'color': '#BA9E5E'}),
                        dbc.CardBody([
                            dcc.Tabs([
                                dcc.Tab(label='Test Spectrum', children=[
                                    html.Div(id='spectraTest', children=[dash_table.DataTable(
                                        id='spectra-test-table', 
                                        data=warm_led_spec.to_dict('records'),  # Initialize with default data
                                        columns=[{"name": i, "id": i} for i in warm_led_spec.columns],   # Initialize with default columns
                                        editable=True,
                                        page_size=1000,  # Number of rows per page
                                        export_format="csv",
                                        export_headers="display",
                                    )])
                                ]),
                                dcc.Tab(label='Reference Spectrum', children=[
                                    html.Div(id='spectraRef', )
                                ]),
                            ]),
                        ])
                    ], style={'margin-top': '20px'}),
                ]),
                dbc.Col(width=8, children=[  # Half the screen width for the graph
                    dbc.Card([
                        dbc.CardHeader("Graph", style={'backgroundColor': '#000000', 'color': '#BA9E5E'}),
                        dbc.CardBody([dcc.Graph(id='plotRef')]),
                    ], style={'margin-top': '20px'}),
                    dbc.Card([
                        dbc.CardHeader("Spectral Similarity Index (SSI)", style={'backgroundColor': '#000000', 'color': '#BA9E5E'}),
                        dbc.CardBody([html.H4(id='ssiText', className='card-text')]),
                    ], style={'margin-top': '20px'}),
                ]),
            ]),
        ]),
        dbc.Tab(label="About", children=[
            dbc.Container([
                html.H2('Introduction'),
                dcc.Markdown('Include the content of INTRODUCTION.md here...'),
                html.H2('Software'),
                html.P('Built using ...'),
                html.H2('License Terms'),
                dcc.Markdown('Include the content of LICENSE.md here...'),
            ], style={'margin-top': '20px'})
        ]),
    ]),
])


@app.callback(
    Output('upload-card', 'style'),
    Input('testChoice', 'value')
)
def update_upload_card_visibility(test_choice):
    if test_choice == 'Custom':
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output('spectra-test-table', 'data'),
    Output('spectra-test-table', 'columns'),
    Output('stored-custom-spec', 'data'),
    Input('testChoice', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_spectra_test_table(test_choice, contents, filename):

    if test_choice == 'Warm LED':
        df = warm_led_spec
    elif test_choice == 'Cool LED':
        df = cool_led_spec
    elif test_choice == 'HMI':
        df = hmi_spec
    elif test_choice == 'Xenon':
        df = xenon_spec
    elif test_choice == 'Custom':
        if contents is not None:
            df = parse_contents(contents, filename)
            custom_spec = df
            df = interpolate_and_normalize(df)
            return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], custom_spec.to_dict('records')
        else:
            df = warm_led_spec
    elif test_choice in fluorescent_specs:
        df = fluorescent_specs[test_choice]
    else:
        return [], [], {}

    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], {}

@app.callback(
    [Output('customCctInput', 'style'),
     Output('submit-cct', 'style'),
     Output('warning-message', 'children')],
    [Input('refChoice', 'value'),
     Input('refCct', 'value')]
)
def update_cct_input_visibility(ref_choice, ref_cct):
    if ref_choice == 'Custom_Blackbody' or ref_choice == 'Custom_Daylight':
        warning_message = ""
        if ref_choice == 'Custom_Daylight' and (ref_cct is not None and ref_cct < 4000):
            warning_message = "CCT must be >= 4000 for Daylight Spectra"
        return {'margin-top': '20px'}, {'margin-top': '10px', 'display': 'inline-block'}, warning_message
    return {'display': 'none'}, {'display': 'none'}, ""

@app.callback(
    [Output('plotRef', 'figure'),
     Output('spectraRef', 'children'),
     Output('ssiText', 'children')],
    [Input('spectra-test-table', 'data'),
     Input('spectra-test-table', 'columns'),
     Input('testChoice', 'value'), 
     Input('refChoice', 'value'), 
     Input('stored-cct-value', 'data'),
     Input('submit-cct', 'n_clicks'), 
     Input('stored-custom-spec', 'data')],
    [State('refCct', 'value')]
)
def update_all_outputs(rows, columns, test_choice, ref_choice, stored_cct_value, n_clicks, custom_spec_data, ref_cct_value):
    fig = go.Figure()
    
    # Check for valid data in the DataTable
    if not rows or not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return fig, html.Div("Invalid data format."), "Spectral Similarity Index: N/A"
    
    test_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])  # Create DataFrame from DataTable rows

    # Plot test spectrum
    if test_choice == 'Custom' and custom_spec_data:
        fig.add_trace(go.Scatter(x=test_df['wavelength'], y=test_df['intensity'], mode='lines', name=f'Test Spectrum [CCT: {int(cct_mccamy(test_df))}]'))
    else:
        fig.add_trace(go.Scatter(x=test_df['wavelength'], y=test_df['intensity'], mode='lines', name=f'Test Spectrum [CCT: {int(cct_mccamy(test_df))}]'))

    # Update reference spectrum based on the CCT value
    if ref_choice == 'D50':
        df = D50_spec
    elif ref_choice == 'D55':
        df = D55_spec
    elif ref_choice == 'D65':
        df = D65_spec
    elif ref_choice == 'D75':
        df = D75_spec
    elif ref_choice == 'Custom_Blackbody':
        if n_clicks is not None:
            custom_spec_bb = planck(ref_cct_value, wavelengths)
            df = interpolate_and_normalize(custom_spec_bb)
        else:
            return fig, html.Div("Please submit a CCT value."), "Spectral Similarity Index: N/A"
    elif ref_choice == 'A':
        custom_spec_bb = planck(2855.542, wavelengths)
        df = interpolate_and_normalize(custom_spec_bb)
    elif ref_choice == 'Custom_Daylight':
        if n_clicks is not None:
            custom_spec_daylight = daylight(ref_cct_value, wavelengths)
            df = interpolate_and_normalize(custom_spec_daylight)
        else:
            return fig, html.Div("Please submit a CCT value."), "Spectral Similarity Index: N/A"
    elif ref_choice == 'Default':
        if cct_mccamy(test_df) < 4000:
            custom_spec_bb = planck(cct_mccamy(test_df), wavelengths)
            df = interpolate_and_normalize(custom_spec_bb)
        else: 
            custom_spec_daylight = daylight(cct_mccamy(test_df), wavelengths)
            df = interpolate_and_normalize(custom_spec_daylight)

    # if ref_choice == 'Custom_Blackbody' or ref_choice == 'Custom_Daylight':
    #     graph_cct_var = ref_cct_value
    # elif ref_choice == 'Default':
    #     graph_cct_var = CCT_MAPPING[test_choice]
    # else:
    #     graph_cct_var = CCT_MAPPING[ref_choice]

    fig.add_trace(go.Scatter(x=df['wavelength'], y=df['intensity'], mode='lines', name=f'Reference Spectrum [CCT: {int(cct_mccamy(df))}]'))

    table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            editable=False,  # Allows users to edit the data in the table
            sort_action="native",  # Allows sorting by columns
            page_action="native",  # Enables pagination
            page_size=1000,  # Number of rows per page
            export_format="csv",
            export_headers="display",
            merge_duplicate_headers=True
        )

    # SSI Calculation
    ref_data = df
    test_data = test_df  # Use the updated test_df from the DataTable

    if test_data is None or ref_data is None:
        ssi_value_text = "Spectral Similarity Index: N/A"
    else:
        test_intensity = test_data['intensity']
        ref_intensity = ref_data['intensity']
        test_wavelengths = test_data['wavelength']
        ref_wavelengths = ref_data['wavelength']

        ssi_value = calculate_ssi(test_wavelengths, test_intensity, ref_wavelengths, ref_intensity)
        ssi_value_text = f"Spectral Similarity Index: {int(ssi_value)}"

    return fig, table, ssi_value_text

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
