def main(args):
    setUpApp()
    return 0

def setUpApp():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Data Generation Inputs"),
                dbc.Label("Number of Modes per Class:"),
                dbc.Input(id="num_modes", type="number", min=1, value=1, step=1),
                
                dbc.Label("Number of Samples per Mode:"),
                dbc.Input(id="num_samples", type="number", min=1, value=100, step=1),

                dbc.Button("Generate Samples", id="generate_btn", color="primary", className="mt-3")
            ], width=3),
            
            dbc.Col([
                dcc.Graph(id="scatter_plot")  
            ], width=9) 
        ], align="center")
    ], fluid=True)

    @app.callback(
        Output('scatter_plot', 'figure'),
        [Input('generate_btn', 'n_clicks')],
        [dash.dependencies.State('num_modes', 'value'),
         dash.dependencies.State('num_samples', 'value')]
    )
    def update_plot(n_clicks, num_modes, num_samples):
        if n_clicks is None:
            return {}

        fig = create_plot(num_modes, num_samples)
        return fig
    
    app.run_server(debug=True)


def generate_class_data(num_modes, num_samples, class_label):
    x_data = []
    y_data = []
    labels = []

    for mode in range(num_modes):
        mean_x = np.random.uniform(-1, 1)
        mean_y = np.random.uniform(-1, 1)
        variance = np.random.uniform(0.05, 0.5)            

        x_mode = np.random.normal(mean_x, variance, num_samples)
        y_mode = np.random.normal(mean_y, variance, num_samples)
            
        x_data.extend(x_mode)
        y_data.extend(y_mode)
        labels.extend([class_label] * num_samples)
            
    return x_data, y_data, labels


def create_plot(num_modes, num_samples):
    x_class0, y_class0, labels_class0 = generate_class_data(num_modes, num_samples, class_label=0)
    
    x_class1, y_class1, labels_class1 = generate_class_data(num_modes, num_samples, class_label=1)

    x_data = x_class0 + x_class1
    y_data = y_class0 + y_class1
    labels = labels_class0 + labels_class1

    scatter = go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(
            size=5,
            color=labels,
            colorscale=['blue', 'red'],  # Blue for class 0, red for class 1
            showscale=False 
        )
    )
    
    layout = go.Layout(
        title="Generated Data Samples (2D Gaussian Modes)",
        xaxis=dict(title="X-axis"),
        yaxis=dict(title="Y-axis"),
        hovermode="closest"
    )
    
    return go.Figure(data=[scatter], layout=layout)

if __name__ == "__main__":
    import dash
    from dash import dcc, html, Input, Output
    import dash_bootstrap_components as dbc
    import numpy as np
    import plotly.graph_objs as go

    main(None)
