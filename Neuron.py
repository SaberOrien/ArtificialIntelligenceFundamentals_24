class Neuron:
    def __init__(self, input_size, learning_rate=0.01, activation='sigmoid'):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.activation = activation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def heaviside(self, x):
        return np.where(x >= 0, 1, 0)

    def relu(self, x):
        return np.maximum(0, x)

    def leaky_relu(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def tanh(self, x):
        return np.tanh(x)

    def sin(self, x):
        return np.sin(x)

    def sign(self, x):
        return np.sign(x)

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        self.z = z  
        if self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'heaviside':
            return self.heaviside(z)
        elif self.activation == 'relu':
            return self.relu(z)
        elif self.activation == 'leaky_relu':
            return self.leaky_relu(z)
        elif self.activation == 'tanh':
            return self.tanh(z)
        elif self.activation == 'sin':
            return self.sin(z)
        elif self.activation == 'sign':
            return self.sign(z)

    def activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation == 'heaviside':
            return 1
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'sin':
            return np.cos(x)
        elif self.activation == 'sign':
            return 0

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, expected_output, epochs=100):
        for epoch in range(epochs):
            for x, d in zip(inputs, expected_output):
                y = self.forward(x)
                error = d - y
                activation_deriv = self.activation_derivative(self.z)
                gradient = error * activation_deriv
                self.weights += self.learning_rate * gradient * x
                self.bias += self.learning_rate * gradient

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

                dbc.Label("Activation Function:"),
                dcc.Dropdown(
                    id="activation_function",
                    options=[
                        {'label': 'Sigmoid', 'value': 'sigmoid'},
                        {'label': 'Heaviside', 'value': 'heaviside'},
                        {'label': 'ReLU', 'value': 'relu'},
                        {'label': 'Leaky ReLU', 'value': 'leaky_relu'},
                        {'label': 'Tanh', 'value': 'tanh'},
                        {'label': 'Sin', 'value': 'sin'},
                        {'label': 'Sign', 'value': 'sign'}
                    ],
                    value='sigmoid'
                ),

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
         dash.dependencies.State('num_samples', 'value'),
         dash.dependencies.State('activation_function', 'value')]
    )
    def update_plot(n_clicks, num_modes, num_samples, activation_function):
        if n_clicks is None:
            return {}

        fig = create_plot(num_modes, num_samples, activation_function)
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

def create_plot(num_modes, num_samples, activation_function):
    x_class0, y_class0, labels_class0 = generate_class_data(num_modes, num_samples, class_label=0)
    x_class1, y_class1, labels_class1 = generate_class_data(num_modes, num_samples, class_label=1)

    x_data = np.column_stack((x_class0 + x_class1, y_class0 + y_class1))
    labels = labels_class0 + labels_class1

    neuron = Neuron(input_size=2, activation=activation_function)
    neuron.train(x_data, labels)

    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    zz = np.array([neuron.forward(point) for point in grid_points])
    zz = zz.reshape(xx.shape)
    
    scatter = go.Scatter(
        x=x_class0 + x_class1,
        y=y_class0 + y_class1,
        mode='markers',
        marker=dict(
            size=5,
            color=labels,
            colorscale=['red', 'blue'],
            showscale=False  
        )
    )

    boundary = go.Contour(
        x=np.linspace(-1.5, 1.5, 200),
        y=np.linspace(-1.5, 1.5, 200),
        z=zz,
        showscale=False,
        opacity=0.7,
        colorscale='RdBu',
        contours=dict(start=0, end=1, size=0.05, coloring='fill'), 
        line=dict(width=2, color='rgba(0,0,0,0)')
    )

    layout = go.Layout(
        title="Generated Data Samples (2D Gaussian Modes) with Probability-based Decision Boundary",
        xaxis=dict(
            title="X-axis",
            range=[-1.5, 1.5],
        ),
        yaxis=dict(
            title="Y-axis",
            range=[-1.5, 1.5],
        ),
        hovermode="closest"
    )

    return go.Figure(data=[scatter, boundary], layout=layout)


if __name__ == "__main__":
    import dash
    from dash import dcc, html, Input, Output
    import dash_bootstrap_components as dbc
    import numpy as np
    import plotly.graph_objs as go

    main(None)
