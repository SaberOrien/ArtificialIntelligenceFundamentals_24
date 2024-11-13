import numpy as np

class ShallowNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size=2, learning_rate=0.01, activation='sigmoid'):
        #create network layers (input -> hidden -> output)
        self.learning_rate = learning_rate
        self.activation = activation
        
        self.layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(self.initialize_layer(prev_size, hidden_size))
            prev_size = hidden_size
        
        #output layer
        self.layers.append(self.initialize_layer(prev_size, output_size))

    def initialize_layer(self, input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        weights = np.random.uniform(-limit, limit, size=(input_size, output_size))
        biases = np.zeros(output_size)
        return {'weights': weights, 'biases': biases}

    def activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'heaviside':
            return np.where(x >= 0, 1, 0)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sin':
            return np.sin(x)
        elif self.activation == 'sign':
            return np.sign(x)

    def activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
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
            return 1

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  #-max to prevent overflow
        return exp_x / np.sum(exp_x, axis=0)

    def forward(self, inputs):
        activations = inputs
        for layer in self.layers[:-1]:
            z = np.dot(activations, layer['weights']) + layer['biases']
            activations = self.activate(z)
            layer['activations'] = activations
            layer['z'] = z
        
        #softmax for output
        output_layer = self.layers[-1]
        z = np.dot(activations, output_layer['weights']) + output_layer['biases']
        activations = self.softmax(z)
        output_layer['activations'] = activations
        output_layer['z'] = z
        return activations

    def backward(self, inputs, expected_output):
        deltas = []
        output = self.layers[-1]['activations']
        error = expected_output - output
        
        delta = error * self.activation_derivative(output)  #output layer delta
        deltas.append(delta)

        #hidden layers
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            delta = np.dot(deltas[-1], next_layer['weights'].T) * self.activation_derivative(layer['activations'])
            deltas.append(delta)

        deltas.reverse()    #to match layer order input -> output

        activation = inputs
        for i in range(len(self.layers)):
            layer = self.layers[i]
            delta = deltas[i]

            layer['weights'] += self.learning_rate * np.outer(activation, delta)
            layer['biases'] += self.learning_rate * np.sum(delta, axis=0)
            
            activation = layer['activations']  #activation to the next layer

    def train(self, inputs, expected_output, epochs=1000):
        for epoch in range(epochs):
            for x, d in zip(inputs, expected_output):
                self.forward(x)
                self.backward(x, d)
            print(epoch)

def create_plot(num_modes, num_samples, activation_function, num_hidden_layers, neurons_per_layer):
    x_class0, y_class0, labels_class0 = generate_class_data(num_modes, num_samples, class_label=0, activation_function=activation_function)
    x_class1, y_class1, labels_class1 = generate_class_data(num_modes, num_samples, class_label=1, activation_function=activation_function)

    x_data = np.column_stack((x_class0 + x_class1, y_class0 + y_class1))
    labels = np.array(labels_class0 + labels_class1)



    hidden_layers = [neurons_per_layer] * num_hidden_layers
    nn = ShallowNeuralNetwork(input_size=2, hidden_layers=hidden_layers, output_size=2, activation=activation_function)
    nn.train(x_data, labels, epochs=500)



    xx, yy = np.meshgrid(np.linspace(min(x_class0 + x_class1), max(x_class0 + x_class1), 200),
                         np.linspace(min(y_class0 + y_class1), max(y_class0 + y_class1), 200))
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    zz = np.array([nn.forward(point)[1] for point in grid_points])
    #print(f"Min zz: {zz.min()}, Max zz: {zz.max()}")
    zz = zz.reshape(xx.shape)

    scatter = go.Scatter(
        x=x_class0 + x_class1,
        y=y_class0 + y_class1,
        mode='markers',
        marker=dict(
            size=5,
            color=labels.argmax(axis=1), 
            colorscale=['red', 'blue'],
            showscale=False
        )
    )

    boundary = go.Contour(
        x=np.linspace(min(x_class0 + x_class1), max(x_class0 + x_class1), 200),
        y=np.linspace(min(y_class0 + y_class1), max(y_class0 + y_class1), 200),
        z=zz,
        showscale=True, 
        opacity=0.8,
        colorscale='RdBu',
        contours=dict(
            start=0, end=1, size=0.05, coloring='fill'
        ),
        line=dict(width=0)
    )

    layout = go.Layout(
        title="Generated Data Samples with Neural Network Decision Boundary",
        xaxis=dict(
            title="X-axis",
            range=[min(x_class0 + x_class1) - 0.1, max(x_class0 + x_class1) + 0.1], 
        ),
        yaxis=dict(
            title="Y-axis",
            range=[min(y_class0 + y_class1) - 0.1, max(y_class0 + y_class1) + 0.1], 
        ),
        hovermode="closest"
    )

    return go.Figure(data=[scatter, boundary], layout=layout)

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
                dbc.Input(id="num_modes", type="number", min=1, value=3, step=1),

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

                dbc.Label("Number of Hidden Layers:"),
                dbc.Input(id="num_hidden_layers", type="number", min=1, value=2, max=4, step=1),

                dbc.Label("Neurons per Hidden Layer:"),
                dbc.Input(id="neurons_per_layer", type="number", min=1, value=5, step=1),

                dbc.Button("Generate Samples", id="generate_btn", color="primary", className="mt-s3")
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
         dash.dependencies.State('activation_function', 'value'),
         dash.dependencies.State('num_hidden_layers', 'value'),
         dash.dependencies.State('neurons_per_layer', 'value')]
    )
    def update_plot(n_clicks, num_modes, num_samples, activation_function, num_hidden_layers, neurons_per_layer):
        if n_clicks is None:
            return {}

        fig = create_plot(num_modes, num_samples, activation_function, num_hidden_layers, neurons_per_layer)
        return fig
    
    app.run_server(debug=True)

def generate_class_data(num_modes, num_samples, class_label, activation_function):
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

        if activation_function in ['tanh', 'sin', 'sign']:
            if class_label == 0:
                labels.extend([[-1, 1]] * num_samples)
            else:
                labels.extend([[1, -1]] * num_samples)
        else:
            if class_label == 0:
                labels.extend([[1, 0]] * num_samples)
            else:
                labels.extend([[0, 1]] * num_samples)

    return x_data, y_data, labels

if __name__ == "__main__":
    import dash
    from dash import dcc, html, Input, Output
    import dash_bootstrap_components as dbc
    import numpy as np
    import plotly.graph_objs as go

    main(None)
