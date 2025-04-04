import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class PINN_NetworkTraffic_New:
    def __init__(self, layer_sizes, data_file, alpha=0.01, learning_rate=1e-3, rep=0):
        """
        layer_sizes: list defining network architecture, e.g., [1, 64, 64, 64, 1]
        data_file: path to the CSV file containing the raw network traffic data
        alpha: weight for the derivative loss term
        learning_rate: constant learning rate for the optimizer
        rep: random seed/replicate index
        """
        self.layer_sizes = layer_sizes
        self.data_file = data_file
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.rep = rep
        
        # Initialize network parameters and optimizer
        self.params = self.init_params()
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
        # Generate training data from CSV
        self.data_generations()
    
    def init_params(self):
        # Xavier initialization for each layer
        keys = jax.random.split(jax.random.PRNGKey(self.rep), len(self.layer_sizes) - 1)
        params = []
        for key, m, n in zip(keys, self.layer_sizes[:-1], self.layer_sizes[1:]):
            W = jax.random.normal(key, (m, n)) * jnp.sqrt(2 / m)
            b = jnp.zeros(n)
            params.append({'W': W, 'b': b})
        return params
    
    def data_generations(self):
        # Read CSV file with specified numeric type for 'ts'
        df = pd.read_csv(self.data_file, low_memory=False, dtype={'ts': np.float32})
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        df = df.dropna(subset=['ts'])
        
        # Aggregate data by rounding timestamp to nearest second
        df['ts_round'] = df['ts'].round(0)
        aggregated = df.groupby('ts_round').size().reset_index(name='flow_count')
        aggregated = aggregated.sort_values('ts_round')
        
        # Extract time and flow count values
        t_data = aggregated['ts_round'].values.astype(np.float32)
        Q_data = aggregated['flow_count'].values.astype(np.float32)
        t_min = t_data.min()
        t_data_norm = t_data - t_min  # normalize time
        
        # Normalize flow count by its maximum value and store Q_max
        self.Q_max = Q_data.max()
        Q_data_norm = Q_data / self.Q_max
        
        # Store training data as JAX arrays (shape: (N, 1))
        self.Ttrain = jnp.array(t_data_norm).reshape(-1, 1)
        self.u_train = jnp.array(Q_data_norm).reshape(-1, 1)
        
        # Compute numerical derivative with a fixed time step dt = 1 and normalize it
        dQ_dt = np.gradient(Q_data, 1)
        dQ_dt_norm = dQ_dt / self.Q_max
        self.u_train_deriv = jnp.array(dQ_dt_norm).reshape(-1, 1)
    
    def neural_net(self, t, params):
        # Forward pass: input t has shape (n, 1)
        X = t
        for layer in params[:-1]:
            X = jnp.tanh(jnp.dot(X, layer['W']) + layer['b'])
        last = params[-1]
        return jnp.dot(X, last['W']) + last['b']
    
    def loss(self, params):
        # Data MSE loss between network prediction and normalized training data
        u_pred = self.neural_net(self.Ttrain, params)
        data_loss = jnp.mean((u_pred - self.u_train)**2)
        
        # Compute the derivative of the network prediction for each sample
        def u_pred_scalar(t_val):
            t_val = jnp.array([[t_val]])
            return self.neural_net(t_val, params)[0, 0]
        pred_deriv = jax.vmap(jax.grad(u_pred_scalar))(self.Ttrain.flatten()).reshape(-1, 1)
        deriv_loss = jnp.mean((pred_deriv - self.u_train_deriv)**2)
        
        total_loss = data_loss + self.alpha * deriv_loss
        return total_loss
    
    def update(self):
        grads = jax.grad(self.loss)(self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
    
    def train(self, iterations):
        loss_history = []
        for i in tqdm(range(iterations)):
            self.update()
            if i % 1000 == 0:
                current_loss = self.loss(self.params)
                loss_history.append(current_loss)
                print(f"Iteration {i}, Loss: {current_loss}")
        return loss_history
    
    def predict(self, t):
        # Predict on input t (assumed shape (n,1))
        return self.neural_net(t, self.params)

# Example usage for network traffic analysis
if __name__ == '__main__':
    # Define network architecture: 1 input, 3 hidden layers with 64 neurons each, 1 output
    layer_sizes = [1, 64, 64, 64, 1]
    data_file = "phase1_NetworkData100k.csv"  # your CSV file
    alpha = 0.01         # weight for derivative loss (adjust as needed)
    learning_rate = 1e-3 # constant learning rate
    rep = 0
    iterations = 20000   # number of training iterations
    
    # Create and train the PINN model
    pinn_model = PINN_NetworkTraffic_New(layer_sizes, data_file, alpha, learning_rate, rep)
    loss_history = pinn_model.train(iterations)
    
    # Prepare prediction: create test time points
    t_train = np.array(pinn_model.Ttrain).flatten()
    Q_train = np.array(pinn_model.u_train).flatten() * pinn_model.Q_max  # denormalize training data
    t_test = np.linspace(t_train.min(), t_train.max(), 200).reshape(-1, 1)
    Q_pred_norm = np.array(pinn_model.predict(jnp.array(t_test))).flatten()
    Q_pred = Q_pred_norm * pinn_model.Q_max  # denormalize predictions
    
    # Plot aggregated training data vs. PINN prediction
    plt.figure(figsize=(10,6))
    plt.scatter(t_train, Q_train, color='blue', s=10, label="Aggregated Data")
    plt.plot(t_test, Q_pred, color='red', linewidth=2, label="PINN Prediction")
    plt.xlabel("Time (s)")
    plt.ylabel("Flow Count Q(t)")
    plt.title("PINN Prediction vs. Aggregated Network Traffic")
    plt.legend()
    plt.show()
    
    # Plot training loss history
    plt.figure(figsize=(10,6))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Iteration (every 1000 iterations)")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.legend()
    plt.show()
