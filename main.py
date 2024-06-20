import numpy as np

# Simulação de um modelo simples de regressão linear
class SimpleLinearModel:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = np.random.rand(3)  # Três características de entrada
        else:
            self.weights = weights
    
    def predict(self, x):
        return np.dot(x, self.weights)
    
    def gradient(self, x, y):
        prediction = self.predict(x)
        error = prediction - y
        grad = error * x
        return grad

# Função de treinamento local
def train_local_model(model, data, epochs, learning_rate):
    for epoch in range(epochs):
        for x, y in data:
            grad = model.gradient(x, y)
            model.weights -= learning_rate * grad
    return model.weights

# Função de agregação de modelos
def federated_averaging(models):
    avg_weights = np.mean([model.weights for model in models], axis=0)
    return avg_weights

# Inicialização do modelo global
global_model = SimpleLinearModel()

# Dados distribuídos em clientes (simulados)
clients_data = [
    [(np.array([1.0, 2.0, 3.0]), 7.0), (np.array([2.0, 3.0, 4.0]), 8.0)],
    [(np.array([1.5, 2.5, 3.5]), 6.0), (np.array([2.5, 3.5, 4.5]), 7.0)],
    [(np.array([1.2, 2.2, 3.2]), 5.0), (np.array([2.2, 3.2, 4.2]), 6.0)]
]

# Parâmetros de treinamento
num_rounds = 5
epochs = 10
learning_rate = 0.01

# Treinamento federado
for round in range(num_rounds):
    local_models = []
    
    for data in clients_data:
        local_model = SimpleLinearModel(global_model.weights.copy())
        local_weights = train_local_model(local_model, data, epochs, learning_rate)
        local_models.append(local_model)
    
    # Agregação dos modelos locais
    global_weights = federated_averaging(local_models)
    global_model.weights = global_weights

# Modelo global final treinado
trained_global_model = global_model
print("Pesos finais do modelo global:", trained_global_model.weights)
