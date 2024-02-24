import numpy as np #Biblioteca para operaciones numéricas
import matplotlib.pyplot as plt #Biblioteca  para graficar

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.random.rand(input_size + 1)#Inicializa los pesos aleatorios
        self.learning_rate = learning_rate #Tasa de aprendizaje
        self.epochs = epochs #Número de épocas de entrenamiento


    def train(self, X, y):
        for epoch in range(self.epochs): #Itera sobre el número de épocas
            for i in range(X.shape[0]): #Itera sobre cada muestra en el conjunto de entrenamiento
                X_with_bias = np.insert(X[i], 0, 1) #Inserta un sesgo en la entrada
                prediction = self.predict(X_with_bias)#Realiza una predicción
                error = y[i] - prediction #Calcula el error
                self.weights += self.learning_rate * error * X_with_bias #Actualiza los pesos

    def predict(self, X):
        activation = np.dot(X, self.weights) #Calcula la activación
        return 1 if activation >= 0 else -1 #Devuelve la clase predicha

def plot_data(X, y, perceptron=None):
    plt.scatter(X[:, 0], X[:, 1], c=y) #Grafica los puntos de datos
    if perceptron: #Grafica la línea que separa las clases si se proporciona un perceptrón entrenado
        plt.plot([-2, 2], [-(perceptron.weights[0] + perceptron.weights[1]*(-2)) / perceptron.weights[2],
                          -(perceptron.weights[0] + perceptron.weights[1]*(2)) / perceptron.weights[2]], 'r-')
    plt.xlabel('X1') #Eje x
    plt.ylabel('X2') #Eje y
    plt.title('Perceptron Separation') #Título de la ventana
    plt.show() #Muestra el gráfico

def read_data(file_path):
     #Lee los datos del archivo CSV y los separa en características (X) y etiquetas (y)
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def main():
    #Lectura de datos
    X_train, y_train = read_data('XOR_trn.csv')
    X_test, y_test = read_data('XOR_tst.csv')

    #Entrenamiento del perceptrón
    perceptron = Perceptron(input_size=X_train.shape[1], epochs=1000)
    perceptron.train(X_train, y_train)

    #Prueba del perceptrón
    correct_predictions = 0
    for i in range(X_test.shape[0]):
        X_with_bias = np.insert(X_test[i], 0, 1)
        prediction = perceptron.predict(X_with_bias)
        if prediction == y_test[i]:
            correct_predictions += 1
    accuracy = correct_predictions / X_test.shape[0]
    print("Accuracy:", accuracy)

    #Graficar datos de prueba y la separación del perceptrón
    plot_data(X_test, y_test, perceptron)

if __name__ == "__main__":
    main()
