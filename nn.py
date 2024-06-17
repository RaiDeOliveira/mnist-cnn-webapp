import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Carregue os dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize os valores dos pixels para o intervalo [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Converta as labels para one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Construa o modelo de rede neural linear
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten a imagem 28x28 para um vetor de 784 elementos
    Dense(10, activation='softmax')  # Camada densa com 10 neurônios para 10 classes de dígitos
])

# Compile o modelo
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Treine o modelo
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Avalie o modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')

# Salve o modelo para um arquivo .h5
model.save('pesos_linear.h5')
