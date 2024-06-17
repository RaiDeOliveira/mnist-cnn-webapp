from tensorflow.keras.optimizers import Adam
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

# Carregando o dataset MNIST e separando os dados de treino e de teste
(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

# Transformando os labels em one-hot encoding
y_treino_cat = to_categorical(y_treino)
y_teste_cat = to_categorical(y_teste)

# Normalização dos dados de entrada
x_treino_norm = x_treino / 255.0
x_teste_norm = x_teste / 255.0

# Reshape dos dados de entrada para adicionar o canal de cor
x_treino_norm = x_treino_norm.reshape(-1, 28, 28, 1)
x_teste_norm = x_teste_norm.reshape(-1, 28, 28, 1)

# Criação do modelo LeNet-5 ajustado
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # Adicionando dropout para regularização
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # Adicionando dropout para regularização
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Adicionando dropout para regularização
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilação do modelo
adam = Adam(learning_rate=0.001)  # Ajustando a taxa de aprendizado
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

# Treinamento do modelo
historico = model.fit(x_treino_norm, y_treino_cat, epochs=20, batch_size=128, validation_split=0.2)

# Exibição do histórico do treinamento
# Gráficos de acurácia
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.legend(['treino', 'validacao'])
plt.xlabel('épocas')
plt.ylabel('acurácia')

# Gráficos de perda
plt.subplot(1, 2, 2)
plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.legend(['treino', 'validacao'])
plt.xlabel('épocas')
plt.ylabel('perda')

plt.show()

# Salvando o modelo
model.save('pesos.h5')