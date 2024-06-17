from tensorflow.keras.models import load_model
import numpy as np
import cv2

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/input_cnn', methods=['GET', 'POST'])
def cnnInputPage():
    prediction = None
    if request.method == 'POST':
        if 'imgInput' not in request.files:
            return "Erro: Nenhum arquivo foi enviado."
        
        img_file = request.files['imgInput']
        
        if img_file.filename == '':
            return "Erro: Nenhum arquivo foi selecionado."
        
        # Salva a imagem enviada em um diretório temporário
        img_path = os.path.join('./temp', img_file.filename)
        img_file.save(img_path)
        
        # Carrega o modelo
        model_path = os.path.abspath('./pesos.h5')
        
        if not os.path.exists(model_path):
            return f"Erro: O arquivo de modelo não foi encontrado em {model_path}."
        
        try:
            modelo_2 = load_model(model_path)
        except Exception as e:
            return f"Erro ao carregar o modelo: {e}"
        
        # Carrega a imagem e converte para escala de cinza
        img = cv2.imread(img_path)
        
        if img is None:
            return "Erro: Falha ao carregar a imagem. Verifique se o arquivo é uma imagem válida."
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Redimensiona a imagem para o tamanho esperado pelo modelo
        img = cv2.resize(img, (28, 28))
        
        # Normaliza a imagem
        img = img / img.max()
        
        # Aplica um limiar binário à imagem
        _, img = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)

        # Prepara a imagem para a predição
        img = img.reshape(1, 28, 28, 1)
        
        # Realiza a predição
        predicao = modelo_2.predict(img)
        prediction = np.argmax(predicao)

    return render_template('input_cnn.html', prediction=prediction)


@app.route('/input_nn', methods=['GET', 'POST'])
def nnInputPage():
    prediction = None
    if request.method == 'POST':
        if 'imgInput' not in request.files:
            return "Erro: Nenhum arquivo foi enviado."
        
        img_file = request.files['imgInput']
        
        if img_file.filename == '':
            return "Erro: Nenhum arquivo foi selecionado."
        
        # Salva a imagem enviada em um diretório temporário
        img_path = os.path.join('./temp', img_file.filename)
        img_file.save(img_path)
        
        # Carrega o modelo
        model_path = os.path.abspath('./pesos_linear.h5')
        
        if not os.path.exists(model_path):
            return f"Erro: O arquivo de modelo não foi encontrado em {model_path}."
        
        try:
            modelo_2 = load_model(model_path)
        except Exception as e:
            return f"Erro ao carregar o modelo: {e}"
        
        # Carrega a imagem e converte para escala de cinza
        img = cv2.imread(img_path)
        
        if img is None:
            return "Erro: Falha ao carregar a imagem. Verifique se o arquivo é uma imagem válida."
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Redimensiona a imagem para o tamanho esperado pelo modelo
        img = cv2.resize(img, (28, 28))
        
        # Normaliza a imagem
        img = img / img.max()
        
        # Aplica um limiar binário à imagem
        _, img = cv2.threshold(img, img.mean(), 255, cv2.THRESH_BINARY)

        # Prepara a imagem para a predição
        img = img.reshape(1, 28, 28, 1)
        
        # Realiza a predição
        predicao = modelo_2.predict(img)
        prediction = np.argmax(predicao)

    return render_template('input_nn.html', prediction=prediction)

if __name__ == '__main__':
    os.makedirs('./temp', exist_ok=True)  # Cria o diretório temporário se não existir
    app.run('0.0.0.0', 8000, True)
