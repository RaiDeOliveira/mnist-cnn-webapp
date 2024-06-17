# Aplicação web para detecção de algarismos

Esse repositório contém uma aplicação web que permite que o usuário faça upload de imagens para serem analisadas por modelos de redes neurais, a fim de detectar algarismos numéricos contidos nelas. O usuário pode optar por uma análise feita por uma Rede Neural Convolucional ou por uma Rede Neural Linear.

## Vídeos de demonstração

### Aplicação web

Para conferir um vídeo de demonstração da aplicação web, [clique aqui](google.com). Nesse vídeo o usuário abre uma janela de terminal e executa a aplicação web de acordo com o [passo 3](#executar-aplicação-web) da seção de instalação e execução. Depois de acessar o endpoint `/input_cnn`, ele envia uma imagem com o algarismo **3** para análise da Rede Neural Convolucional e obtém o número detectado por ela na parte inferior da página. O usuário repete o processo após clicar no botão de troca para a Rede Neural Linear, analisando a mesma imagem e obtendo o mesmo resultado.

### Treinamento dos modelos

## Instalação e execução

### Pré-requisitos

- Git instalado
- Python 3 e pip instalados

### Passo a passo

#### Instalação

1. Abra uma janela de terminal no diretório de sua preferência, clone o repositório e adentre o diretório recém criado através dos seguintes comandos:

```bash
git https://github.com/RaiDeOliveira/mnist-cnn-webapp.git
cd mnist-cnn-webapp
```

2. Na mesma janela de terminal, digite os seguintes comandos para instalar as dependências necessárias para rodar o projeto:

> :bulb: **Dica:** Se você preferir, crie e ative um ambiente virtual em Python através sos seguintes comandos antes dessa etapa:
> `python3 -m venv venv`
> `source venv/bin/activate` (comando específico para SO Linux)

```bash
pip install flask tensorflow keras matplotlib numpy np_utils opencv-python
```

#### Executar aplicação web

3. Para executar a aplicação web, digite o seguinte comando na mesma janela de terminal:

```bash
cd web-app
flask --app app run
```

#### Treinar modelo CNN

4. Para treinar o modelo de Rede Neural Convolucional, digite o seguinte comando a partir da raiz do diretório com o conteúdo do repositório:

```bash
python3 cnn.py
```

#### Treinar modelo LNN

5. Para treinar o modelo de Rede Neural Linear, digite o seguinte comando a partir da raiz do diretório com o conteúdo do repositório:

```bash
python3 lnn.py
```
