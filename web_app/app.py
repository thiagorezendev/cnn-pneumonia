# web_app/app.py

import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Necessário para o color_mode='rgb' no generator (se usar)
import numpy as np
import cv2 # Para manipulação de imagens (pip install opencv-python)
import imghdr # Para verificar o tipo de imagem

# --- Configurações da Aplicação ---
app = Flask(__name__)
# Define o diretório para upload de imagens
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Certifica que o diretório de uploads existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define o tamanho da imagem para o modelo (o mesmo que você usou no treinamento)
IMG_HEIGHT = 150 
IMG_WIDTH = 150  

# Mapeamento de classes (0 ou 1 para os nomes das categorias)
# Ex: {'NORMAL': 0, 'PNEUMONIA': 1} ou vice-versa
CLASS_NAMES = ['NORMAL', 'PNEUMONIA'] # Ajuste se o seu mapeamento for diferente

# --- Carregar o Modelo CNN Treinado ---
# Caminho para o modelo salvo (ajuste se a pasta web_app não estiver na raiz do seu projeto)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pneumonia_detection_model.keras')

# Função para carregar o modelo de forma segura
loaded_model = None
try:
    loaded_model = load_model(MODEL_PATH)
    print(f"Modelo carregado com sucesso de: {MODEL_PATH}")
except Exception as e:
    print(f"ERRO ao carregar o modelo: {e}")
    print(f"Verifique se o arquivo '{MODEL_PATH}' existe e está acessível.")

# --- Funções de Pré-processamento de Imagem para Inferência ---
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem em {image_path}")
    
    # Converter para RGB se necessário (OpenCV carrega em BGR por padrão)
    if len(img.shape) == 3 and img.shape[2] == 3: # Se já é 3 canais (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2: # Se for escala de cinza (2D)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # Converte para 3 canais RGB duplicando o canal cinza
    
    # Redimensionar a imagem para o tamanho esperado pelo modelo
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalizar os pixels (dividir por 255.0)
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Adicionar uma dimensão extra para representar o lote (batch)
    # De (IMG_HEIGHT, IMG_WIDTH, 3) para (1, IMG_HEIGHT, IMG_WIDTH, 3)
    img_input = np.expand_dims(img_normalized, axis=0)
    
    return img_input

# --- Rotas da Aplicação Flask ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None:
        return jsonify({'error': 'Modelo não carregado. Verifique os logs do servidor.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if file:
        # Salvar o arquivo temporariamente
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Verificar se é uma imagem válida
        file_type = imghdr.what(filepath)
        if not file_type: # imghdr.what retorna None se não for imagem
            os.remove(filepath) # Remove o arquivo inválido
            return jsonify({'error': 'Formato de arquivo não suportado. Por favor, envie uma imagem.'}), 400

        try:
            # Pré-processar a imagem para o modelo
            processed_image = preprocess_image(filepath)
            
            # Fazer a previsão
            prediction = loaded_model.predict(processed_image)
            
            # A saída é uma probabilidade de ser a classe '1' (Pneumonia)
            probability_pneumonia = prediction[0][0]
            
            # Classificar com base no limiar (0.5)
            if probability_pneumonia > 0.5:
                predicted_class_name = CLASS_NAMES[1] # 'PNEUMONIA'
            else:
                predicted_class_name = CLASS_NAMES[0] # 'NORMAL'
            
            # Retornar o resultado
            return jsonify({
                'prediction': predicted_class_name,
                'probability_pneumonia': float(probability_pneumonia), # Garante que seja um float serializável
                'image_url': f"/{filepath}" # Retorna o caminho da imagem para exibição no front-end
            })
            
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f"Erro no processamento da imagem ou previsão: {e}"}), 500
        finally:
            # Remover o arquivo após o processamento para limpar o servidor
            os.remove(filepath)

# --- Executar a Aplicação Flask ---
if __name__ == '__main__':
    # Usar 0.0.0.0 para que a aplicação seja acessível de outras máquinas na rede local
    # debug=True é útil durante o desenvolvimento para recarregar o servidor automaticamente
    app.run(host='0.0.0.0', port=5000, debug=True)