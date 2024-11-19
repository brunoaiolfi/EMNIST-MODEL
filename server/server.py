from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
import imutils
from sklearn.preprocessing import LabelBinarizer
import joblib
from PIL import Image
from io import BytesIO
from flask_cors import cross_origin

LB = joblib.load('./lb.pkl')

app = Flask(__name__)

# Carregar o modelo TensorFlow
cnn_model = tf.keras.models.load_model('model.keras')  # Substitua 'model.h5' pelo seu modelo
svm_model = joblib.load('./svm.pkl')
lr_model = joblib.load('./logistic.pkl')
pca = joblib.load('./pca.pkl')

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def center_content(img):
    h, w = img.shape
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(img))
    centered = np.zeros_like(img)
    offset_x = (centered.shape[1] - w) // 2
    offset_y = (centered.shape[0] - h) // 2
    centered[offset_y:offset_y+h, offset_x:offset_x+w] = img[y:y+h, x:x+w]
    return centered


def correct_skew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def get_letters(image_bytes, model, LB, model_type="cnn", pca=None):
    """
    Função para processar uma imagem e realizar a classificação de caracteres.
    
    Parâmetros:
        img (str): Caminho para a imagem de entrada.
        model: Modelo de classificação (CNN, SVM ou Regressão Logística).
        LB: LabelBinarizer para decodificar as predições.
        model_type (str): Tipo de modelo ('cnn', 'svm', 'lr').
        pca: Modelo PCA para redução de dimensionalidade (necessário para SVM e LR).
        
    Retorna:
        letters (list): Lista de caracteres classificados.
        image: Imagem processada com contornos.
    """
    letters = []
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    # Encontrar contornos
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]

            # Pré-processamento comum
            roi = correct_skew(roi)
            roi = center_content(roi)
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0

            if model_type == "cnn":
                # Pré-processamento para CNN
                thresh = np.expand_dims(thresh, axis=-1)
                thresh = thresh.reshape(1, 32, 32, 1)
                ypred = model.predict(thresh)
                ypred = LB.inverse_transform(ypred)
                [x] = ypred
                letters.append(x)

            elif model_type in ["svm", "lr"]:
                # Achatar e aplicar PCA, se disponível
                thresh_flatten = thresh.flatten().reshape(1, -1)
                if pca:
                    thresh_flatten = pca.transform(thresh_flatten)
                ypred = model.predict(thresh_flatten)
                letters.append(ypred[0])

    return letters, image

def np_array_to_base64(np_array):
    # Converte o array NumPy para uma imagem PIL
    image = Image.fromarray(np_array)

    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    # Converte a imagem para uma string base64
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return img_str


def get_word(letter):
    word = "".join(letter)
    return word

@app.route('/predict/<model_type>', methods=['POST'])
@cross_origin()
def predict(model_type):
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    try:
        image_base64 = file.read()


        if model_type == "cnn":
            letter,image = get_letters(image_base64, cnn_model, LB, 'cnn')
        elif model_type == "svm":
            letter,image = get_letters(image_base64, svm_model, None, 'svm', pca)
        elif model_type == "lr":
            letter,image = get_letters(image_base64, lr_model, None, 'lr', pca)
        else:
            return jsonify({"error": "Invalid model type"}), 400

        
        word = get_word(letter)
        base64_image = np_array_to_base64(image)
        return jsonify({'prediction': word, 'image': base64_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
