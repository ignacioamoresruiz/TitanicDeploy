from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Crear la app de Flask
app = Flask(__name__)

# Cargar el modelo previamente entrenado
model = pickle.load(open('/workspaces/TitanicDeploy/Model/gbk_model.pkl', 'rb'))
model_svc = pickle.load(open('/workspaces/TitanicDeploy/Model/svc_model.pkl', 'rb'))

# Ruta principal para mostrar el formulario
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar el formulario y hacer la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Capturar los valores del formulario
    pclass = int(request.form['pclass'])
    sex = request.form['sex']
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    agegroup = float(request.form['agegroup'])  # Asegurarse de que sea float
    embarked = request.form['embarked']
    cabinbool = int(request.form['cabinbool'])
    title = int(request.form['title'])
    fareband = int(request.form['fareband'])  # FareBand debe ser int (aunque sea categórico en pandas)
    model_selected = int(request.form['modelo'])

    # Convertir sexo a binario
    sex_binary = 1 if sex == 'female' else 0

    # Mapeo de embarked a valores numéricos
    embarked_map = {'S': 1, 'C': 2, 'Q': 3}
    embarked_encoded = embarked_map.get(embarked, 1)  # Por defecto 'S'

    # Crear el array de entrada para el modelo (verificar que el orden sea correcto)
    input_features = np.array([[pclass, sex_binary, sibsp, parch, 
                                embarked_encoded, agegroup, cabinbool, 
                                title, fareband]])

    # Seleccionar el modelo basado en la opción del usuario
    if model_selected == 1:
        # Usar Gradient Boosting Classifier
        probas = model.predict_proba(input_features)
    elif model_selected == 2:
        # Usar Support Vector Machine (SVC)
        probas = model_svc.predict_proba(input_features)

    # Mostrar ambas probabilidades
    prob_fallecer = probas[0][0]  # Probabilidad de fallecer (clase 0)
    prob_supervivir = probas[0][1]  # Probabilidad de sobrevivir (clase 1)

    # Convertir a porcentaje
    survival_rate = round(prob_supervivir * 100, 2)

    # Determinar el mensaje y la imagen basado en la probabilidad de supervivencia
    if survival_rate > 75:
        message = "¡Felicidades! Tu probabilidad de supervivencia es bastante alta."
        image_url = "img/high_survival.png"  # Imagen para alta probabilidad
    elif survival_rate > 50:
        message = "Tienes una buena probabilidad de sobrevivir."
        image_url = "img/medium_survival.png"  # Imagen para probabilidad media
    elif survival_rate > 25:
        message = "Tu probabilidad de supervivencia es baja."
        image_url = "img/low_survival.png"  # Imagen para baja probabilidad
    else:
        message = "Lamentablemente, tu probabilidad de supervivencia es muy baja."
        image_url = "img/very_low_survival.png"  # Imagen para muy baja probabilidad

    # Renderizar la página de resultados
    return render_template('result.html', survival_rate=survival_rate, message=message, image_url=image_url)

if __name__ == '__main__':
    # Obtener el puerto del entorno de Google Cloud Run o usar el 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
