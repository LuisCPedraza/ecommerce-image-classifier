cat > README.md << 'EOF'
# Clasificador de Imágenes para E-commerce

## Descripción
Aplicación web que permite a administradores de tiendas en línea subir imágenes de productos y clasificarlas automáticamente en categorías predefinidas (e.g., "camisetas", "pantalones", "zapatos") usando un modelo de red neuronal basado en MobileNetV2 con transfer learning. Optimiza la gestión de inventario mediante integración con base de datos.

## Tecnologías
- **Backend:** Python con FastAPI
- **Frontend:** HTML/CSS/JavaScript
- **Base de Datos:** SQLite (desarrollo) / MySQL (producción)
- **IA:** TensorFlow/Keras con MobileNetV2
- **Control de Versiones:** GitHub

## Instalación y Configuración
1. Clona el repositorio:
   \`\`\`
   git clone https://github.com/LuisCPedraza/ecommerce-image-classifier.git
   cd ecommerce-image-classifier
   \`\`\`
2. Crea y activa entorno virtual:
   \`\`\`
   python -m venv venv
   source venv/bin/activate
   \`\`\`
3. Instala dependencias:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`
4. Configura variables de entorno (copia .env.example a .env y edita):
   \`\`\`
   cp .env.example .env
   \`\`\`
5. Para el dataset: Ver model/train.py para instrucciones.

## Uso
- Entrena el modelo: \`python model/train.py\`
- Ejecuta el backend: \`uvicorn backend.app.main:app --reload\`
- Abre frontend/index.html en un navegador.

### Frontend
- Abre http://localhost:8000/frontend/index.html.
- Sube imagen, ve predicción (categoría, confianza, ID, imagen).
- Lista de productos con imágenes y fechas.

### Endpoints API
- POST /predict/upload: Sube imagen, predice, guarda (multipart/form-data, file).
- GET /predict/{id}: Producto por ID.
- GET /predict/products: Lista todos.
- DELETE /predict/{id}: Elimina producto (opcional).

### Despliegue Local
- venv activo: `python backend/run.py`.
- Frontend: http://localhost:8000/frontend/index.html.
- API: http://localhost:8000/docs (Swagger).

### Despliegue Cloud Opcional
- **Backend (Heroku):** Instala CLI (`npm i -g heroku`), `heroku create ecommerce-classifier-api`, `git push heroku main`, set env vars (`heroku config:set DB_URL=sqlite:///app.db MODEL_PATH=model/saved_model/ecommerce_classifier.keras`).
- **Frontend (Vercel):** `npm i -g vercel`, cd frontend, `vercel` (set API_BASE = 'https://tu-app.herokuapp.com/predict').
- **DB Prod:** Cambia .env a MySQL (DB_URL=mysql://user:pass@localhost/db), rerun Alembic (`alembic upgrade head`).

### Pruebas End-to-End
- Sube imagen en frontend → Predicción (e.g., zapato → "Sneaker" ~90%).
- Lista actualiza con imagen visible.
- DELETE ID 1: curl -X DELETE http://localhost:8000/predict/1.

### Entrenamiento del Modelo
- Con venv activo: `cd model && python train.py`
- Genera `saved_model/ecommerce_classifier/` y `training_history.png`.
- Espera ~5-10 min en CPU (precisión esperada: ~90%+).

## Estructura del Proyecto
- /backend: API y lógica de negocio.
- /frontend: Interfaz de usuario.
- /model: Entrenamiento e inferencia IA.
- /docs: Documentación.

## Licencia
MIT License.
EOF
