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

## Estructura del Proyecto
- /backend: API y lógica de negocio.
- /frontend: Interfaz de usuario.
- /model: Entrenamiento e inferencia IA.
- /docs: Documentación.

## Licencia
MIT License.
EOF
