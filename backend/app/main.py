from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.predict import router as predict_router
from .database import Base, engine
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

load_dotenv()

# Crear tablas DB
Base.metadata.create_all(bind=engine)

app = FastAPI(title="E-commerce Image Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="../frontend"), name="frontend")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.include_router(predict_router)

@app.get("/")
def root():
    return {"message": "API Clasificador de Imágenes E-commerce - ¡Listo!"}
