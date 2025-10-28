import sys
import os
# Agrega root (/ecommerce-image-classifier) al PYTHONPATH para ver 'model'
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.product import Product
import shutil
from datetime import datetime

from model.predict import predict_image  # Ahora se ve desde root

router = APIRouter(prefix="/predict", tags=["predict"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Solo imágenes PNG/JPG")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    image_path = os.path.join(UPLOAD_DIR, filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predicción
    category, confidence = predict_image(image_path)

    # Guardar en DB
    db_product = Product(category=category, image_path=image_path)
    db.add(db_product)
    db.commit()
    db.refresh(db_product)

    return {"product_id": db_product.id, "category": category, "confidence": f"{confidence:.2f}%", "image_path": image_path}

@router.get("/products")
def list_products(db: Session = Depends(get_db)):
    products = db.query(Product).all()
    return [{"id": p.id, "category": p.category, "image_path": p.image_path, "created_at": p.created_at.isoformat()} for p in products]

@router.get("/{product_id}")
def get_prediction(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    return {"id": product.id, "category": product.category, "image_path": product.image_path}


