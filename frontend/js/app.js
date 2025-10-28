const API_BASE = '/predict';  // Proxy o localhost:8000 en prod

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = document.getElementById('imageFile').files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            document.getElementById('result').innerHTML = `
                <h3>Predicción:</h3>
                <p>Categoría: <strong>${data.category}</strong></p>
                <p>Confianza: <span class="confidence">${data.confidence}</span></p>
                <p>ID: ${data.product_id}</p>
                <img src="${data.image_path}" alt="Producto" style="max-width: 200px;">
            `;
            loadProducts();  // Recarga lista
        } else {
            alert('Error: ' + data.detail || 'Upload falló');
        }
    } catch (error) {
        alert('Error de red: ' + error);
    }
});

async function loadProducts() {
    try {
        const response = await fetch(`${API_BASE}/products`);
        const products = await response.json();
        const listDiv = document.getElementById('productsList');
        listDiv.innerHTML = products.map(p => `
            <div class="product-card">
                <div>
                    <strong>${p.category}</strong> (ID: ${p.id})
                    <br>Fecha: ${p.created_at}
                </div>
                <div>
                    <img src="${p.image_path}" alt="${p.category}" style="max-width: 100px;">
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error cargando productos:', error);
    }
}

// Carga inicial
loadProducts();
