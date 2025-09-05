# SignMicroservice

Microservicio de **IA** para procesamiento y análisis de imágenes, diseñado para ser ligero y ejecutarse con **FastAPI**.  
Incluye soporte para modelos de **PyTorch (CPU)** y herramientas de visión por computadora.

---

## ��� Requisitos

- Python 3.10 o superior  
- pip actualizado  
- Virtualenv  

---

## ⚙️ Instalación

Clona el repositorio y entra en el directorio:

```bash
git clone https://github.com/tu-usuario/SignMicroservice.git
cd SignMicroservice
```

Crea y activa el entorno virtual, luego instala dependencias:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio 2>/dev/null
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install fastapi "uvicorn[standard]" pillow numpy opencv-python-headless python-multipart
```

