# 🎯 Sistema de Identificación (ID) y Re-identificación (Re-ID) de Personas

Sistema inteligente en tiempo real y offline para **Identificación Facial (ID con HOG)** y **Re-identificación Corporal (Re-ID con LBP)** de personas. Combina detectores de aprendizaje profundo (YOLOv8) con descriptores geométricos y texturales clásicos (HOG 3,840-dims y LBP 1,888-dims), clasificadores SVM calibrados por probabilidades (Platt Scaling) y una interfaz gráfica profesional en PyQt6.

---

## 📌 Tabla de Contenidos
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Novedades y Optimización del Sistema](#-novedades-y-optimización-del-sistema)
- [Requisitos e Instalación](#-requisitos-e-instalación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Guía de Uso y Ejecución en 3 Pasos](#-guía-de-uso-y-ejecución-en-3-pasos)
  - [Paso 1: Entrenamiento de ID Facial](#paso-1-entrenamiento-de-id-facial)
  - [Paso 2: Inferencia en Tiempo Real y Captura Dinámica](#paso-2-inferencia-en-tiempo-real-y-captura-dinámica)
  - [Paso 3: Entrenamiento de Re-ID Corporal](#paso-3-entrenamiento-de-re-id-corporal)
- [Configuración (`config.yaml`)](#-configuración-configyaml)
- [Dashboard de Visualización (PyQt6)](#-dashboard-de-visualización-pyqt6)
- [Evaluación y Reportes](#-evaluación-y-reportes)

---

## 🧠 Arquitectura del Sistema

```
                         ┌─────────────────────────┐
                         │   Fotograma de Entrada   │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │ YOLOv8m (Person Detect) │
                         │ (Device: GPU / CUDA)    │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │ Face Visibility Router  │
                         └──────┬───────────┬──────┘
             (Rostro Válido)    │           │ (Rostro Borroso / Ocluido)
                   ┌────────────┘           └────────────┐
                   ▼                                     ▼
        ┌─────────────────────┐               ┌─────────────────────┐
        │   Rama Facial (ID)  │               │  Rama Cuerpo (ReID) │
        ├─────────────────────┤               ├─────────────────────┤
        │ - Alignment (Ojos)  │               │ - BBox Silueta 128x256│
        │ - Normalización 96x96│               │ - CLAHE Grises      │
        │ - LAB CLAHE Luminance│               │ - Grid 4x8 (32 blq) │
        │ - HOG (3,840 dims)  │               │ - LBP-U Fino (1,888d)│
        │ - Subspace Ensemble │               │ - Hellinger (L1-sqrt)│
        │ - Calibrated SVM    │               │ - Calibrated LinearSVC│
        └──────────┬──────────┘               └──────────┬──────────┘
                   └────────────┬────────────────────────┘
                                │
                         ┌──────▼──────┐
                         │   Decision  │
                         │   Engine    │
                         │(Gate T=0.75)│
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  Dashboard  │
                         │    PyQt6    │
                         └─────────────┘
```

---

## 🚀 Novedades y Optimización del Sistema

1. **Invariancia a la Distancia (Igual Prioridad Cerca y Lejos)**:
   - **Distancia Lejana**: Cobertura mejorada para detectar rostros lejanos desde **$18 \times 18\text{ px}$** y siluetas corporales desde **$14 \times 28\text{ px}$**. Aplica **interpolación bicúbica (`cv2.INTER_CUBIC`)** al reescalar a $96 \times 96$ BGR para reconstruir bordes nítidos de ojos y nariz en vectores HOG.
   - **Distancia Cercana**: Aplica **relleno por reflexión (`BORDER_REFLECT_101`)** en `face_normalizer.py` para evitar que recortes en primer plano pegados a los bordes del fotograma queden mutilados.

2. **Normalización Adaptativa de Iluminación y Sombras (CLAHE)**:
   - **Rama Facial (HOG)**: Conversión al espacio de color **LAB** y aplicación de **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) sobre el canal $L^*$ de luminancia antes de la extracción HOG.
   - **Rama Corporal (LBP)**: Aplicación de CLAHE sobre la escala de grises de la silueta del torso antes del cálculo de histogramas espaciales LBP-U.

3. **Aceleración por GPU (NVIDIA CUDA 12.4)**:
   - Inferencia súper veloz mediante PyTorch compilado con CUDA 12.4 y `device: "gpu"` en `config.yaml`.

4. **Control Estricto de Falsos Positivos**:
   - Umbral crítico de aceptación $T_{\text{aceptación}} = 0.75$ configurado en `config.yaml` e integrado directamente en `PipelineOrchestrator`.
   - **Filtro de Ambigüedad Top-1 vs Top-2**: Si el margen de probabilidad entre las dos mejores clases es menor al $12\%$, la predicción se marca como `"Desconocido"`.

---

## 📦 Requisitos e Instalación

### Requisitos Previos
- **Sistema Operativo**: Windows 10/11 (o Linux/macOS)
- **Python**: 3.10, 3.11 o 3.12 (64-bit)
- **GPU (Opcional pero recomendado)**: NVIDIA con soporte CUDA 12.4

### Instalación

1. **Navegar a la carpeta del proyecto**:
   ```powershell
   cd "C:\Users\Ariel\Documents\Programación\Sexto\ABC\Proyecto\ABC_Proyecto"
   ```

2. **Crear y activar el entorno virtual (`venv`)**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   *(Si PowerShell bloquea la ejecución de scripts, corre primero: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*.

3. **Instalar PyTorch con soporte para GPU CUDA 12.4**:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Instalar dependencias del proyecto**:
   ```powershell
   pip install -r requirements.txt
   pip install PyQt6 setuptools
   ```

5. **Verificar que la GPU esté activa**:
   ```powershell
   python -c "import torch; print('CUDA Activo:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
   ```

---

## 📁 Estructura del Proyecto

```
ABC_Proyecto/
├── branching/             # Ruteo dinámico entre rama Facial (ID) y Corporal (Re-ID)
├── classification/        # Ensamble de subespacios SVM y modelos serializados (.pkl)
├── core/                  # Lectura de video, scheduler y orquestador central
├── dataset/               # Almacenamiento de imágenes, capturas y modelos
│   ├── captures/          # Capturas dinámicas generadas en ejecución (hasta 75/sujeto)
│   ├── models/            # Modelos SVM entrenados (svm_facial y svm_reid)
│   ├── raw_images/        # Dataset base de imágenes por persona
│   └── raw_videos/        # Videos de prueba de cámaras de seguridad
├── decision_engine/       # Umbrales de aceptación, exclusión mutua y descarte
├── detection_tracking/    # Detectores YOLOv8m y trackers (DeepSORT)
├── dynamic_capture/       # Filtros de nitidez, resolución y guardado asíncrono
├── evaluation/            # Generación de matrices de confusión, curvas ROC y CMC
├── feature_extraction/    # Extracción HOG (Facial 96x96) y LBP-U (Corporal 128x256)
│   ├── body/              # Aislamiento de torso y mallas LBP 4x8
│   └── face/              # YuNet detector, normalizador afín, CLAHE y calidad facial
├── models/                # Pesos base ONNX/PT (e.g. face_detection_yunet.onnx)
├── preprocessing/         # Filtros de balance de blanco, CLAHE y mejora de imagen
├── reports/               # Auditorías y reportes de evaluación MLOps
├── retraining/            # Motor de aumentación de datos y validación cruzada
├── scripts/               # Scripts CLI para auditoría y entrenamiento SVM
│   ├── audit_face_dataset.py
│   ├── train_facial_svm.py
│   └── train_reid_svm.py
├── ui/                    # Interfaz gráfica de usuario en PyQt6 (Dashboard)
│   ├── dashboard.py       # Ventana principal del sistema
│   └── worker_thread.py   # Hilo de procesamiento de video en segundo plano
├── utils/                 # Cargador de config.yaml, logger e I/O de archivos
├── config.yaml            # Archivo de configuración centralizado
├── main.py                # Punto de entrada principal
└── requirements.txt       # Lista de dependencias de Python
```

---

## 🚀 Guía de Uso y Ejecución en 3 Pasos

### Paso 1: Entrenamiento de ID Facial (HOG)
Entrena el ensamble de subespacios HOG (Global, Superior, Inferior) sobre los rostros normalizados a $96 \times 96$ BGR de `dataset/raw_images/`:

```powershell
python scripts/train_facial_svm.py
```
> **Resultado**: Modelo guardado en `dataset/models/svm_facial/svm_facial_model.pkl`.

---

### Paso 2: Inferencia en Tiempo Real y Captura Dinámica
Lanza el Dashboard interactivo en PyQt6 para procesar los videos de `dataset/raw_videos/` o tu cámara web:

```powershell
python main.py --ui
```

> **¿Qué ocurre durante la ejecución?**
> 1. El sistema identifica a las personas en tiempo real usando el modelo facial.
> 2. Muestra cuadros **Verdes (`ID`)** para identificación facial y **Naranjas (`Re-ID`)** para silueta corporal.
> 3. El módulo de **Captura Dinámica** guarda automáticamente hasta 75 imágenes nítidas de cada sujeto en `dataset/captures/<Persona>/` marcadas con el icono 📸.

---

### Paso 3: Entrenamiento de Re-ID Corporal (LBP)
Una vez recolectadas las capturas en `dataset/captures/`, entrena el modelo de Re-identificación corporal basado en LBP-U sobre siluetas de $128 \times 256$:

```powershell
python scripts/train_reid_svm.py
```
> **Resultado**: Modelo definitivo guardado en `dataset/models/svm_reid/svm_reid_model.pkl`.

---

## ⚙️ Configuración (`config.yaml`)

El archivo **`config.yaml`** permite ajustar todos los parámetros operativos del sistema:

```yaml
video:
  source: "dataset/raw_videos/Video Camara 1.mov"  # Ruta del video o 0 para webcam
  target_fps: 30

detection:
  yolo_model_path: "yolov8m.pt"
  conf_threshold: 0.45                       # Umbral para detectar personas lejanas
  device: "gpu"                              # 'gpu' (CUDA) o 'cpu'

tracking:
  algorithm: "deepsort"                      # deepsort
  frame_rate: 30

face:
  yunet_model_path: "models/face_detection_yunet.onnx"
  conf_threshold: 0.75

decision:
  t_aceptacion: 0.75                         # Umbral de confianza crítica
  voting_decay: 0.98

capture:
  base_dir: "dataset/captures"
  max_captures_per_track: 75
  sharpness_threshold: 100.0

models:
  svm_facial_path: "dataset/models/svm_facial/svm_facial_model.pkl"
  svm_reid_path: "dataset/models/svm_reid/svm_reid_model.pkl"
```

---

## 🖥️ Dashboard de Visualización (PyQt6)

El Dashboard incluye:
- **Reproductor de Video Principal**: Muestra las cajas delimitadoras coloreadas por rama (Verde = ID Facial, Naranja = Re-ID Corporal, Rojo = Desconocido) con etiquetas de confianza e ícono 📸.
- **Tabla Live de Personas Activas**: Lista los `Track ID`, identidad asignada, nivel de confianza y rama utilizada en tiempo real.
- **Gráficos de Distribución**: Barras de progreso que muestran el porcentaje de cuadros procesados por la rama ID vs Re-ID.
- **Log de Eventos**: Consola con marcas de tiempo para nuevas detecciones, capturas almacenadas y avisos del sistema.
- **Controles Interactivos**: Botones para Pausar/Reanudar, Seleccionar cualquier Video desde el explorador o Activar la Cámara Web en vivo.
- **Badges de Estado**: Indicadores luminosos en la barra superior que confirman la carga de los modelos `SVM Facial` y `SVM Re-ID`.

---

## 📊 Evaluación y Reportes

El módulo de evaluación (`evaluation/`) genera reportes completos de rendimiento MLOps:
- **Matrices de Confusión**: Evaluaciones $(N+1) \times (N+1)$ incluyendo la clase de descarte `"Desconocido"`.
- **Curvas ROC & CMC**: Generación de curvas Receiver Operating Characteristic y Cumulative Match Characteristic (Rank-1 a Rank-5).
- **Reportes Exportables**: Gráficos e informes de precisión global almacenados en la carpeta `reports/`.
