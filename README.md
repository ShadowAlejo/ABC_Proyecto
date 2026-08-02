# 🎯 Sistema de Identificación (ID) y Re-identificación (Re-ID) de Personas

Sistema inteligente en tiempo real y offline para **Identificación Facial (ID)** y **Re-identificación Corporal (Re-ID)** de personas. Combina detectores modernos de aprendizaje profundo para detección y seguimiento con descriptores geométricos/texturales clásicos y clasificadores SVM de alto rendimiento.

---

## 📌 Tabla de Contenidos
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Requisitos e Instalación](#-requisitos-e-instalación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Guía de Uso y Ejecución](#-guía-de-uso-y-ejecución)
  - [1. Auditoría del Dataset](#1-auditoría-del-dataset)
  - [2. Entrenamiento de Modelos SVM](#2-entrenamiento-de-modelos-svm)
  - [3. Ejecución del Pipeline](#3-ejecución-del-pipeline)
- [Configuración (`config.yaml`)](#-configuración-configyaml)
- [Dashboard de Visualización (PyQt6)](#-dashboard-de-visualización-pyqt6)
- [Evaluación y MLOps](#-evaluación-y-mlops)

---

## 🧠 Arquitectura del Sistema

```
                         ┌─────────────────────────┐
                         │   Fotograma de Entrada   │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │ YOLOv8n (Person Detect) │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │ Face Visibility Router  │
                         └──────┬───────────┬──────┘
       (YoloFace Conf ≥ 0.60)   │           │ (YoloFace Conf < 0.60)
                  ┌─────────────┘           └─────────────┐
                  ▼                                       ▼
       ┌─────────────────────┐                 ┌─────────────────────┐
       │   Rama Facial (ID)  │                 │  Rama Cuerpo (ReID) │
       ├─────────────────────┤                 ├─────────────────────┤
       │ - YoloFace landmarks│                 │ - BBox Silueta 128x256│
       │ - CLAHE + Unsharp   │                 │ - Grid 4x8 (32 bloques)│
       │ - Descriptor HOG    │                 │ - LBP-U Fino (R=1, P=8) │
       │ - Calibrated SVM    │                 │ - Hellinger (L1-sqrt)│
       │ - (Out-of-Core)     │                 │ - Calibrated LinearSVC│
       └──────────┬──────────┘                 └──────────┬──────────┘
                  └─────────────┬─────────────────────────┘
                                │
                         ┌──────▼──────┐
                         │   Decision  │
                         │   Engine    │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  Dashboard  │
                         │    PyQt6    │
                         └─────────────┘
```

### Componentes Clave:
1. **Detección Stateless**:
   - **YOLOv8n**: Detecta cajas delimitadoras de personas (`person`, confianza $\ge 0.40$).
   - **ID Efímeros**: Asignación instantánea de identificadores sin inercia temporal (`F{frame}_D{idx}`).
2. **Ruteo Dinámico y Extraición (`face_visibility_router`)**:
   - **Rama ID (Facial)**: Activada cuando YoloFace detecta rostro con confianza $\ge 0.60$. Extrae descriptores HOG con normalización iterativa y alineación biométrica de 5 landmarks. El entrenamiento subyacente maneja +100,000 vectores usando **Out-of-Core Processing (`np.memmap`)**. El clasificador es un `CalibratedClassifierCV` que provee Platt Scaling perfecto.
   - **Rama Re-ID (Corporal)**: Fallback automático y robusto al cambio de ropa. Mantiene la silueta completa a $128 \times 256$, divide en 32 bloques y extrae patrones LBP-U aplicando **Normalización Hellinger (L1-sqrt)**. Su vector comprimido de **1,888 dimensiones** permite entrenar un SVC lineal súper veloz 100% en memoria.
3. **Motor de Decisión (`decision_engine`)**:
   - **`ThresholdAcceptanceGate`**: Umbral crítico de aceptación ($T = 0.65$).
   - **`Exclusión Mutua`**: Resuelve colisiones intra-fotograma asegurando que cada identidad solo pueda ser asignada a la detección de mayor confianza.
4. **Captura Dinámica (`dynamic_capture`)**:
   - Filtra y guarda imágenes de alta calidad (hasta 75 por ID) evaluando resolución, nitidez Laplaciana, intervalo temporal y desplazamiento postural.

---

## 📦 Requisitos e Instalación

### Requisitos Previos
- **Python**: 3.10, 3.11 o 3.12 (64-bit)
- **Git**

### Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <URL_DEL_REPOSITO>
   cd id_reid_system
   ```

2. **Crear y activar el entorno virtual**:
   ```bash
   # En Windows:
   python -m venv venv
   .\venv\Scripts\activate

   # En Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias requeridas**:
   ```bash
   pip install -r requirements.txt
   pip install PyQt6 lapx
   pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git --no-deps
   ```

> ⚠️ **Nota sobre Modelos y Caché**: Por diseño, este repositorio omite en su control de versiones (`.gitignore`) todos los archivos temporales, carpetas de caché (`__pycache__`) y modelos pesados preentrenados/entrenados (como `yolov8n.pt` o los generados localmente `.pkl`). YOLOv8 se descarga automáticamente al primer uso y los modelos SVM de identidad deben ser entrenados mediante los scripts correspondientes.


---

## 📁 Estructura del Proyecto

```
id_reid_system/
├── branching/             # Ruteo entre rama Facial (ID) y Corporal (Re-ID)
├── classification/        # Modelos SVM serializados y convertidores de confianza
├── core/                  # Ingesta de video, despachador de frames y orquestador
├── dataset/               # Estructura de almacenamiento de datos
│   ├── captures/          # Capturas dinámicas generadas en ejecución
│   ├── models/            # Modelos SVM entrenados (.pkl)
│   ├── raw_images/        # Imágenes de entrenamiento por sujeto
│   └── raw_videos/        # Videos de entrada para inferencia
├── decision_engine/       # Votación temporal, umbral de aceptación e inercia
├── detection_tracking/    # Detectores YOLOv8n
├── dynamic_capture/       # Filtros de calidad para recolección de muestras
├── evaluation/            # Generadores de matrices de confusión, ROC y CMC
├── feature_extraction/    # Extracción de características HOG (cara) y LBP-U (cuerpo)
│   ├── body/              # Torso isolator, masking sigmoideo, LBP multi-escala
│   └── face/              # YoloFace detector, normalizador, HOG, calidad facial
├── models/                # Pesos base (e.g. yolov8n-face.pt)
├── preprocessing/         # Filtros de imagen (CLAHE, White-Patch, Anti-aliasing)
├── reports/               # Reportes de auditoría y métricas de evaluación
├── retraining/            # Augmentación, balanceo de clases y K-Fold CV
├── scripts/               # Scripts CLI para auditoría y entrenamiento SVM
│   ├── audit_face_dataset.py
│   ├── train_facial_svm.py
│   └── train_reid_svm.py
├── ui/                    # Interfaz gráfica de usuario en PyQt6
│   ├── dashboard.py       # Ventana principal del Dashboard
│   └── worker_thread.py   # Hilo de ejecución en segundo plano (QThread)
├── utils/                 # Utilidades de configuración, I/O y registro (logging)
├── config.yaml            # Archivo de configuración centralizado
├── main.py                # Punto de entrada principal
├── requirements.txt       # Lista de dependencias de Python
└── yolov8n.pt             # Pesos del detector YOLOv8n (se auto-descarga si no existe)
```

---

## 🚀 Guía de Uso y Ejecución

### 1. Auditoría del Dataset
Antes de entrenar, audita las imágenes en `dataset/raw_images/<sujeto>/` para identificar rostros no detectables o de baja calidad:

```bash
python scripts/audit_face_dataset.py
```
> Genera un informe detallado en `reports/dataset_audit/face_audit_report.csv`.

---

### 2. Entrenamiento de Modelos SVM

#### A. Entrenar el SVM Facial (Rama ID - Out-Of-Core)
Procesa masivamente el dataset dividiéndolo en subespacios corporales y volcando la aumentación paralela directamente a disco mediante arreglos mapeados (`np.memmap`) implementando **Zero Data Leakage CV**.

```bash
python scripts/train_facial_svm.py
```
> Modelo guardado en: `dataset/models/svm_facial/svm_facial_model.pkl`

#### B. Entrenar el SVM Re-ID (Rama Corporal - Hellinger)
Captura la silueta de $128 \times 256$, inyecta aumentación estrictamente geométrica (Flips, Cutout, Scaling) y comprime la malla LBP a 1,888 características robustas antes de compilar el Pipeline en RAM.

```bash
python scripts/train_reid_svm.py
```
> Modelo guardado en: `dataset/models/svm_reid/svm_reid_model.pkl`

---

### 3. Ejecución del Pipeline

#### Opción A: Dashboard Visual PyQt6 (Recomendado)
Lanza la interfaz gráfica interactiva con reproductor de video, tabla live de identificaciones, gráfico de distribución de ramas y consola de eventos:

```bash
python main.py --ui
```

#### Opción B: Modo OpenCV Clásico
Ejecuta el pipeline desplegando una ventana simple de OpenCV:

```bash
python main.py
```
> Presiona la tecla `q` para detener la ejecución.

---

## ⚙️ Configuración (`config.yaml`)

El archivo `config.yaml` controla todos los hiperparámetros del sistema sin necesidad de modificar código:

```yaml
video:
  source: "dataset/raw_videos/camara2.mp4"  # Ruta de video o 0 para cámara web
  target_fps: 15                            # Framerate objetivo del scheduler

detection:
  yolo_model_path: "yolov8n.pt"
  conf_threshold: 0.40
  device: "cpu"                             # 'cpu' o 'cuda'

face:
  yoloface_model_path: "yolov8n-face.pt"
  conf_threshold: 0.60

decision:
  t_aceptacion: 0.65                       # Umbral de aceptación (T_aceptación)

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
- **Panel de Video Principal**: Muestra las cajas delimitadoras coloreadas por rama (Verde = ID Facial, Naranja = Re-ID Corporal, Rojo = Desconocido) junto a la etiqueta e ícono de captura 📸.
- **Tabla Live de Personas Activas**: Lista en tiempo real los `Track ID`, identidad asignada, nivel de confianza y rama utilizada.
- **Gráfico de Distribución**: Barras de proporción que comparan la frecuencia de uso de la rama ID vs Re-ID.
- **Log de Eventos**: Consola interactiva con marcas de tiempo para nuevas detecciones, capturas guardadas y advertencias.
- **Controles de Reproducción**: Botones para Pausar/Reanudar, Seleccionar Video desde archivo o Activar Cámara Web.
- **Indicadores de Estado**: Muestra el estado de disponibilidad de los modelos SVM en la barra superior.

---

## 📊 Evaluación y MLOps

El sistema implementa criterios estrictos de promoción de modelos (`ModelPromotionGatekeeper`):
- **Cross-Validation**: Evaluación 5-Fold Stratified K-Fold.
- **Criterio de Promoción**: $\Delta F1 \ge 0.01$ y $F1_{\text{macro}} \ge 0.75$.
- **Métricas Generadas**: Matriz de Confusión $(N+1) \times (N+1)$, Curvas ROC, Curvas CMC (Rank-1 y Rank-5) y reportes HTML exportables.
