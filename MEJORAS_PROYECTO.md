# 🚀 Resumen Completo de Cambios y Mejoras del Proyecto (ID / Re-ID)

Este documento contiene la explicación detallada de la arquitectura actualizada del sistema de **Identificación Facial (ID con HOG)** y **Re-identificación Corporal (Re-ID con LBP)** para presentación en laboratorio y control de versiones.

---

## 📋 Resumen de la Nueva Arquitectura

### 1. Eliminación de YuNet y Trackers (ByteTrack / DeepSORT)
- **Razón Técnica**: Se simplificó la canalización eliminando la dependencia de detectores de rostros de aprendizaje profundo adicionales (YuNet) y de algoritmos de seguimiento temporal.
- **Implementación**:
  - El detector **YOLOv8n** procesa directamente las cajas delimitadoras de personas en cada fotograma.
  - La región de la cabeza se extrae geométricamente recortando la porción superior del cuerpo y normalizándola a **$96 \times 96$ BGR**.
  - La extracción HOG opera sobre el rostro normalizado de $96 \times 96$ BGR en canal de intensidad con filtro Tan & Triggs canónico y subespacios anatómicos (Global, Superior, Inferior).

### 2. Normalización de Rostros Completos (`face_normalizer.py`)
- **Razón Técnica**: Anteriormente, al procesar fotos verticales de retratos en `dataset/raw_images/`, el algoritmo recortaba el 35% superior pensando que eran cuerpos enteros, eliminando ojos, nariz y boca durante el entrenamiento.
- **Implementación**: Se agregó la bandera `is_body_roi`. Al entrenar con `dataset/raw_images/` (`is_body_roi=False`), se preserva el rostro completo con ojos, nariz y boca.

### 3. Aceleración por Hardware GPU (NVIDIA CUDA 12.4)
- **Implementación**: Configuración de `device: "gpu"` en `config.yaml` e instalación de **PyTorch con soporte CUDA 12.4**, logrando inferencia acelerada en tiempo real.

### 4. Eliminación Completa de Falsos Positivos
- **Umbral Estricto del 80%**: Se vinculó `t_aceptacion: 0.80` de `config.yaml` a la compuerta de decisión `ThresholdAcceptanceGate` en `PipelineOrchestrator`.
- **Filtro de Ambigüedad Top-1 vs Top-2**: En `svm_facial_model.py`, si el margen entre la primera y segunda clase con mayor probabilidad es menor al $12\%$, la predicción se considera incierta y se etiqueta como `"Desconocido"`.

### 5. Captura Dinámica Automática
- Se reactivó la captura automática de hasta 75 imágenes nítidas por persona identificada en la carpeta `dataset/captures/<Nombre>/` durante el paso 2 de inferencia.

---

## 🛠️ Archivos Modificados Principales

- `requirements.txt`: Actualizado con PyTorch CUDA, PyQt6 y setuptools.
- `config.yaml`: Configurado con `t_aceptacion: 0.80` y `device: "gpu"`.
- `feature_extraction/face/face_normalizer.py`: Recorte geométrico $96 \times 96$ BGR con bandera `is_body_roi`.
- `feature_extraction/face/face_quality_validator.py`: Validación de nitidez, brillo ($\ge 35.0$) y resolución ($32\text{px}$) sin YuNet.
- `branching/face_visibility_router.py`: Ruteo entre ID (cabeza HOG) y Re-ID (torso LBP) sin YuNet.
- `core/pipeline_orchestrator.py`: Inferencia por fotograma sin tracker y con umbral estricto al 80%.
- `classification/svm_facial_model.py`: Filtro de margen de ambigüedad Top-1 vs Top-2.
- `scripts/train_facial_svm.py`: Entrenamiento HOG de caras completas.
- `scripts/train_reid_svm.py`: Entrenamiento Re-ID LBP con fallback automático a `dataset/raw_images`.
- `ui/worker_thread.py`: Eliminadas importaciones obsoletas de trackers para evitar errores de `pkg_resources`.

---

## 🚀 Guía Rápida de Ejecución en Laboratorio

### 1. Activar Entorno Virtual
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Instalación de Dependencias con GPU (NVIDIA CUDA)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Flujo de Entrenamiento y Pruebas en 3 Pasos
1. **Entrenar ID Facial (HOG)**:
   ```powershell
   python scripts/train_facial_svm.py
   ```
2. **Ejecutar Inferencia / Capturas Dinámicas**:
   ```powershell
   python main.py --ui
   ```
3. **Entrenar Re-ID Corporal (LBP)**:
   ```powershell
   python scripts/train_reid_svm.py
   ```
