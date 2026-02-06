# 🧬 BioPhys-Tech Lab: Audio Classifier V1.0

Este repositorio contiene el **Producto Mínimo Viable (MVP)** para la clasificación automatizada de bioacústica. El sistema utiliza una arquitectura híbrida de vanguardia para identificar especies a partir de espectrogramas, optimizado para el monitoreo de biodiversidad en tiempo real.

## 🚀 Logros del Proyecto
* **Precisión de Élite:** Se alcanzó un **77.75% de precisión en validación** en la Época 10.
* **Eficiencia Térmica y de Cómputo:** Implementación de **offline caching**, logrando un **12x speedup** en el entrenamiento (de 2 horas a ~10 min por época).
* **Arquitectura PhD-Level:** Integración de **EfficientNet + Transformers** con preprocesamiento avanzado (**PCEN + Spectral Gating**).

## 📊 Historial de Entrenamiento (Bioacoustics Model)

| Época | Train Loss | Train Acc | Val Loss | Val Acc | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 3.38 | 13.88% | 2.44 | 28.75% | Warm-up |
| 5 | 0.94 | 70.94% | 1.04 | 70.25% | **Hito >70%** |
| 8 | 0.53 | 82.81% | 0.92 | 73.75% | Modelo Estable |
| 9 | 0.47 | 85.31% | 0.91 | 75.75% | Optimización |
| **10** | **0.41** | **87.75%** | **0.82** | **77.75%** | **MVP FINAL** |

## 🛠️ Estructura del Repositorio
* `src/`: Scripts principales de entrenamiento, cargador de datos y arquitectura del modelo.
* `predict.py`: Script de inferencia para clasificar nuevos archivos de audio.
* `requirements.txt`: Dependencias necesarias para replicar el entorno de desarrollo.
* `checkpoints/`: (Descarga externa) Contiene el archivo de pesos `best_model.pth`.

## 💻 Uso e Inferencia
Para verificar el modelo con un archivo de audio de prueba:
```bash
python predict.py
```
Debido a la densidad de parámetros (151 MB), el modelo final se encuentra alojado externamente: 👉 (https://1drv.ms/f/c/E911E7181C90BA7E/IgB4rFVopgK3Q5BSsBa93h1wATPtDnp63mZBBAgcxtDfntI?e=g05NqD)

Desarrollado por: Cristian Javier Ibadango Avilez Institución: BioPhys-Tech Lab | Ibarra, Ecuador
