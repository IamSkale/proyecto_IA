# 🎲 Generador de Escenarios para Dungeons & Dragons con IA

![Versión](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

Sistema de generación automática de escenarios para **Dungeons & Dragons** utilizando un Modelo de Lenguaje Grande (LLM) como componente central. El sistema genera descripciones estructuradas de pueblos, reinos o continentes, incluyendo aspectos físicos, políticos, económicos y sociales.


## ✨ Características

- **Generación inteligente**: Usa LLM para crear escenarios coherentes y detallados
- **Evaluación automática**: El mismo LLM evalúa si el escenario cumple las restricciones
- **Múltiples escalas**: Genera descripciones para pueblos, reinos o continentes
- **Diversidad de razas**: Soporta 14 razas diferentes (humanos, elfos, enanos, etc.)
- **Variedad de ambientes**: 17 ambientes naturales y temáticos
- **Regeneración iterativa**: Mejora automática hasta cumplir criterios de calidad
- **Interfaz web**: Aplicación Flask con interfaz amigable
- **Sistema RAG**: Recupera contextos relevantes mediante embeddings semánticos (similitud coseno) e inyecta esa información en el prompt del LLM

## 📦 Requisitos

### Hardware
- **RAM mínima**: 4GB (8GB recomendado)
- **Almacenamiento**: 2GB libres
- **GPU**: Opcional (para aceleración)

### Software
- **Python**: 3.10 o superior
- **Sistema operativo**: Windows, Linux o macOS

## 🔧 Instalación

### Clonar o crear el proyecto
- git clone https://github.com/IamSkale/dnd_scenaries_gen
- cd dnd_scenaries_gen

### Crear entorno virtual
- python -m venv venv
- source venv/bin/activate  # En Windows: venv\Scripts\activate

### Instalar dependencias
- pip install flask llama-cpp-python python-dotenv sentence-transformers

### Descargar el modelo
- mkdir models
- cd models
- wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf
- cd ..

### Ejecutar el proyecto
- python app.py
