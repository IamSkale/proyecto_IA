from flask import Flask, render_template, jsonify, request, send_from_directory
import json
import os
from datetime import datetime
import sys

# Agregar el directorio actual al path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Importar el pipeline RAG
try:
    from rag import RAGPipeline
    RAG_AVAILABLE = True
    print("✅ RAG Pipeline importado correctamente")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"❌ Error importando RAG: {e}")
    print("   Asegúrate de que la carpeta 'rag' exista con los archivos necesarios")

app = Flask(__name__,
           template_folder=os.path.join(BASE_DIR, 'templates'),
           static_folder=os.path.join(BASE_DIR, 'static'))

# Inicializar el pipeline RAG
rag_pipeline = None
if RAG_AVAILABLE:
    try:
        rag_pipeline = RAGPipeline(
            model="qwen2.5-3b",
            use_gpu=False  # Cambia a True si tienes GPU
        )
        print("✅ RAG Pipeline inicializado correctamente")
    except Exception as e:
        print(f"❌ Error inicializando RAG: {e}")
        rag_pipeline = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Ruta para generar escenario usando RAG
@app.route('/api/generar-escenario', methods=['POST'])
def generar_escenario():
    try:
        data = request.json
        descripcion_usuario = (data.get('descripcion') or '').strip()

        if not descripcion_usuario:
            return jsonify({"success": False, "error": "Por favor, describe el escenario que quieres crear."}), 400

        texto_generado = ""
        usando_ia = False
        evaluacion = None
        raza = ambiente = extension = None

        if rag_pipeline:
            try:
                descripcion, contextos, evaluacion, raza, ambiente, extension = rag_pipeline.generar_desde_descripcion(descripcion_usuario)

                texto_generado = descripcion
                usando_ia = True
                print(f"✅ Escenario generado con {len(contextos)} contextos")

                if evaluacion:
                    print(f"📊 Evaluación: {evaluacion['puntuacion']:.1f}% - {'Aprobado' if evaluacion['cumple'] else 'Rechazado'}")

            except Exception as e:
                print(f"❌ Error en RAG: {e}")
                texto_generado = f"❌ Error generando escenario: {str(e)}"
        else:
            texto_generado = "⚠️ RAG no disponible. Por favor, instala las dependencias necesarias."

        return jsonify({
            "success": True,
            "texto": texto_generado,
            "usando_ia": usando_ia,
            "evaluacion": evaluacion,  # Incluir evaluación en la respuesta
            "raza": raza,
            "ambiente": ambiente,
            "extension": extension
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Ruta para obtener los textos guardados
@app.route('/api/textos', methods=['GET'])
def obtener_textos():
    json_path = os.path.join(BASE_DIR, 'data.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return jsonify(data)
    except FileNotFoundError:
        datos_por_defecto = {"textos": []}
        guardar_json(datos_por_defecto, json_path)
        return jsonify(datos_por_defecto)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta para guardar un escenario generado
@app.route('/api/textos', methods=['POST'])
def agregar_texto():
    json_path = os.path.join(BASE_DIR, 'data.json')
    try:
        nuevo_texto = request.json
        
        if 'id' not in nuevo_texto:
            nuevo_texto['id'] = int(datetime.now().timestamp() * 1000)
        
        # Leer el archivo JSON actual
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        else:
            data = {"textos": []}
        
        # Agregar el nuevo texto
        data['textos'].append(nuevo_texto)
        
        # Guardar en el archivo JSON
        guardar_json(data, json_path)
        
        return jsonify({"success": True, "mensaje": "Escenario guardado exitosamente"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Ruta para eliminar un texto
@app.route('/api/textos/<int:id>', methods=['DELETE'])
def eliminar_texto(id):
    json_path = os.path.join(BASE_DIR, 'data.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        data['textos'] = [texto for texto in data['textos'] if texto['id'] != id]
        guardar_json(data, json_path)
        
        return jsonify({"success": True, "mensaje": "Escenario eliminado exitosamente"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def guardar_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # Crear carpetas necesarias
    os.makedirs(os.path.join(BASE_DIR, 'templates'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    
    print(f"\n🏰 Servidor de Generación de Escenarios D&D")
    print(f"📍 http://localhost:5000")
    print(f"📁 Directorio: {BASE_DIR}")
    print(f"🤖 RAG disponible: {RAG_AVAILABLE and rag_pipeline is not None}")
    print(f"📦 Modelos: {os.path.join(BASE_DIR, 'models')}")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True, host='localhost', port=5000)