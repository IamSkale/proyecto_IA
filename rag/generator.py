import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("⚠️ llama-cpp-python no instalado. Ejecuta: pip install llama-cpp-python")


class RAGGenerator:
    # Opciones de modelo para D&D
    MODEL_OPTIONS = {
        "qwen2.5-1.5b": {
            "name": "Qwen2.5-1.5B-Instruct",
            "gguf": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
            "hf_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "context": 32768,
            "ram_gb": 2,
        },
        "qwen2.5-3b": {
            "name": "Qwen2.5-3B-Instruct",
            "gguf": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            "hf_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "context": 32768,
            "ram_gb": 4,
        }
    }
    
    # Rangos de palabras según la extensión del territorio
    EXTENSION_LONGITUD = {
        "pueblo": {"min": 150, "max": 300, "promedio": 225},
        "reino": {"min": 300, "max": 600, "promedio": 450},
        "región": {"min": 600, "max": 1000, "promedio": 800}
    }

    def __init__(self, model="qwen2.5-1.5b", model_path=None, use_gpu=False):
        self.model_name = model
        self.model = None
        self.use_gpu = use_gpu
        
        if model_path:
            self.model_path = Path(model_path)
        else:
            model_info = self.MODEL_OPTIONS.get(model, self.MODEL_OPTIONS["qwen2.5-1.5b"])
            self.model_path = Path("models") / model_info["gguf"]
            self.model_info = model_info
        
        print(f"\n🎲 Inicializando generador D&D con {self.model_info['name']}")
        print(f"   Model path: {self.model_path}")
        print(f"   GPU activada: {use_gpu}")
        
        self._cargar_modelo()

    def _cargar_modelo(self):
        if not LLAMA_AVAILABLE:
            print("⚠️ llama-cpp-python no instalado")
            return False
            
        try:
            if not self.model_path.exists():
                print(f"❌ Modelo no encontrado: {self.model_path}")
                print(f"\n📥 Para descargar el modelo:")
                print(f"   mkdir -p models")
                print(f"   cd models")
                print(f"   wget https://huggingface.co/{self.model_info['hf_id']}/resolve/main/{self.model_info['gguf']}")
                return False
            
            print(f"   📂 Cargando modelo...")
            
            n_gpu_layers = -1 if self.use_gpu else 0
            
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=8192,  # Aumentado de 4096 a 8192 para mejor contexto
                n_threads=4,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                seed=42,
                # Parámetros adicionales para mejor finalización
                logits_all=False,
                # Evita que corte en medio de palabras
                token_healing=True
            )
            print(f"   ✅ Modelo cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"   ❌ Error cargando modelo: {e}")
            return False

    def _obtener_longitud_por_extension(self, extension: str) -> dict:
        """Obtiene el rango de palabras según la extensión"""
        extension_lower = extension.lower()
        return self.EXTENSION_LONGITUD.get(extension_lower, {"min": 150, "max": 300, "promedio": 225})

    def generar_descripcion_dnd(self, raza, ambiente, extension, contextos=None, temperature=0.7):
        """Genera una descripción de escenario D&D basada en los selectores, enriquecida con contextos recuperados por RAG"""

        # Obtener rango de palabras según extensión
        rango = self._obtener_longitud_por_extension(extension)
        max_tokens = rango["max"]

        print(f"   📏 Longitud objetivo: {rango['min']}-{rango['max']} palabras ({extension})")
        if contextos:
            print(f"   📚 Inyectando {len(contextos)} contexto(s) recuperado(s) en el prompt")

        # Construir prompt específico para D&D con indicación de longitud y contextos recuperados
        prompt = self._build_dnd_prompt(raza, ambiente, extension, rango, contextos)
        
        if self.model:
            try:
                response = self._generate(prompt, max_tokens, temperature)
                if response:
                    # Verificar y ajustar longitud si es necesario
                    palabras = len(response.split())
                    print(f"   📝 Texto generado: {palabras} palabras")
                    
                    # Si es muy corto, sugerir regeneración en la evaluación
                    if palabras < rango["min"]:
                        print(f"   ⚠️ El texto es más corto de lo esperado para un {extension}")
                    
                    return response
            except Exception as e:
                print(f"❌ Error generando: {e}")
        
        # Fallback si no hay modelo
        return self._fallback_descripcion(raza, ambiente, extension)
    
    def _generate(self, prompt, max_tokens, temperature):
        """Genera respuesta usando el modelo - método público para el evaluador"""
        formatted_prompt = self._format_prompt(prompt)
        
        response = self.model(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["<|im_end|>", "<|endoftext|>", "\n\n\n", "Human:", "User:"],
            echo=False,
        )
        
        texto = response['choices'][0]['text'].strip()
        
        # Si el texto está vacío o es muy corto, reintentar una vez
        if len(texto.split()) < 20 and max_tokens > 100:
            print(f"   ⚠️ Texto demasiado corto, reintentando...")
            response = self.model(
                formatted_prompt,
                max_tokens=max_tokens * 2,
                temperature=temperature,
                stop=["<|im_end|>", "<|endoftext|>"],
                echo=False,
            )
            texto = response['choices'][0]['text'].strip()
        
        # Limpiar texto truncado
        if texto and not texto.endswith(('.', '!', '?')):
            ultimo_punto = max(texto.rfind('.'), texto.rfind('!'), texto.rfind('?'))
            if ultimo_punto > len(texto) // 2:
                texto = texto[:ultimo_punto + 1]
        
        return texto
    
    def _format_prompt(self, user_message):
        return f"""<|im_start|>system
Eres un Dungeon Master experto en Dungeons & Dragons. Creas descripciones inmersivas de escenarios para aventuras. Tus descripciones son útiles para el juego.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    
    def _formatear_contextos(self, contextos):
        """Convierte los contextos recuperados por el retriever en una sección de prompt"""
        if not contextos:
            return ""

        lineas = "\n".join(
            f"- [relevancia {c['score']:.2f}] {c['contexto']}" for c in contextos
        )
        return f"""

    CONTEXTO RECUPERADO (información de referencia real sobre la raza y el ambiente, úsala explícitamente para dar coherencia y detalles concretos a la descripción, no la ignores):
    {lineas}
"""

    def _build_dnd_prompt(self, raza, ambiente, extension, rango, contextos=None):
        """Construye el prompt enfocado en aspectos políticos, económicos y sociales según extensión"""

        contexto_texto = self._formatear_contextos(contextos)

        # Prompt específico para PUEBLO (más simple, enfocado en comunidad)
        if extension.lower() == "pueblo":
            return f"""Crea una descripción de aproximadamente {rango['min']}-{rango['max']} palabras para un PUEBLO en Dungeons & Dragons con estas características:

    Raza principal: {raza}
    Ambiente predominante: {ambiente}
{contexto_texto}
    Enfócate en:
    - **Gobierno local**: ¿Quién lidera el pueblo? (alcalde, consejo de ancianos, cacique)
    - **Economía**: ¿De qué vive la gente? (depende del ambiente)
    - **Estructura social**: ¿Cómo se relacionan los habitantes?
    - **Conflictos locales**: ¿Problemas con bandidos, impuestos abusivos, escasez de recursos?
    - **Relaciones externas**: ¿A qué reino o región pertenece?

    No necesitas describir paisajes, colores o detalles visuales a menos que sean importantes para entender la política local."""

        # Prompt específico para REINO (más complejo, enfocado en política y economía regional)
        elif extension.lower() == "reino":
            return f"""Crea una descripción de aproximadamente {rango['min']}-{rango['max']} palabras para un REINO en Dungeons & Dragons con estas características:

    Raza principal: {raza}
    Ambiente predominante: {ambiente}
{contexto_texto}
    Enfócate en:
    - **Sistema de gobierno**: Monarquía absoluta, feudal, consejo mágico, teocracia
    - **Línea de sucesión**: ¿Hay crisis sucesoria? ¿El gobernante es fuerte o débil?
    - **Economía**: Principales recursos, rutas comerciales, moneda, gremios poderosos
    - **Estructura social**: Nobleza, clero, gremios, campesinos - ¿cómo se relacionan?
    - **Conflictos activos**: Guerras con vecinos, rebeliones internas, disputas territoriales
    - **Relaciones exteriores**: Alianzas, tratados, enemigos, vasallaje

    Prioriza intrigas políticas, luchas de poder y dinámicas económicas sobre descripciones visuales."""

        # Prompt específico para REGIÓN (muy detallado, múltiples facciones)
        else:  # región
            return f"""Crea una descripción de aproximadamente {rango['min']}-{rango['max']} palabras para una REGIÓN en Dungeons & Dragons con estas características:

    Raza principal: {raza}
    Ambiente predominante: {ambiente}
{contexto_texto}
    Enfócate en:
    - **Estructura de poder**: Múltiples reinos, ciudades-estado, territorios autónomos
    - **Relaciones entre facciones**: Alianzas, guerras, tratados comerciales, espionaje
    - **Economía regional**: Comercio entre territorios, recursos estratégicos, rivalidades económicas
    - **Jerarquías sociales**: Diferencias entre clases sociales, privilegios, esclavitud o libertad
    - **Conflictos mayores**: Guerras regionales, disputas fronterizas, crisis políticas
    - **Tensiones raciales**: ¿Cómo se relaciona la raza {raza} con otras razas en la región?

    Esta región debe sentirse viva y compleja, con múltiples actores políticos y económicos compitiendo por poder."""
    
    def _fallback_descripcion(self, raza, ambiente, extension):        
        rango = self._obtener_longitud_por_extension(extension)
        
        # Descripciones base
        descripciones_base = {
            "humanos": f"🏰 **El {extension} de {ambiente}**\n\nLas caravanas comerciales recorren los caminos de este territorio humano, donde castillos y aldeas salpican el paisaje. Los mercaderes ofrecen mercancías exóticas, mientras que los guardias patrullan las murallas protegiéndose de bandidos y monstruos. En las tabernas, los aventureros escuchan rumores sobre tesoros perdidos y mazmorras olvidadas.\n\n⚠️ **Posibles encuentros**: Bandidos en los caminos, gremios de ladrones en las ciudades, y bestias que atacan los rebaños.",
            
            "elfos": f"🧝 **El {extension} de {ambiente} élfico**\n\nEntre los antiguos árboles, luces mágicas parpadean como estrellas caídas. Los elfos han esculpido sus hogares en la madera viva, creando una ciudad que respira con el bosque. Puentes de cuerda conectan plataformas elevadas, y el canto de los bardos elfos se mezcla con el susurro de las hojas. Los visitantes deben demostrar respeto por la naturaleza para ser bienvenidos.\n\n⚠️ **Posibles encuentros**: Guardianes del bosque, criaturas feéricas y druidas protectores.",
            
            "enanos": f"⛏️ **El {extension} de {ambiente} enano**\n\nBajo la montaña, martillos resuenan contra el metal en interminables forjas. Grandes salas de piedra tallada albergan estatuas de héroes ancestrales, mientras ríos de lava iluminan pasadizos secretos. Los enanos comercian con gemas preciosas y armaduras de calidad legendaria, pero desconfían de los forasteros que se aventuran demasiado cerca de sus tesoros.\n\n⚠️ **Posibles encuentros**: Duergar en las profundidades, trampas enanas y criaturas de la oscuridad.",
            
            "orcos": f"👹 **El {extension} de {ambiente} orco**\n\nHogueras humeantes marcan los campamentos orcos que se extienden por este territorio agreste. Los guerreros pisan fuerte, probando su fuerza en combates rituales mientras los chamanes invocan a espíritus salvajes. Los no orcos rara vez son bienvenidos, y solo los más fuertes sobreviven el tiempo suficiente para ganarse un respeto rudimentario.\n\n⚠️ **Posibles encuentros**: Patrullas orcas, bestias de guerra y trampas de caza.",
        }
        
        descripcion_base = descripciones_base.get(raza, f"📖 **El {extension} de {ambiente}**\n\nUn territorio misterioso dominado por los {raza}. Los viajeros cuentan historias de maravillas y peligros en igual medida. Quienes se aventuran aquí deben prepararse para lo inesperado.\n\n⚠️ **Posibles encuentros**: Criaturas locales, trampas naturales y habitantes desconfiados.")
        
        # Añadir detalles adicionales según la extensión
        if extension.lower() in ["reino", "región", "continente"]:
            detalles_extra = f"\n\n✨ **Características del {extension}:**\n"
            if extension.lower() == "reino":
                detalles_extra += "- Varias ciudades y aldeas bajo un mismo gobierno\n- Fronteras definidas y caminos comerciales\n- Conflictos políticos y alianzas entre nobles"
            elif extension.lower() == "región":
                detalles_extra += "- Diversidad de terrenos y ecosistemas\n- Múltiples asentamientos de diferentes razas\n- Ricas historias locales y tradiciones únicas"
            else:  # continente
                detalles_extra += "- Inmensa extensión de tierras y mares\n- Civilizaciones enteras y culturas diversas\n- Grandes eventos históricos que moldearon el mundo"
            
            descripcion_base += detalles_extra

        return descripcion_base