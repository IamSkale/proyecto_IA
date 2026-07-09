from .retriever import RAGRetriever
from .generator import RAGGenerator
from .evaluator import RAGEvaluator

# Palabras clave para detectar la extensión del territorio (no es un dato "semántico"
# como raza/ambiente, sino una escala estructural, por eso se resuelve por palabras clave)
EXTENSION_PALABRAS_CLAVE = {
    "pueblo": ["pueblo", "aldea", "villa", "poblado"],
    "reino": ["reino", "imperio", "corona", "monarquía", "monarquia", "reinado"],
    "región": ["región", "region", "continente", "territorios", "tierras"],
}


class RAGPipeline:
    def __init__(self, model="qwen2.5-3b", model_path=None, use_gpu=False, auto_evaluar=True, max_intentos=3):
        print(f"\n🎲 Inicializando RAG Pipeline para D&D")
        print(f"   Modelo: {model}")
        print(f"   Auto-evaluación con IA: {'✅ Activada' if auto_evaluar else '❌ Desactivada'}")
        
        self.auto_evaluar = auto_evaluar
        self.max_intentos = max_intentos
        
        self.retriever = RAGRetriever()
        self.generator = RAGGenerator(
            model=model,
            model_path=model_path,
            use_gpu=use_gpu
        )
        
        if auto_evaluar:
            self.evaluator = RAGEvaluator(self.generator)
    
    def generar_escenario(self, raza, ambiente, extension):
        print(f"\n🏰 Generando escenario D&D")
        print(f"   Raza: {raza}")
        print(f"   Ambiente: {ambiente}")
        print(f"   Extensión: {extension}")

        # Recuperar los contextos EXACTOS de la raza y el ambiente ya determinados,
        # para inyectarlos en el prompt del generador y en la evaluación
        contextos = self.retriever.obtener_contextos(raza, ambiente)
        print(f"   📚 Contextos recuperados ({len(contextos)}):")
        for c in contextos:
            print(f"      - [{c['score']:.2f}] {c['id']}")

        if self.auto_evaluar and hasattr(self, 'evaluator'):
            # Usar evaluación con IA y corrección automática, pasando los contextos recuperados
            descripcion, evaluacion = self.evaluator.regenerar_si_necesario(
                raza, ambiente, extension, contextos, self.max_intentos
            )
            return descripcion, contextos, evaluacion
        else:
            # Modo simple sin evaluación
            descripcion = self.generator.generar_descripcion_dnd(raza, ambiente, extension, contextos=contextos)
            return descripcion, contextos, None

    def generar_desde_descripcion(self, descripcion_usuario):
        raza, ambiente, raza_score, ambiente_score = self.retriever.determinar_raza_y_ambiente(descripcion_usuario)
        extension = self._detectar_extension(descripcion_usuario)

        print(f"\n🔍 Parámetros detectados a partir de la descripción del usuario")
        print(f"   Raza: {raza} (similitud semántica {raza_score:.2f})")
        print(f"   Ambiente: {ambiente} (similitud semántica {ambiente_score:.2f})")
        print(f"   Extensión: {extension}")

        descripcion, contextos, evaluacion = self.generar_escenario(raza, ambiente, extension)
        return descripcion, contextos, evaluacion, raza, ambiente, extension

    def _detectar_extension(self, texto: str) -> str:
        """Detecta la extensión del territorio (pueblo/reino/región) por palabras clave.
        No es un dato semántico como raza/ambiente, sino una escala estructural del texto."""
        texto_lower = texto.lower()
        for extension, palabras_clave in EXTENSION_PALABRAS_CLAVE.items():
            if any(palabra in texto_lower for palabra in palabras_clave):
                return extension
        return "pueblo"