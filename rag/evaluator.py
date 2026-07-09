from typing import Dict, Tuple
import json
import re

class RAGEvaluator:    
    # Rangos de palabras orientativos (solo como referencia, no para evaluación estricta)
    EXTENSION_LONGITUD = {
        "pueblo": {"min": 150, "max": 300, "promedio": 225},
        "reino": {"min": 300, "max": 600, "promedio": 450},
        "región": {"min": 600, "max": 1000, "promedio": 800}
    }
    
    def __init__(self, generator):
        self.generator = generator
        print("🎲 Inicializando Evaluador de Escenarios D&D - Modo IA (evaluación por el modelo)")
    
    def evaluar_escenario_con_ia(self, texto: str, raza: str, ambiente: str, extension: str) -> Dict:
        prompt_evaluacion = f"""Eres un Dungeon Master experto evaluando una descripción de escenario para Dungeons & Dragons.

CARACTERÍSTICAS QUE DEBE CUMPLIR:
- Raza predominante: {raza}
- Ambiente: {ambiente}
- Extensión del territorio: {extension}

DESCRIPCIÓN DEL ESCENARIO A EVALUAR:
{texto}

INSTRUCCIONES:
Evalúa si esta descripción es adecuada para un escenario de D&D. No te bases solo en palabras clave, sino en el contenido real.

Responde SOLO con un JSON válido en este formato exacto, sin texto adicional:
{{
    "cumple": true/false,
    "puntuacion": (número entre 0 y 100),
    "raza_adecuada": true/false,
    "ambiente_adecuado": true/false,
    "extension_adecuada": true/false,
    "calidad_narrativa": (número entre 0 y 100),
    "nivel_detalle": (número entre 0 y 100),
    "texto_completo": true/false,
    "problemas": ["problema1", "problema2"],
    "puntos_fuertes": ["punto1", "punto2"],
    "sugerencias": ["sugerencia1", "sugerencia2"]
}}

CRITERIOS DE EVALUACIÓN:
1. ¿La descripción captura la esencia de la raza {raza} (su cultura, valores, forma de vida)?
2. ¿El ambiente {ambiente} está bien representado (atmósfera, peligros, oportunidades)?
3. ¿El nivel de detalle es apropiado para un {extension}?
4. ¿La narrativa es evocadora y útil para un Dungeon Master?
5. ¿El texto se siente completo (no cortado a mitad)?

Sé flexible pero honesto. Una descripción puede ser buena sin mencionar explícitamente cada palabra clave."""
        
        try:
            # Usar temperatura baja para evaluación consistente
            respuesta = self.generator._generate(prompt_evaluacion, max_tokens=800, temperature=0.2)
            
            # Extraer JSON de la respuesta
            json_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*}', respuesta, re.DOTALL)
            if json_match:
                evaluacion = json.loads(json_match.group())
            else:
                # Fallback si no encuentra JSON
                evaluacion = self._evaluacion_fallback(texto, raza, ambiente, extension)
            
            # Asegurar que todos los campos existan
            evaluacion = self._completar_evaluacion(evaluacion)
            
            return evaluacion
            
        except Exception as e:
            print(f"   ⚠️ Error en evaluación con IA: {e}")
            return self._evaluacion_fallback(texto, raza, ambiente, extension)
    
    def _completar_evaluacion(self, evaluacion: dict) -> dict:
        campos_por_defecto = {
            "cumple": False,
            "puntuacion": 50,
            "raza_adecuada": False,
            "ambiente_adecuado": False,
            "extension_adecuada": False,
            "calidad_narrativa": 50,
            "nivel_detalle": 50,
            "texto_completo": True,
            "problemas": ["No se pudo evaluar correctamente"],
            "puntos_fuertes": [],
            "sugerencias": ["Intenta generar de nuevo"]
        }
        
        for campo, valor_defecto in campos_por_defecto.items():
            if campo not in evaluacion:
                evaluacion[campo] = valor_defecto
        
        return evaluacion
    
    def _evaluacion_fallback(self, texto: str, raza: str, ambiente: str, extension: str) -> dict:
        palabras = len(texto.split())
        rango = self.EXTENSION_LONGITUD.get(extension.lower(), {"min": 150})
        
        return {
            "cumple": palabras >= rango["min"] and len(texto) > 200,
            "puntuacion": min(100, (palabras / rango["min"]) * 100) if palabras < rango["min"] else 70,
            "raza_adecuada": raza.lower() in texto.lower(),
            "ambiente_adecuado": ambiente.lower() in texto.lower(),
            "extension_adecuada": palabras >= rango["min"],
            "calidad_narrativa": 60,
            "nivel_detalle": 50,
            "texto_completo": texto.endswith(('.', '!', '?')),
            "problemas": ["Evaluación automática por fallo de IA"],
            "puntos_fuertes": [],
            "sugerencias": ["Reintentar generación"]
        }
    
    def _construir_prompt_correccion(self, texto: str, raza: str, ambiente: str, extension: str, evaluacion: dict, contextos=None) -> str:
        problemas = evaluacion.get("problemas", [])
        sugerencias = evaluacion.get("sugerencias", [])

        texto_problemas = "\n".join([f"- {p}" for p in problemas]) if problemas else "- Ninguno detectado específicamente"
        texto_sugerencias = "\n".join([f"- {s}" for s in sugerencias]) if sugerencias else "- Mejorar la calidad general"

        contexto_texto = ""
        if contextos:
            lineas = "\n".join(f"- {c['contexto']}" for c in contextos)
            contexto_texto = f"\nCONTEXTO DE REFERENCIA (asegúrate de reflejarlo en la versión corregida):\n{lineas}\n"

        return f"""Eres un Dungeon Master experto. Necesitas mejorar la siguiente descripción de un escenario de D&D.

CARACTERÍSTICAS REQUERIDAS:
- Raza: {raza}
- Ambiente: {ambiente}
- Extensión: {extension}
{contexto_texto}
DESCRIPCIÓN ACTUAL:
{texto}

EVALUACIÓN DE LA VERSIÓN ANTERIOR:
- Puntuación: {evaluacion.get('puntuacion', 50)}/100
- Raza adecuada: {'Sí' if evaluacion.get('raza_adecuada') else 'No'}
- Ambiente adecuado: {'Sí' if evaluacion.get('ambiente_adecuado') else 'No'}
- Extensión adecuada: {'Sí' if evaluacion.get('extension_adecuada') else 'No'}
- Calidad narrativa: {evaluacion.get('calidad_narrativa', 50)}/100

PROBLEMAS DETECTADOS:
{texto_problemas}

SUGERENCIAS DE MEJORA:
{texto_sugerencias}

INSTRUCCIONES DE CORRECCIÓN:
1. Mantén el mismo tono y estilo narrativo
2. Aborda específicamente los problemas mencionados
3. Mejora la calidad general de la descripción
4. Asegúrate de que la raza {raza} sea central en la descripción
5. El ambiente {ambiente} debe sentirse vivo y relevante
6. El nivel de detalle debe ser apropiado para un {extension}
7. NO incluyas explicaciones de lo que cambiaste
8. Responde SOLO con el texto mejorado, sin comentarios adicionales

Genera la versión mejorada del escenario:"""
    
    def corregir_con_ia(self, texto: str, raza: str, ambiente: str, extension: str, evaluacion: dict, contextos=None) -> str:
        prompt = self._construir_prompt_correccion(texto, raza, ambiente, extension, evaluacion, contextos)
        
        try:
            texto_corregido = self.generator._generate(prompt, max_tokens=1000, temperature=0.6)
            
            # Limpiar el resultado
            if texto_corregido.startswith('"') and texto_corregido.endswith('"'):
                texto_corregido = texto_corregido[1:-1]
            
            return texto_corregido.strip()
            
        except Exception as e:
            print(f"   ⚠️ Error en corrección con IA: {e}")
            return texto
    
    def regenerar_si_necesario(self, raza: str, ambiente: str, extension: str, contextos=None, max_intentos: int = 3) -> Tuple[str, Dict]:
        texto = None
        evaluacion = None

        for intento in range(max_intentos):
            print(f"\n🔄 Intento {intento + 1} de {max_intentos}")

            if intento == 0:
                # Primera generación normal
                texto = self.generator.generar_descripcion_dnd(raza, ambiente, extension, contextos=contextos)
            else:
                # Intentos siguientes: corregir basado en evaluación del modelo
                if evaluacion:
                    print(f"   🔧 Corrigiendo según evaluación del modelo...")
                    texto = self.corregir_con_ia(texto, raza, ambiente, extension, evaluacion, contextos)
                else:
                    # Regenerar desde cero con temperatura diferente
                    print(f"   🎲 Regenerando escenario...")
                    texto = self.generator.generar_descripcion_dnd(raza, ambiente, extension, contextos=contextos, temperature=0.8)
            
            # Evaluar con el modelo (NO análisis local)
            print(f"   🤖 Evaluando con IA...")
            evaluacion = self.evaluar_escenario_con_ia(texto, raza, ambiente, extension)
            
            puntuacion = evaluacion.get('puntuacion', 0)
            cumple = evaluacion.get('cumple', False)
            
            print(f"   📊 Puntuación IA: {puntuacion:.1f}%")
            print(f"   🎯 Raza adecuada: {'✅' if evaluacion.get('raza_adecuada') else '❌'}")
            print(f"   🌍 Ambiente adecuado: {'✅' if evaluacion.get('ambiente_adecuado') else '❌'}")
            print(f"   📏 Extensión adecuada: {'✅' if evaluacion.get('extension_adecuada') else '❌'}")
            print(f"   ✍️ Calidad narrativa: {evaluacion.get('calidad_narrativa', 0):.1f}%")
            print(f"   📄 Texto completo: {'✅' if evaluacion.get('texto_completo') else '❌'}")
            
            if cumple and puntuacion >= 70:
                print(f"   ✅ Escenario ACEPTADO por el modelo después de {intento + 1} intentos")
                return texto, evaluacion
            else:
                problemas = evaluacion.get("problemas", [])
                if problemas:
                    print(f"   ⚠️ El modelo detectó problemas:")
                    for problema in problemas[:3]:
                        print(f"      - {problema}")
        
        print(f"   ⚠️ Se alcanzó el máximo de intentos. Usando el último escenario generado")
        return texto, evaluacion