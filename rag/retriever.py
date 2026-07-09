# Retriever con recuperación semántica real (embeddings + similitud coseno)

from sentence_transformers import SentenceTransformer, util


class RAGRetriever:
    # Modelo de embeddings multilingüe, liviano y apto para CPU
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Categorías de la base de conocimiento (deben coincidir con las claves de _cargar_contextos_dnd)
    RAZAS = [
        "humanos", "elfos", "alto elfo", "enanos", "enano del valle", "orcos",
        "medio orco", "duendes", "duende de roca", "halflings", "medio elfo",
        "tieflings", "dragonborns", "tritones"
    ]
    AMBIENTES = [
        "bosques", "montañas", "desiertos", "junglas", "cuevas", "pantanos",
        "praderas", "tundra", "costas", "steampunk", "postapocaliptico",
        "gotico", "medieval", "japones", "cristal", "flotante", "volcanico"
    ]

    def __init__(self):
        print("🎲 Inicializando Retriever para D&D (embeddings semánticos)")
        self.contextos_dnd = self._cargar_contextos_dnd()
        self._ids = list(self.contextos_dnd.keys())
        self._textos = list(self.contextos_dnd.values())
        self._indice_por_id = {id_: i for i, id_ in enumerate(self._ids)}

        print(f"   📥 Cargando modelo de embeddings: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self._embeddings = self.model.encode(
            self._textos, convert_to_tensor=True, normalize_embeddings=True
        )
        print(f"   ✅ Índice semántico construido con {len(self._ids)} documentos")

    def _cargar_contextos_dnd(self):
        """Carga contextos predefinidos para D&D para todas las razas y ambientes"""
        return {
            # ==================== RAZAS ====================
            "humanos": "Los humanos son adaptables y construyen civilizaciones diversas. Son conocidos por su ambición, su capacidad para unirse contra amenazas comunes y su tendencia a expandir sus territorios. Los reinos humanos suelen tener sistemas feudales, monarquías o repúblicas mercantiles.",
            
            "elfos": "Los elfos son seres mágicos y longevos que viven en armonía con la naturaleza. Tienen una conexión profunda con el mundo feérico y son excelentes arqueros, magos y artistas. Valoran la belleza, el arte y la tradición ancestral.",
            
            "alto elfo": "Los Altos Elfos son la élite de la sociedad élfica. Son más altos, más justos y más poderosos mágicamente que sus parientes. Habitan en ciudades de cristal y marfil en bosques ancestrales, gobernados por nobles casas que remontan su linaje milenios atrás.",
            
            "enanos": "Los enanos son maestros artesanos que viven bajo las montañas. Valoran el honor, la familia, los tesoros minerales y la artesanía perfecta. Sus reinos subterráneos son fortalezas inexpugnables llenas de grandes salas, forjas humeantes y pasadizos secretos.",
            
            "enano del valle": "Los Enanos del Valle son más adaptables que sus primos de la montaña. Viven en colinas y valles fértiles, combinando la artesanía enana con la agricultura y el comercio. Son más abiertos a los forasteros y suelen tener relaciones comerciales con otras razas.",
            
            "orcos": "Los orcos son guerreros feroces que valoran la fuerza bruta por encima de todo. Viven en tribus nómadas que recorren tierras agrestes, liderados por el más fuerte. Respetan el poder y la habilidad en combate más que cualquier otra cualidad.",
            
            "medio orco": "Los Medios Orcos son hijos de humanos y orcos, viviendo entre dos mundos. Heredan la fuerza de sus ancestros orcos y la astucia de sus ancestros humanos. Suelen ser marginados pero demuestran su valía como feroces guerreros o líderes pragmáticos.",
            
            "duendes": "Los Gnomos son inventores y alquimistas brillantes. Viven en comunidades subterráneas o en bosques, siempre experimentando con nuevas creaciones. Su curiosidad insaciable y su optimismo los llevan a crear artefactos maravillosos y a veces peligrosos.",
            
            "duende de roca": "Los Gnomos de Roca viven bajo las montañas y son expertos en joyería, mecánica y alquimia. Son los más prácticos de los gnomos, creando dispositivos ingeniosos y gemas mágicas. Valoran la tradición y el conocimiento transmitido por generaciones.",
            
            "halflings": "Los Halflings son una raza pequeña, alegre y hospitalaria. Prefieren una vida tranquila en pueblos agrícolas, valorando la comida, la amistad y las pequeñas aventuras. Son increíblemente afortunados y difíciles de atrapar cuando no quieren ser encontrados.",
            
            "medio elfo": "Los Medios Elfos combinan la gracia élfica con la ambición humana. Se sienten atraídos por las ciudades humanas donde su herencia élfica les da ventaja social, pero también valoran la conexión con la naturaleza. Son diplomáticos natos y adaptables.",
            
            "tieflings": "Los Tieflings tienen sangre infernal que marca su apariencia con cuernos, colas o ojos sin pupilas. Son marginados por su herencia pero desarrollan una gran determinación. Muchos se convierten en hechiceros, brujos o pícaros que desafían las expectativas.",
            
            "dragonborns": "Los Dragonborns son humanoides dragónicos orgullosos de su herencia. Valoran el honor, la lealtad y la fuerza. Sus clanes están organizados en castas, y cada miembro busca demostrar su valor para honrar a sus ancestros dragón.",
            
            "tritones": "Los Tritones son habitantes de las profundidades oceánicas. Son nobles guerreros que protegen los mares de amenazas como sahuagines y krakens. En tierra son torpes y desconfiados, pero en el agua son ágiles y poderosos.",
            
            # ==================== AMBIENTES NATURALES ====================
            "bosques": "Los bosques están llenos de vida y misterio. Los árboles antiguos ocultan secretos druídicos, criaturas feéricas y ruinas olvidadas. Son reinos de elfos, dríadas y bestias mágicas, donde cada claro puede esconder una maravilla o un peligro.",
            
            "montañas": "Las montañas son lugares peligrosos pero ricos en minerales. Sus picos nevados y profundas cavernas albergan enanos, gigantes, dragones y criaturas de las profundidades. Los pasos son traicioneros y los deslizamientos de roca son comunes.",
            
            "desiertos": "Los desiertos son implacables pero esconden ruinas antiguas y tesoros olvidados bajo la arena. Tribus nómadas recorren las dunas, y criaturas como dragones azules o lamias acechan en templos enterrados. Las noches son frías y los días abrasadores.",
            
            "junglas": "Las junglas son densas y húmedas, llenas de vida en cada hoja. Ruinas cubiertas de musgo albergan trampas mortales y artefactos poderosos. Tabaxi, yuan-ti y dinosaurios habitan entre la vegetación. Cada paso puede revelar una criatura que acecha.",
                        
            "cuevas": "Las cuevas son oscuras y húmedas, llenas de estalactitas y pasadizos que se bifurcan. Murciélagos, goblins, trogloditas y criaturas olvidadas acechan en las profundidades. Cada esquina puede revelar un tesoro o una muerte segura.",
            
            "pantanos": "Los pantanos son lugares tristes y enfermos, llenos de niebla venenosa y aguas estancadas. Brujas, hags y criaturas no muertas acechan entre los cipreses. Cada paso puede hundirte en el barro o despertar algo que debería haber quedado dormido.",
            
            "praderas": "Las praderas son extensiones de hierba dorada que se mecen con el viento. Tribus nómadas recorren la llanura cazando bisontes enormes. Los centauros galopan libres y las tormentas eléctricas barren el horizonte sin previo aviso.",
            
            "tundra": "La tundra es un páramo helado donde la supervivencia es una lucha constante. El viento corta como cuchillas y la nieve cubre ruinas de civilizaciones olvidadas. Yetis, osos polares y pueblos bárbaros resisten el frío implacable.",
            
            "costas": "Las costas son donde la tierra se encuentra con el mar. Acantilados escarpados, calas escondidas y puertos bulliciosos. Sirenas, piratas y criaturas marinas acechan entre las olas, mientras los pescadores cuentan historias de tormentas y naufragios.",
            
            # ==================== AMBIENTES TEMÁTICOS ====================
            "steampunk": "En este mundo steampunk, el vapor y los engranajes son la base de la tecnología. Dirigibles surcan los cielos y autómatas realizan trabajos pesados. La aristocracia vive en mansiones victorianas mientras los trabajadores se hacinan en barrios industriales llenos de humo. La magia se canaliza a través de artefactos mecánicos.",
            
            "postapocaliptico": "Un mundo devastado donde restos de civilizaciones antiguas yacen entre desiertos radiactivos. Los sobrevivientes luchan por recursos básicos mientras mutantes y máquinas olvidadas acechan en las ruinas. La magia salvaje distorsiona la realidad en zonas contaminadas.",
            
            "gotico": "Un mundo de castillos en ruinas, bosques neblinosos y oscuros secretos. Vampiros, hombres lobo y seres malditos acechan en la noche. La iglesia lucha contra lo sobrenatural mientras la nobleza decadente oculta terribles crímenes. La tristeza y la melancolía impregnan el ambiente.",
            
            "medieval": "Un mundo de castillos feudales, caballeros andantes y siervos de la gleba. La iglesia tiene un poder absoluto sobre las almas. Los bosques están llenos de bandidos, bestias y brujas. Las cruzadas y las guerras señoriales son constantes. La peste y el hambre acechan en cada invierno.",
            
            "japones": "Tierras de samuráis, monjes guerreros y demonios. Castillos de madera se alzan sobre pueblos que cultivan arroz. Los señores feudales luchan por el poder mientras los shinobi ejecutan misiones secretas. Los kami y yokai habitan en bosques y montañas sagradas.",
            
            "cristal": "Cavernas maravillosas donde cristales gigantes emiten luz propia. Las formaciones minerales crean paisajes oníricos de colores brillantes. Habitan gnomos, elementales de tierra y criaturas que se alimentan de energía geomántica. El eco distorsiona el sonido y la magia se siente más potente.",
            
            "flotante": "Islas que desafían la gravedad suspendidas sobre un abismo infinito. Se conectan mediante puentes de cuerda o vuelos cortos. Habitan aarakocra, magos poderosos y criaturas aladas. El viento es constante y caer del borde es una muerte segura.",
            
            "volcanico": "Tierras de ceniza, ríos de lava y montañas que escupen fuego. Los cultistas adoran a elementales de fuego y dragones rojos. Los asentamientos se protegen en cuevas ignífugas. La agricultura es imposible pero los minerales son abundantes."
        }
    
    def determinar_raza_y_ambiente(self, query: str):
        """Determina la raza y el ambiente predominantes en la consulta libre del usuario.
        Primero busca una mención literal (nombres propios de fantasía como 'tieflings' o
        'steampunk' no siempre quedan bien representados por embeddings genéricos). Si el
        usuario no menciona ninguno explícitamente, usa similitud semántica (embeddings)
        para inferir el que mejor encaje según la descripción."""
        query_lower = query.lower()
        query_embedding = self.model.encode(
            [query], convert_to_tensor=True, normalize_embeddings=True
        )
        similitudes = util.cos_sim(query_embedding, self._embeddings)[0]

        raza, raza_score = self._determinar_categoria(query_lower, similitudes, self.RAZAS)
        ambiente, ambiente_score = self._determinar_categoria(query_lower, similitudes, self.AMBIENTES)

        print(f"   🧬 Raza predominante detectada: {raza} (similitud {raza_score:.2f})")
        print(f"   🗺️  Ambiente predominante detectado: {ambiente} (similitud {ambiente_score:.2f})")

        return raza, ambiente, raza_score, ambiente_score

    def _determinar_categoria(self, query_lower, similitudes, ids_candidatos):
        """Mención literal primero (alta precisión para nombres propios, admitiendo
        singular/plural, ej. 'bosque' vs 'bosques'); si no hay ninguna, recurre a la
        similitud semántica para inferirla de la descripción."""
        for id_ in ids_candidatos:
            formas = {id_, id_[:-1]} if id_.endswith("s") else {id_}
            if any(forma in query_lower for forma in formas):
                return id_, 1.0

        return self._mejor_coincidencia(similitudes, ids_candidatos)

    def _mejor_coincidencia(self, similitudes, ids_candidatos):
        """Encuentra, dentro de una categoría (razas o ambientes), el id con mayor similitud"""
        mejor_id, mejor_score = None, -1.0
        for id_candidato in ids_candidatos:
            score = float(similitudes[self._indice_por_id[id_candidato]])
            if score > mejor_score:
                mejor_id, mejor_score = id_candidato, score
        return mejor_id, mejor_score

    def obtener_contextos(self, raza: str, ambiente: str):
        """Recupera los contextos exactos de la raza y el ambiente ya determinados,
        para pasárselos al generador (qué generar) y al evaluador (contra qué evaluar)"""
        contextos = []

        if raza in self.contextos_dnd:
            contextos.append({"id": raza, "score": 1.0, "contexto": self.contextos_dnd[raza]})

        if ambiente in self.contextos_dnd:
            contextos.append({"id": ambiente, "score": 1.0, "contexto": self.contextos_dnd[ambiente]})

        if contextos:
            return contextos

        # Si por algún motivo no se reconocen los ids, usar contextos generales
        return [
            {"id": "general1", "score": 0.5, "contexto": "En D&D, los escenarios pueden variar desde mazmorras oscuras hasta reinos mágicos. Cada ambiente tiene sus propias reglas y habitantes que moldean la vida en el lugar."},
            {"id": "general2", "score": 0.5, "contexto": "Los Dungeon Masters deben crear descripciones atmosféricas para inmersión, considerando la política local, economía y relaciones entre facciones."},
            {"id": "general3", "score": 0.5, "contexto": "La geografía, el clima y las criaturas nativas definen las oportunidades y peligros de un territorio para los aventureros."}
        ]