// Array para almacenar los textos
let textos = [];
let escenarioActual = null; // Almacenar el escenario generado actualmente
let parametrosActuales = null; // raza, ambiente y extensión detectados por la IA

// Elementos del DOM
const descripcionInput = document.getElementById('descripcionUsuario');
const btnGenerarIA = document.getElementById('btnGenerarIA');
const btnGuardar = document.getElementById('btnGuardar');
const loadingIndicator = document.getElementById('loadingIndicator');
const previewGroup = document.getElementById('previewGroup');
const vistaPreviaDiv = document.getElementById('vistaPrevia');
const parametrosDetectados = document.getElementById('parametrosDetectados');
const parametrosTextoDiv = document.getElementById('parametrosTexto');

// Función para generar escenario con IA
async function generarEscenarioConIA() {
    const descripcion = descripcionInput.value.trim();

    if (!descripcion) {
        alert('⚠️ Por favor, describe el escenario que quieres crear.');
        return;
    }

    // Mostrar loading
    loadingIndicator.style.display = 'block';
    previewGroup.style.display = 'none';
    parametrosDetectados.style.display = 'none';
    btnGenerarIA.disabled = true;
    btnGuardar.disabled = true;

    try {
        const response = await fetch('/api/generar-escenario', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                descripcion: descripcion
            })
        });

        if (!response.ok) {
            throw new Error('Error al generar el escenario');
        }

        const result = await response.json();

        if (result.success) {
            escenarioActual = result.texto;
            parametrosActuales = {
                raza: result.raza,
                ambiente: result.ambiente,
                extension: result.extension
            };

            vistaPreviaDiv.innerHTML = formatearTextoEscenario(escenarioActual);
            previewGroup.style.display = 'block';

            if (result.raza && result.ambiente && result.extension) {
                parametrosTextoDiv.textContent = `Raza: ${result.raza} | Ambiente: ${result.ambiente} | Extensión: ${result.extension}`;
                parametrosDetectados.style.display = 'block';
            }

            btnGuardar.disabled = false;

            // Mostrar indicador de que se usó IA
            if (result.usando_ia) {
                vistaPreviaDiv.classList.add('ia-generado');
            }
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('❌ Error al generar el escenario. Por favor, intenta de nuevo.');
        vistaPreviaDiv.innerHTML = '❌ Error al generar el escenario. Intenta de nuevo.';
        previewGroup.style.display = 'block';
    } finally {
        loadingIndicator.style.display = 'none';
        btnGenerarIA.disabled = false;
    }
}

// Función para formatear el texto del escenario
function formatearTextoEscenario(texto) {
    // Convertir saltos de línea a <br> y formato básico
    return texto.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// Cargar datos del servidor al iniciar
async function cargarDatosIniciales() {
    try {
        const response = await fetch('/api/textos');
        if (!response.ok) {
            throw new Error('No se pudo cargar los datos del servidor');
        }
        const data = await response.json();
        textos = data.textos || [];
        renderizarLista();
        console.log('Datos cargados del servidor:', textos);
    } catch (error) {
        console.error('Error al cargar datos:', error);
        mostrarError('Error al cargar los textos. Por favor, recarga la página.');
    }
}

// Mostrar mensaje de error
function mostrarError(mensaje) {
    const contenedor = document.getElementById('listaTextos');
    contenedor.innerHTML = `<div class="texto-item" style="text-align: center; color: #dc3545;">❌ ${mensaje}</div>`;
}

function renderizarLista() {
    const contenedor = document.getElementById('listaTextos');
    
    if (textos.length === 0) {
        contenedor.innerHTML = '<div class="texto-item" style="text-align: center; color: #999;">📝 No hay escenarios aún. ¡Crea tu primer escenario con IA usando el botón + Crear Escenario!</div>';
        return;
    }
    
    contenedor.innerHTML = textos.map(texto => {
        // Crear una clase basada en el ambiente (normalizar nombre)
        const ambienteClass = texto.ambiente ? `ambiente-${texto.ambiente.replace(/ /g, '-').toLowerCase()}` : '';
        return `
            <div class="texto-item ${ambienteClass}" data-id="${texto.id}">
                <div class="texto-contenido">${escapeHtml(texto.texto)}</div>
                <button class="btn-eliminar" onclick="eliminarTexto(${texto.id})">🗑️</button>
            </div>
        `;
    }).join('');
}

// Función para escapar HTML (seguridad)
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Función para guardar el escenario generado por IA
async function guardarEscenario() {
    if (!escenarioActual || !parametrosActuales) {
        alert('⚠️ Por favor, genera un escenario con IA primero.');
        return false;
    }

    const nuevoItem = {
        id: Date.now(),
        texto: escenarioActual,
        descripcion: descripcionInput.value.trim(),
        raza: parametrosActuales.raza,
        ambiente: parametrosActuales.ambiente,
        extension: parametrosActuales.extension
    };
    
    try {
        const response = await fetch('/api/textos', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(nuevoItem)
        });
        
        if (!response.ok) {
            throw new Error('Error al guardar el escenario');
        }
        
        const result = await response.json();
        if (result.success) {
            await cargarDatosIniciales(); // Recargar la lista
            return true;
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('❌ Error al guardar el escenario. Por favor, intenta de nuevo.');
        return false;
    }
}

// Función para eliminar texto
async function eliminarTexto(id) {
    if (confirm('¿Estás seguro de que quieres eliminar este escenario?')) {
        try {
            const response = await fetch(`/api/textos/${id}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error('Error al eliminar el escenario');
            }
            
            const result = await response.json();
            if (result.success) {
                await cargarDatosIniciales(); // Recargar la lista
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('❌ Error al eliminar el escenario. Por favor, intenta de nuevo.');
        }
    }
}

// Resetear el modal
function resetearModal() {
    escenarioActual = null;
    parametrosActuales = null;
    descripcionInput.value = '';
    previewGroup.style.display = 'none';
    parametrosDetectados.style.display = 'none';
    btnGuardar.disabled = true;
    vistaPreviaDiv.innerHTML = '';
    loadingIndicator.style.display = 'none';
}

// Modal functionality
const modal = document.getElementById('modalAgregar');
const btnAgregar = document.getElementById('btnAgregar');
const btnCancelar = document.getElementById('btnCancelar');
const closeBtn = document.querySelector('.close');

// Abrir modal
btnAgregar.onclick = function() {
    resetearModal();
    modal.style.display = 'block';
}

// Cerrar modal
function cerrarModal() {
    resetearModal();
    modal.style.display = 'none';
}

closeBtn.onclick = cerrarModal;
btnCancelar.onclick = cerrarModal;

// Evento para generar con IA
btnGenerarIA.onclick = generarEscenarioConIA;

// Guardar escenario
btnGuardar.onclick = async function() {
    const exito = await guardarEscenario();
    if (exito) {
        cerrarModal();
    }
}

// Cerrar modal haciendo clic fuera
window.onclick = function(event) {
    if (event.target == modal) {
        cerrarModal();
    }
}

// Inicializar la aplicación
cargarDatosIniciales();