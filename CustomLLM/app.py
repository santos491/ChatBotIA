import streamlit as st
import time
from rag_engine import RAGEngine
from config import CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, EMBEDDING_MODEL

st.set_page_config(
    page_title="Gemini Inventarios - Asistente Virtual",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght=400;500;700&display=swap');
    
    :root {
        --bg-primary: #FFFFFF;
        --bg-secondary: #F0F4F9;
        --text-main: #1F1F1F;
        --text-sub: #444746;
        --gemini-blue: #1A73E8;
        --gemini-navy: #041E49;
    }
    
    .stApp {
        background-color: #FFFFFF;
        font-family: 'Google Sans', sans-serif;
        color: #1F1F1F;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}

    div[data-testid="stSidebarCollapseButton"] {
        position: fixed !important;
        top: 15px !important;
        left: 15px !important;
        z-index: 999999 !important;
        display: block !important;
        visibility: visible !important;
    }

    div[data-testid="stSidebarCollapseButton"] button {
        background-color: #E8F0FE !important;
        border-radius: 50% !important;
        width: 46px !important;
        height: 46px !important;
        border: 2px solid #1A73E8 !important;
        box-shadow: 0 4px 10px rgba(4, 30, 73, 0.15) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease-in-out !important;
    }

    div[data-testid="stSidebarCollapseButton"] button:hover {
        background-color: #D2E3FC !important;
        transform: scale(1.05);
    }

    div[data-testid="stSidebarCollapseButton"] svg {
        color: #041E49 !important;
        width: 24px !important;
        height: 24px !important;
    }

    section[data-testid="stSidebar"] ~ div[data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebar"] div[data-testid="stSidebarCollapseButton"] button {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] ~ div[data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebar"] div[data-testid="stSidebarCollapseButton"] svg {
        color: #FFFFFF !important;
    }

    .sidebar-content-wrapper {
        margin-top: 50px;
    }

    .welcome-container {
        max-width: 800px;
        margin: 5rem auto 2rem auto;
        text-align: left;
    }
    
    .gemini-gradient-text {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(45deg, #1A73E8, #4285F4, #041E49);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    
    .gemini-sub-text {
        font-size: 1.5rem;
        color: #444746;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .suggestion-card {
        background-color: #F0F4F9;
        padding: 1.2rem;
        border-radius: 12px;
        cursor: pointer;
        transition: background-color 0.2s ease;
        border: none;
    }
    
    .suggestion-card:hover {
        background-color: #E3E8EF;
    }
    
    .suggestion-card p {
        margin: 0;
        font-size: 0.95rem;
        color: #1F1F1F;
        line-height: 1.4;
    }

    .chat-wrapper {
        max-width: 820px;
        margin: 0 auto;
        padding: 10px;
    }
    
    [data-testid="stSidebar"] {
        background-color: #041E49 !important;
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1 {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stFileUploader {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 10px;
    }

    /* Forzar estilos visibles en las burbujas de chat nativas */
    div[data-testid="stChatMessage"] {
        background-color: #F0F4F9 !important;
        color: #1F1F1F !important;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    
    div[data-testid="stChatMessage"] p, 
    div[data-testid="stChatMessage"] span, 
    div[data-testid="stChatMessage"] div,
    div[data-testid="stChatMessage"] li {
        color: #1F1F1F !important;
    }

    div[data-testid="stChatMessageContent"] {
        color: #1F1F1F !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_engine():
    return RAGEngine()


def main():
    try:
        rag_engine = initialize_rag_engine()
    except Exception as e:
        st.error(f"**🚨 ERROR DE CONEXIÓN:** {str(e)}")
        return

    stats = rag_engine.get_collection_stats()

    with st.sidebar:
        st.markdown('<div class="sidebar-content-wrapper">', unsafe_allow_html=True)
        st.markdown("<h2 style='font-weight:700;'>📦 Datos de Inventario</h2>", unsafe_allow_html=True)
        st.markdown("Sube los reportes, PDFs de stock o manuales de almacén para alimentar al asistente.")
        st.markdown("---")
        
        uploaded_files = st.file_uploader(
            "Cargar documentos de inventario",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Actualizar Base Virtual", type="primary", use_container_width=True):
                with st.spinner("Procesando datos de inventario..."):
                    total_chunks = 0
                    for uploaded_file in uploaded_files:
                        try:
                            num_chunks, source = rag_engine.ingest_file(uploaded_file=uploaded_file)
                            total_chunks += num_chunks
                        except Exception as e:
                            st.sidebar.error(f"Error en {uploaded_file.name}: {str(e)}")
                    
                    if total_chunks > 0:
                        st.sidebar.success(f"Indexados {total_chunks} fragmentos con éxito.")
                        time.sleep(1)
                        st.rerun()
        
        st.markdown("---")
        st.caption(f"**Estatus:** Connected")
        st.caption(f"**Bloques en memoria:** {stats.get('num_chunks', 0)}")
        
        if stats.get('num_chunks', 0) > 0:
            if st.button("🗑️ Purgar Datos", type="secondary", use_container_width=True):
                rag_engine.clear_database()
                st.cache_resource.clear()
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-container">
            <h1 class="gemini-gradient-text">Hola, soy tu Asistente de Inventarios</h1>
            <p class="gemini-sub-text">¿En qué puedo ayudarte a consultar hoy sobre el stock o almacén?</p>
        </div>
        """, unsafe_allow_html=True)
        
        if stats.get("num_chunks", 0) > 0:
            st.markdown("""
            <div class="chat-wrapper">
                <div class="suggestion-grid">
                    <div class="suggestion-card"><p>📊 "¿Cuál es el balance actual de existencias del último reporte?"</p></div>
                    <div class="suggestion-card"><p>🔍 "Busca si hay discrepancias o alertas de stock mínimo"</p></div>
                    <div class="suggestion-card"><p>🚚 "Procedimiento para ingresos de mercancía en tránsito"</p></div>
                </div>
            </div>
            <br>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="chat-wrapper" style="text-align: center; color: #64748B; padding: 2rem; background: #F0F4F9; border-radius:12px;">
                💡 <strong>El sistema está listo:</strong> Usa el botón azul redondo de la esquina superior izquierda para desplegar el panel y arrastrar tus archivos.
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    
    # Historial del chat limpio sin bloques de fuentes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Captura e interacción con el RAG
    if prompt := st.chat_input("Escribe tu consulta sobre productos, códigos o stock..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analizando base de datos de inventario..."):
                start_time = time.time()
                # El backend sigue obteniendo las fuentes para operar, pero ya no las pintamos
                answer, sources = rag_engine.query(
                question=prompt,
                user_id="inventarios"
                                    )
                elapsed_time = time.time() - start_time
                
            st.markdown(answer)
            st.caption(f"⏱️ Respuesta generada localmente en {elapsed_time:.2f} segundos.")
                        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
        
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()