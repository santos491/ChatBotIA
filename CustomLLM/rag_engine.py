import os
from datetime import datetime
from typing import List, Optional, Tuple

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIRECTORY,
    COLLECTION_NAME,
    RETRIEVAL_TOP_K,
    SYSTEM_PROMPT
)

from document_processor import DocumentProcessor


class RAGEngine:

    def __init__(self):
        """
        Inicializamos el motor RAG cargando todos los componentes.
        """

        print("Inicializando embeddings...")

        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url="http://localhost:11434"
        )

        print("Inicializando modelo LLM...")

        self.llm = OllamaLLM(
            model=LLM_MODEL,
            base_url="http://localhost:11434",
            temperature=0.1
        )

        self.doc_processor = DocumentProcessor()

        # ==========================
        # Base vectorial de documentos
        # ==========================

        self.vector_store: Optional[Chroma] = None

        # ==========================
        # Memoria persistente
        # ==========================

        self.memory_store = Chroma(
            persist_directory="./memory_db",
            embedding_function=self.embeddings,
            collection_name="conversation_memory"
        )

        self._load_existing_store()

    def _load_existing_store(self):
        """
        Carga la base vectorial de documentos existente.
        """

        if os.path.exists(CHROMA_PERSIST_DIRECTORY):

            try:

                self.vector_store = Chroma(
                    persist_directory=CHROMA_PERSIST_DIRECTORY,
                    embedding_function=self.embeddings,
                    collection_name=COLLECTION_NAME
                )

                count = self.vector_store._collection.count()

                if count > 0:
                    print(
                        f"Base documental cargada con {count} fragmentos."
                    )

            except Exception as e:

                print(
                    f"Error cargando base documental: {e}"
                )

                self.vector_store = None

    # =====================================================
    # MEMORIA
    # =====================================================

    def save_memory(
        self,
        question: str,
        answer: str,
        user_id: str = "default"
    ):
        """
        Guarda la conversación en memoria persistente.
        """

        memory_text = f"""
Usuario: {question}

Asistente: {answer}
"""

        self.memory_store.add_texts(
            texts=[memory_text],
            metadatas=[
                {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        )

    def retrieve_memory(
        self,
        question: str,
        user_id: str = "default",
        k: int = 3
    ) -> str:
        """
        Recupera recuerdos relevantes.
        """

        try:

            docs = self.memory_store.similarity_search(
                question,
                k=k,
                filter={"user_id": user_id}
            )

            return "\n\n".join(
                doc.page_content
                for doc in docs
            )

        except Exception as e:

            print(f"Error recuperando memoria: {e}")

            return ""

    def clear_memory(self):
        """
        Borra la memoria conversacional.
        """

        try:

            self.memory_store.delete_collection()

            self.memory_store = Chroma(
                persist_directory="./memory_db",
                embedding_function=self.embeddings,
                collection_name="conversation_memory"
            )

            print("Memoria eliminada.")

        except Exception as e:

            print(f"Error eliminando memoria: {e}")

    # =====================================================
    # DOCUMENTOS
    # =====================================================

    def add_documents(
        self,
        documents: List[Document]
    ) -> int:

        if not documents:
            return 0

        if self.vector_store is None:

            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY,
                collection_name=COLLECTION_NAME
            )

        else:

            self.vector_store.add_documents(documents)

        return len(documents)

    def ingest_file(
        self,
        file_path: str = None,
        uploaded_file=None
    ) -> Tuple[int, str]:

        documents = self.doc_processor.process_document(
            file_path=file_path,
            uploaded_file=uploaded_file
        )

        num_added = self.add_documents(documents)

        source_name = (
            uploaded_file.name
            if uploaded_file
            else os.path.basename(file_path)
        )

        return num_added, source_name

    # =====================================================
    # CONSULTA
    # =====================================================

    def query(
        self,
        question: str,
        user_id: str = "default"
    ) -> Tuple[str, List[Document]]:

        if self.vector_store is None:

            return (
                "Aún no hay documentos cargados.",
                []
            )

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": RETRIEVAL_TOP_K
            }
        )

        relevant_docs = retriever.invoke(question)

        context = "\n\n---\n\n".join([
            f"[Fuente: {doc.metadata.get('source', 'Desconocido')}, "
            f"Bloque {doc.metadata.get('chunk_id', '?')}]\n"
            f"{doc.page_content}"
            for doc in relevant_docs
        ])

        memory_context = self.retrieve_memory(
            question=question,
            user_id=user_id
        )

        prompt = f"""
{SYSTEM_PROMPT}

====================
MEMORIA PREVIA
====================

{memory_context}

====================
DOCUMENTACIÓN
====================

{context}

====================
PREGUNTA
====================

{question}

Instrucciones:

1. Prioriza la documentación.
2. Usa la memoria solo como contexto adicional.
3. Si la documentación contradice la memoria, usa la documentación.
4. Si no existe información suficiente, indícalo claramente.
"""

        answer = self.llm.invoke(prompt)

        self.save_memory(
            question=question,
            answer=answer,
            user_id=user_id
        )

        return answer, relevant_docs

    # =====================================================
    # ESTADÍSTICAS
    # =====================================================

    def get_collection_stats(self) -> dict:

        if self.vector_store is None:

            return {
                "status": "empty",
                "num_chunks": 0
            }

        try:

            count = self.vector_store._collection.count()

            memory_count = (
                self.memory_store._collection.count()
            )

            return {
                "status": "ready",
                "num_chunks": count,
                "num_memories": memory_count,
                "persist_directory": CHROMA_PERSIST_DIRECTORY
            }

        except Exception as e:

            return {
                "status": "error",
                "error": str(e)
            }

    def clear_database(self):
        """
        Elimina documentos vectorizados.
        """

        if self.vector_store is not None:

            self.vector_store.delete_collection()

            self.vector_store = None

        import shutil

        if os.path.exists(CHROMA_PERSIST_DIRECTORY):
            shutil.rmtree(CHROMA_PERSIST_DIRECTORY)

        print("Base documental eliminada.")