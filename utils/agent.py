import streamlit as st
from pypdf import PdfReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import io
import time
from utils.logging_config import logger
import traceback

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_content_pages(text, llm_model):
    try:
        start_time = time.time()
        response = llm_model.generate_content(
            f"""Extrae contenido relevante del texto y al menos 3 puntos clave: {text}
            usando el esquema JSON:
            {{
                "content": str,
                "mainly_points": list,
            }}:
            """).text
        
        response = json.loads(response)
        logger.info(f"Generaciión de contenido por paginas correcto, tiempo: {time.time()-start_time} s")
        return response

    except Exception as e:
            traceback.print_exc()
            logger.error(f"Generación de contenido por paginas incorrecto: {e}")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=10, max=20))
def generate_global_content(text, llm_model):
    try:
        start_time = time.time()
        time.sleep(5)
        response = llm_model.generate_content(
            f"""genera un resumen global de todo el contenido (no más de 50 palabras): {text}
            usando el esquema JSON:
            {{
                "content": str,
            }}:
            """).text
        
        response = json.loads(response)
        logger.info(f"Generación de resumen global del contenido correcto, tiempo: {time.time()-start_time} s")
        return response["content"]

    except Exception as e:
            logger.error(f"Generación de resumen global del contenido incorrecto: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_response(query, context):
    try:
        start_time = time.time()
        response = st.session_state['gemini_llm'].generate_content(
            f"""Responde la pregunta: {query}
            usando el contexto: {context}
            usando el esquema JSON:
            {{
                "response": str,
            }}:
            """).text
        
        response = json.loads(response)
        logger.info(f"Generación de respuesta segun consulta correcto, tiempo: {time.time()-start_time} s")
        return response["response"]

    except Exception as e:
            logger.error(f"Generación de respuesta segun consulta incorrecto: {e}")

def create_object_retriever(pages_summary_content, pages_content, embed_model):
    try:
        start_time = time.time()

        nodos = []
        node_parser = SimpleNodeParser()

        for i in range(len(pages_content)):
            main_text = str(pages_summary_content[i]["mainly_points"])
            metadata = {"content": pages_summary_content[i]["content"]}
            nodo = node_parser.get_nodes_from_documents(
                [Document(text=main_text, metadata=metadata)]
            )[0]
            nodos.append(nodo)

        index = VectorStoreIndex(nodes=nodos, embed_model=embed_model)
        retriever = index.as_retriever(similarity_top_k=3)
        
        logger.info(f"Creación de objeto recuperador correcto, tiempo: {time.time()-start_time} s")
        return retriever

    except Exception as e:
            traceback.print_exc()
            logger.error(f"Creación de objeto recuperador incorrecto: {e}")

def process_data(bytes_data, llm_model, embed_model):
    try:
        start_time = time.time()
        pdf_bytes = io.BytesIO(bytes_data)
        reader = PdfReader(pdf_bytes)
        
        pages_content = []
        pages_summary_content = []

        c = 0
        for page in reader.pages:
            c+=1
            content = page.extract_text()
            pages_content.append(content)
            pages_summary_content.append(generate_content_pages(content, llm_model))

        none_indexes = []

        for i in range(len(pages_summary_content)):
            if pages_summary_content[i] is None:
                none_indexes.append(i)

        pages_summary_content = [element for i, element in enumerate(pages_summary_content) if i not in none_indexes]
        pages_content = [element for i, element in enumerate(pages_content) if i not in none_indexes]

        global_content = "\n".join([element['content'] for element in pages_summary_content])
        resume_global_content = generate_global_content(global_content, llm_model)

        object_retriever = create_object_retriever(pages_summary_content, pages_content, embed_model)

        logger.info(f"Procesamiento de archivo correcto, tiempo: {time.time()-start_time} s")

        return resume_global_content, object_retriever

    except Exception as e:
            traceback.print_exc()
            logger.error(f"Procesamiento de archivo incorrecto: {e}")


def parameterize_agent(resume_global_content, object_retriever, llm_model, memory):
    def extract_relevant_content(question: str) -> list:
        """Extract relevant content from context document"""
        response_reliable = object_retriever.retrieve(question)

        if response_reliable:
            metadata_response = [{"content": res.metadata, "mainly_points": res.text} for res in response_reliable] if response_reliable else []
            return generate_response(question, metadata_response)

        return generate_response(question, {"mainly_points": None,
                        "content": None})

    try:
        start_time = time.time()
        extract_relevant_content_tool = FunctionTool.from_defaults(fn=extract_relevant_content)

        prompt_context = f"""
            - Eres un experto en {resume_global_content}.
            - Su objetivo es responder todas las preguntas del usuario hasta que tenga total claridad sobre el tema.
            - Utilice la herramienta 'extract_relevant_content_tool' para extraer contenido relevante para responder las preguntas.
            - NO DEBE INVENTAR NI ASUMIR RESPUESTAS.
            - NO DEBE RESPONDER PREGUNTAS QUE NO ESTEN RELACIONADS AL TEMA.
        """
        agent = ReActAgent.from_tools([extract_relevant_content_tool], llm=llm_model, context=prompt_context, 
                                    verbose=True, max_iterations=10, memory=memory)
        
        logger.info(f"Parametrización de agente correcta, tiempo: {time.time()-start_time} s")
        return agent

    except Exception as e:
            logger.error(f"Parametrización de agente incorrecta: {e}")
             

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_evaluation(query, real_response, generate_response):
    try:
        start_time = time.time()
        response = st.session_state['gemini_llm'].generate_content(
            f"""Realiza un analisis comparativo de la pregunta: {query}
            Y lo que deberia ser la respuesta: {real_response}
            Con la respuesta generada: {generate_response}
            - Valida si las respuestas (real y generada) son similares o responden a lo mismo contextualmente, retorna un bool.
            - Retorna un analisis de tu decision en formato str.
            usando el esquema JSON:
            {{
                "validacion": bool,
                "analisis": str
            }}:
            """).text
        
        response = json.loads(response)
        logger.info(f"Generación de evalución segun consulta correcto, tiempo: {time.time()-start_time} s")
        return response["validacion"], response["analisis"]

    except Exception as e:
            logger.error(f"Generación de respuesta segun consulta incorrecto: {e}")
