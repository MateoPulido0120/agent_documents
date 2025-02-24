import streamlit as st
from pypdf import PdfReader
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleObjectNodeMapping
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import io
import time
from utils.logging_config import logger

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
            logger.error("Generaciión de contenido por paginas incorrecto")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_global_content(text, llm_model):
    try:
        start_time = time.time()
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
            logger.error("Generación de resumen global del contenido incorrecto")

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
            logger.error("Generación de respuesta segun consulta incorrecto")

def create_object_retriever(list_objects, embed_model):
    try:
        start_time = time.time()
        # object-node mapping and nodes
        obj_node_mapping = SimpleObjectNodeMapping.from_objects(list_objects)
        nodes = obj_node_mapping.to_nodes(list_objects)

        # object index
        object_index = ObjectIndex(
            index=VectorStoreIndex(nodes=nodes, embed_model=embed_model),
            object_node_mapping=obj_node_mapping,
        )
        
        logger.info(f"Creación de objeto recuperador correcto, tiempo: {time.time()-start_time} s")
        return object_index.as_retriever(similarity_top_k=3)

    except Exception as e:
            logger.error("Creación de objeto recuperador incorrecto")

def process_data(bytes_data, llm_model, embed_model):
    try:
        start_time = time.time()
        pdf_bytes = io.BytesIO(bytes_data)
        reader = PdfReader(pdf_bytes)
        
        pages_content = []
        c = 0
        for page in reader.pages:
            c+=1
            pages_content.append(generate_content_pages(page.extract_text(), llm_model))

        global_content = "\n".join([element['content'] for element in pages_content])
        resume_global_content = generate_global_content(global_content, llm_model)

        object_retriever = create_object_retriever(pages_content, embed_model)

        logger.info(f"Procesamiento de archivo correcto, tiempo: {time.time()-start_time} s")
        return resume_global_content, object_retriever

    except Exception as e:
            logger.error("rocesamiento de archivo incorrecto")


def parameterize_agent(resume_global_content, object_retriever, llm_model, memory):
    def extract_relevant_content(question: str) -> list:
        """Extract relevant content from context document"""
        response_reliable = object_retriever.retrieve(question)
        return generate_response(question, response_reliable)

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
            logger.error("Parametrización de agente incorrecta")
             

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
            logger.error("Generación de respuesta segun consulta incorrecto")
