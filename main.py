import streamlit as st
import pandas as pd
import google.generativeai as genai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from utils.agent import process_data, parameterize_agent, process_data_test
from utils.logging_config import logger
import time

def initialize_session():
    if "gemini_llm" not in st.session_state:
        genai.configure(api_key=st.secrets["API_GEMINI"])
        generation_config=genai.GenerationConfig(response_mime_type="application/json", candidate_count=1, temperature=0, max_output_tokens=1000)
        safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
 
        st.session_state['gemini_llm'] = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config, safety_settings=safety_settings) 

    if "openai_llm" not in st.session_state:
        st.session_state['openai_llm'] = OpenAI(model="gpt-4", api_key=st.secrets["API_OPENAI"])
    
    if "embed_model" not in st.session_state:
        st.session_state['embed_model'] = OpenAIEmbedding(model="text-embedding-3-large", api_key=st.secrets["API_OPENAI"])

    if "memory" not in st.session_state:
        st.session_state['memory'] = ChatMemoryBuffer.from_defaults(llm=st.session_state['openai_llm'], token_limit=8000)

    if 'object_retriever_agent' not in st.session_state:
        st.session_state['object_retriever_agent'] = None

    if 'resume_global_content' not in st.session_state:
        st.session_state['resume_global_content'] = None

    if "messages_agent" not in st.session_state:
        st.session_state['messages_agent'] = []

    if 'agent' not in st.session_state:
        st.session_state['agent'] = None

    if 'object_retriever_test' not in st.session_state:
        st.session_state['object_retriever_test'] = None

    if 'resume_global_content_test' not in st.session_state:
        st.session_state['resume_global_content_test'] = None

    if 'test_file' not in st.session_state:
        st.session_state['test_file'] = None


if __name__ == "__main__":

    initialize_session()

    st.set_page_config(layout="wide")

    st.title("Agente ReAct - procesador de documentos (RAG)")

    tab1, tab2 = st.tabs(["Agente", "Evaluación"])
    
    with tab1:
        with st.expander("Instrucciones de Uso"):
            st.markdown("""
            ## 1. Carga del Archivo PDF

            - **Formato**: Documento en formato PDF.
            - **Contenido**: Este documento será procesado para permitir consultas sobre su contenido.
            - **Carga**: Sube el archivo a la aplicación mediante el botón de carga.

            ## 2. Realización de Consultas

            Una vez cargado el archivo PDF, puedes comenzar a realizar preguntas sobre su contenido.

            - **Escribe tu consulta** en el cuadro de entrada de texto.
            - La aplicación analizará el documento y proporcionará la respuesta basada en el contenido del PDF.

            ## 3. Resultados

            - Se generará una respuesta basada en la información extraída del documento.
            - La precisión de la respuesta dependerá de la claridad de la consulta y la calidad del contenido en el PDF.
            """)

        with st.expander("Logs"):
            with open("static/app_logs.log", "r") as log_file:
                logs = log_file.read()

            st.text_area("Registro de Logs", logs, height=300)

        uploaded_file_agent = st.file_uploader("Elige un archivo PDF", accept_multiple_files=False, type="PDF", key="file_agent")
        if uploaded_file_agent is not None:
            if st.session_state['object_retriever_agent'] is None:
                bytes_data = uploaded_file_agent.getvalue()
                with st.spinner(text="Procesando archivo..."):
                    st.session_state['resume_global_content'], st.session_state['object_retriever_agent']  = process_data(bytes_data, 
                                                                                                                          st.session_state['gemini_llm'],
                                                                                                                          st.session_state['embed_model'])
                    st.toast('Procesado!', icon='✅')

        if st.session_state['resume_global_content'] is not None and st.session_state['object_retriever_agent'] is not None:
            st.session_state['agent'] = parameterize_agent(st.session_state['resume_global_content'], 
                                                                  st.session_state['object_retriever_agent'],
                                                                  st.session_state['openai_llm'],
                                                                  st.session_state['memory'])
            if prompt := st.chat_input("Pregunta lo que quieras..."):
                # Add user message to chat history
                st.session_state['messages_agent'].append({"role": "user", "content": prompt})

                start_time = time.time()
                response = st.session_state['agent'].chat(prompt)
                logger.info(f"Generación de respuesta por parte del agente correcta, tiempo: {time.time()-start_time} s")

                # Add assistant response to chat history
                st.session_state['messages_agent'].append({"role": "assistant", "content": response})

            for message in st.session_state['messages_agent']:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    with tab2:
        with st.expander("Instrucciones de Uso"):
            st.markdown("""
            ## 1. Carga de Archivos

            Para utilizar la aplicación, debes cargar dos archivos:

            ### a) Archivo PDF

            - **Formato**: Documento en formato PDF.
            - **Contenido**: Este documento será procesado para extraer información relevante.
            - **Carga**: Sube el archivo a la aplicación mediante el botón de carga.

            ### b) Archivo Excel

            - **Formato**: Archivo en formato `.xlsx`.
            - **Estructura**: El archivo debe contener exactamente **tres columnas** con la siguiente estructura:
            1. **Preguntas**: Contiene preguntas relacionadas con el documento PDF cargado.
            2. **Respuestas**: Contiene las respuestas esperadas para cada pregunta.
            3. **Validez (Booleano)**: Contiene valores `True` o `False` que indican si la pregunta y respuesta corresponden al contexto del documento.
            - **Ejemplo de formato:**
            """)

            data = {
                "Pregunta": [
                    "¿Cuál es el tema principal del documento?",
                    "¿Cuál es la capital de Francia?"
                ],
                "Respuesta": [
                    "Inteligencia Artificial",
                    "París"
                ],
                "Validez": [True, False]
            }

            df = pd.DataFrame(data)
            st.table(df)

            st.markdown("""
            ## 2. Procesamiento de los Datos

            Una vez cargados los archivos:

            1. La aplicación procesará el contenido del **PDF**.
            2. Comparará las preguntas y respuestas proporcionadas en el **Excel** con la información extraída.
            3. Evaluará el rendimiento del agente basándose en los valores booleanos de la tercera columna.

            ## 3. Resultados

            - Se generará un análisis del desempeño basado en la comparación de respuestas.
            - Se mostrarán estadísticas sobre la precisión del agente en relación con el contexto del documento.
            """)
        
        uploaded_file_test = st.file_uploader("Elige un archivo PDF", accept_multiple_files=False, type="PDF", key="file_test_agent")
        if uploaded_file_test is not None:
            if st.session_state['object_retriever_test'] is None:
                bytes_data = uploaded_file_test.getvalue()
                with st.spinner(text="Procesando archivo de test..."):
                    st.session_state['resume_global_content_test'], st.session_state['object_retriever_test']  = process_data(bytes_data, 
                                                                                                                          st.session_state['gemini_llm'],
                                                                                                                          st.session_state['embed_model'])
                    st.toast('Procesado!', icon='✅')


        uploaded_file_answer = st.file_uploader("Elige un archivo XLS de 3 columnas", accept_multiple_files=False, type="xls", key="file_test")
        if uploaded_file_answer is not None:
            if st.session_state['test_file'] is None:
                bytes_data = uploaded_file_answer.getvalue()
                with st.spinner(text="Procesando archivo de test..."):
                    st.session_state['test_file'] = process_data_test(bytes_data, 
                                                                st.session_state['gemini_llm'],
                                                                st.session_state['embed_model'])
                    st.toast('Procesado!', icon='✅')
