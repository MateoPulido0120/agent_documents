# Agente RAG con LlamaIndex para Procesamiento de Documentos

Este proyecto implementa un **agente ReAct** con funciones de **Retrieval-Augmented Generation (RAG)** para procesar archivos PDF de manera eficiente. Se utiliza **LlamaIndex** para dividir los documentos en **chunks** (resúmenes de cada página) y almacenarlos en nodos en memoria, eliminando la necesidad de bases de datos locales o en la nube para almacenar incrustaciones.

## 🛠 Tecnologías Utilizadas

*   **LlamaIndex**: Framework para estructurar, indexar y consultar documentos.
*   **GPT-4 y Gemini Flash**: Modelos de lenguaje utilizados para el procesamiento y generación de respuestas.
*   **Streamlit**: Interfaz web para cargar documentos y realizar consultas.
*   **Python**: Lenguaje de programación principal del proyecto.

## 🚀 Funcionamiento

1.  **Carga del documento PDF**: El usuario sube un archivo a la aplicación.
2.  **Procesamiento del documento**:
    *   Se divide el documento en **chunks** de texto.
    *   Cada chunk se almacena como un nodo en memoria.
3.  **Consultas sobre el documento**:
    *   El usuario realiza preguntas en lenguaje natural.
    *   El agente busca los nodos más relevantes en memoria.
    *   Se genera una respuesta basada en la información recuperada.
4.  **Respuesta en tiempo real**: No se almacena información persistente, lo que optimiza la privacidad y el rendimiento.

## 🎯 Objetivo

Facilitar la consulta eficiente de documentos sin necesidad de almacenamiento externo de incrustaciones, asegurando respuestas rápidas y precisas gracias a la combinación de RAG y modelos avanzados de IA.

---

🔹 **Desarrollado con pasión por Mateo Pulido Aponte** 🔹