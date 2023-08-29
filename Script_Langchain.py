import os

import uvicorn
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Annoy

from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import UnstructuredHTMLLoader

from langchain.chains import RetrievalQA
import streamlit as st


os.environ['OPENAI_API_KEY'] = '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'
default_name_document = 'documento.html'

def process_document(
        path: str = 'http://catarina.udlap.mx/u_dl_a/tales/documentos/lis/rivera_l_a/capitulo4.pdf',
        is_local: bool = False,
        question: str = 'Titulo del archivo'
):
    _, loader = os.system(f'curl -o {default_name_document} {path}'), PyPDFLoader(
        f"./{default_name_document}") if not is_local \
        else PyPDFLoader(path)

    #loader = BSHTMLLoader("C:/Users/PCEO/Downloads/Documento-de-examen-Grupo1.html")
    #loader = UnstructuredHTMLLoader("C:/Users/PCEO/Downloads/Documento-de-examen-Grupo1.html")

    document = loader.load_and_split()

    print(document[-1])

    #arch = Chroma.from_documents(document, embedding=OpenAIEmbeddings())
    arch = Annoy.from_documents(document, embedding=HuggingFaceEmbeddings())


    pr = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        # tipo de procesamiento de archivo
        chain_type='map_reduce',
        retriever=arch.as_retriever()
    )
    st.write(pr.run(question))
    #print(pr.run(question))


def client():
    st.title('Manejo de LLM con Langchain')
    uploader = st.file_uploader('Subir PDF', type='pdf')

    if uploader:
        with open(f'./{default_name_document}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('¡PDF Guardado con éxito!')

    question = st.text_input('Escribir la pregunta, ejemplo: "Generar un resumen de 20 palabras sobre el documento"',
                             placeholder='Obtener respuestas sobre su pdf', disabled=not uploader)

    if st.button('Enviar Pregunta'):
        if uploader:
            process_document(
                path=default_name_document,
                is_local=True,
                question=question
            )
        else:
            st.info('Cargando PDF por defecto')
            process_document()

if __name__ == '__main__':
    client()
    #os.environ['PORT'] = '1045'
    #uvicorn.run("Script_Langchain:client", host="127.0.0.1", port=int(os.environ['PORT']), log_level="info")
    #client()

    # Se asigna el puerto pero no puede ser llamado
    #puertodef = 1045
    #uvicorn.run("Script_Langchain:client", host="127.0.0.1", port=puertodef, log_level="info")

    #process_document()
