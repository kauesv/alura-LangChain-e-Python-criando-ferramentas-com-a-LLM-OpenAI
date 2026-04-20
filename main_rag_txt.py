from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# TextLoader - Carrega conteúdos de arquivos TXT e transforma em documentos utilizáveis no fluxo.
# PyPDFLoader - Carrega conteúdos de arquivos PDF e transforma em documentos utilizáveis no fluxo.
# FAISS - Biblioteca de busca vetorial que armazena embeddings (vetores) para similaridade e recuperação eficiente.
# RecursiveCharacterTextSplitter - Divide textos grandes em partes menores (chunks), preservando contexto para melhorar processamento e busca.

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
)

embeddings = OpenAIEmbeddings()
# Carrega os arquivos
documento = TextLoader(
    fr"D:\Meus Documentos\Documentos\projetos\alura\alura-LangChain-e-Python-criando-ferramentas-com-a-LLM-OpenAI\documentos\GTB_gold_Nov23.txt",
    encoding="utf-8"
).load()

# Divide textos
pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
).split_documents(documento)

# Recupera os dados que vão ser usados pelo modelo, base de dados vetorial
dados_recuperados = FAISS.from_documents(
    pedacos, embeddings
).as_retriever(search_kwargs={"k":2}) # quanto maior o número mais pedaços de texto, nesse caso os 2 mais semelhantes

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido"),
        ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:")
    ]
)

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke({
        "query": pergunta, "contexto":contexto
    })

print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão platinum"))