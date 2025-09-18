from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
loader=PyPDFLoader('faq.pdf')
faq_data=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
all_splits=text_splitter.split_documents(faq_data)
embedding=OpenAIEmbeddings(model='text-embedding-3-large',api_key=OPEN_API_KEY)
llm=ChatOpenAI(model='gpt-4o')
persist_dir='chroma_faq'
if not os.path.exists(persist_dir):
    print('Creating new Chroma store')
    vectorstore=Chroma.from_documents(
        documents=all_splits,
        embedding=embedding,
        persist_directory=persist_dir
    )
    #vectorstore.persist()
else:
    print('loading existing Chroma Store')
    vectorstore=Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
    )

retriever=vectorstore.as_retriever(k=3)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
question_answering_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system","사용자의 질문에 대해 아래 context에 기반하여 답변해. 모르면 콜센터로 문의하라고 답변해. :\n\n {context}"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)
document_chain=create_stuff_documents_chain(llm,question_answering_prompt)|StrOutputParser()

query_augmentation_prompt=ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "system","기본의 대화 내용을 활용하여 사용자의 질문한 의도를 파악해서 한 문장의 명료한 질문으로 변환해. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현해: \n:\n{query}",
        ),
    ]
)
query_augmentation_chain=query_augmentation_prompt|llm|StrOutputParser()





