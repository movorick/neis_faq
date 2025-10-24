# 임베딩 모델 선언하기
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os, sys, zipfile, gdown
from langchain_chroma import Chroma

# 언어 모델 불러오기
llm = ChatOpenAI(model="gpt-4o")

# Load Chroma store
url = "https://drive.google.com/uc?id=1_XS1t7NqV2AdB12bd6SJJL9NfLiWKFsT"
output = "faq_chroma.zip"
#gdown.download(url, output, quiet=False)
#with zipfile.ZipFile(output, "r") as zip_ref:
#    zip_ref.extractall("chroma_db")

# 폴더 없을 때만 다운로드 & 압축 해제
if not os.path.exists("faq_chroma"): 
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall("faq_chroma")
# ---- 2. 압축 구조 확인 & 올바른 persist_directory 찾기 ----
def find_chroma_dir(base_dir="faq_chroma"):
    for root, dirs, files in os.walk(base_dir):
        if "chroma.sqlite3" in files:
            return root
    return None

persist_directory = find_chroma_dir("faq_chroma")
if not persist_directory:
    raise RuntimeError("❌ chroma.sqlite3 not found inside extracted folder!")

print(f"✅ Loading Chroma store from: {persist_directory}")


# ---- 3. 임베딩 & 벡터스토어 로드 ----
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Create retriever
retriever = vectorstore.as_retriever(k=3)

# Create document chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser # 문자열 출력 파서를 불러옵니다.

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context} 모르면 콜센터로 문의하라고 답변해줘(064-710-0900)",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(llm, question_answering_prompt) | StrOutputParser()

# query augmentation chain
query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"), # 기존 대화 내용
        (
            "system",
            "기존의 대화 내용을 활용하여 사용자의 아래 질문의 의도를 파악하여 명료한 한 문장의 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라. :\n\n{query} 답변 끝에는 상세한 내용을 콜센터로 연락하라고 답변해줘",
        ),
    ]
)

query_augmentation_chain = query_augmentation_prompt |llm| StrOutputParser()











