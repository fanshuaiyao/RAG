# 这是一个RAG项目，实现对pdf的分割，相似度匹配和重排序，然后将得分最高的提示词喂给大模型

# reranker rag
import sys

from FlagEmbedding import FlagReranker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

loader = PyPDFLoader(r"./期刊_基于wifi的室内目标检测与定位方法.pdf", extract_images=True)
pages = loader.load()

print(pages[1])
print(type(pages))

memory_address = id(pages)
print(f"Memory Address: {memory_address}")

object_size = sys.getsizeof(pages)
print(f"Object Size: {object_size} bytes")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
)

pages_splitter = loader.load_and_split(text_splitter)

object_size = sys.getsizeof(pages_splitter)
print(f"Object Size: {object_size} bytes")

print(len(pages_splitter))
# print(pages_splitter)



vectorstore = Chroma.from_documents(
    pages_splitter,
    # embedding=OpenAIEmbeddings(openai_api_key=api_key, base_url="https://oneapi.xty.app/v1"),
)

# 这里可以修改问题
query = "根据采集信号的不同，基于WiFi的室内定位技术可分为哪两种？"
docs = vectorstore.similarity_search(query)
docs_and_scores = vectorstore.similarity_search_with_score(query)
# print(docs[3].page_content)
print()
print()
print(docs[0].page_content)
print()
print()
print(docs[1].page_content)
print()
print()
print(docs[2].page_content)
print()
print()
print(docs[3].page_content)

print()


# 重排序
# reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)  # use fp16 can speed up computing
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
# print(11111111111)
scores = reranker.compute_score([[query, docs[0].page_content], [query, docs[1].page_content],
                                 [query, docs[2].page_content],
                                 [query, docs[3].page_content]])

print(scores)


# 得到最大分数的切片
value = max(scores)
score_index = scores.index(max(scores))
print(docs[score_index].page_content)

print()

from openai import OpenAI
import httpx

client = OpenAI(
    base_url="https://oneapi.xty.app/v1",
    api_key="sk...",
    http_client=httpx.Client(
        base_url="https://oneapi.xty.app/v1",
        follow_redirects=True,
    ),
)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        # 第一条消息，表示系统向用户打招呼。
        {"role": "system", "content": "Hello!"},
        # 第一条消息，表示系统向用户打招呼或提问。
        {"role": "user", "content": "问题："+query+"/n已知："+docs[score_index].page_content},
    ]
)

print('123')
print(completion.choices)
