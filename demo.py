# Navie rag
# sys 模块用于访问与 Python 解释器交互的功能。
import sys

# PyPDFLoader 类用于从 PDF 文件中加载文本内容和图像。
from langchain_community.document_loaders import PyPDFLoader

# RecursiveCharacterTextSplitter 类用于将文本内容分割为较小的块以提高处理效率
# 0.0.16
from langchain.text_splitter import RecursiveCharacterTextSplitter

# os 模块提供了与操作系统交互的功能。
import os

# Chroma 类用于创建文本向量存储。
from langchain_community.vectorstores import Chroma

# OpenAIEmbeddings 类用于使用 OpenAI 的 API 来获取文本的嵌入向量
from langchain_openai import OpenAIEmbeddings

# extract_images=True 表示要提取 PDF 中的图像
loader = PyPDFLoader(r"./期刊_基于wifi的室内目标检测与定位方法.pdf", extract_images=True)
pages = loader.load()
 
# print(pages[1])
print(type(pages))

# 获取了 pages 变量的内存地址，并将其打印出来。
memory_address = id(pages)
print(f"Memory Address: {memory_address}")

#  使用 sys.get_sizeof() 函数获取了 pages 变量的大小，并将其以字节为单位打印出来
object_size = sys.getsizeof(pages)
print(f"Object Size: {object_size} bytes")


text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 200,
                chunk_overlap  = 50,
                length_function = len,
            )
 
# 调用 load_and_split() 方法，将加载的 PDF 文件内容进行分割，并将结果存储在 pages_splitter 变量中
pages_splitter = loader.load_and_split(text_splitter)

object_size = sys.getsizeof(pages_splitter)
print(f"Object Size: {object_size} bytes")

print(len(pages_splitter))
# print(pages_splitter)

# api_key = "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"
# os.environ["OPENAI_API_KEY"] = api_key

api_key = "sk-Yaxfgnab670mCBVn0dC084E40a6f4bE3BeA5DbC82e292a79"
# os.environ["OPENAI_API_KEY"] = api_key




vectorstore = Chroma.from_documents(
    pages_splitter,
    embedding=OpenAIEmbeddings(openai_api_key=api_key, base_url="https://oneapi.xty.app/v1"),
)

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





 
