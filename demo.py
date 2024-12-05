# 导入所需模块和类，用于实现对PDF文档的操作、文本分割、相似度计算、向量存储以及大模型的交互

# 导入Python系统相关的模块，允许访问和控制脚本运行时环境，
# 包括但不限于命令行参数、模块搜索路径、标准输入输出等。
import sys

# 导入OpenAI库中的主接口类或模块，用于与OpenAI的API服务进行交互。
# 通过初始化OpenAI客户端，可以方便地调用OpenAI提供的各种AI功能，
# 如自然语言生成、模型推理等。
from openai import OpenAI

# 导入httpx第三方库，这是一个高效且功能丰富的HTTP客户端，
# 支持同步和异步操作，用于执行HTTP请求，与OpenAI或其他HTTP API进行通信。
import httpx

# 导入自定义的FlagReranker类，用于对检索结果进行重新排序评分
from FlagEmbedding import FlagReranker

# 导入langchain库中的RecursiveCharacterTextSplitter类，用于递归字符级别的文本分割
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 导入langchain_community库中的PyPDFLoader类，用于加载和提取PDF文档内容
from langchain_community.document_loaders import PyPDFLoader

# 导入langchain_community.vectorstores中的Chroma类，用于构建基于文档内容的向量存储
from langchain_community.vectorstores import Chroma

# 导入langchain_openai库，用于获取OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings

# 加载PDF文档，并保存页面内容
loader = PyPDFLoader(r"./期刊_基于wifi的室内目标检测与定位方法.pdf", extract_images=True)
pages = loader.load()  # 获取PDF的所有页面内容
print(pages[1])  # 打印第二页的内容
print(type(pages))  # 输出pages变量的类型（通常是列表或其他可迭代的对象）

# 输出pages变量所占内存地址
memory_address = id(pages)
print(f"Memory Address: {memory_address}")

# 输出pages变量所占用的内存大小
object_size = sys.getsizeof(pages)
print(f"Object Size: {object_size} bytes")

# 创建一个文本分割器，设置分块大小、重叠部分大小及长度度量函数
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, length_function=len)

# 使用文本分割器对PDF页面内容进行细粒度分割
pages_splitter = loader.load_and_split(text_splitter)

# 输出分割后文本所占用的内存大小
object_size = sys.getsizeof(pages_splitter)
print(f"Object Size: {object_size} bytes")

# 输出分割后的文本片段数量
print(len(pages_splitter))

# 注释掉输出具体分割内容的代码行

# 指定OpenAI API密钥（这里替换为实际API密钥）
api_key = "sk...",

# 使用OpenAI Embeddings构建向量知识库，存储分割后的文档内容
vectorstore = Chroma.from_documents(
    pages_splitter,
    embedding=OpenAIEmbeddings(openai_api_key=api_key, base_url="https://oneapi.xty.app/v1"),
)

# 设置要查询的问题
query = "根据采集信号的不同，基于WiFi的室内定位技术可分为哪两种？"

# 在向量知识库中进行相似性搜索，找出与查询最相关的文档片段
docs = vectorstore.similarity_search(query)
# 获取相似片段及其对应的相似度得分
docs_and_scores = vectorstore.similarity_search_with_score(query)

# 打印前四个最相关的文档片段内容
for i in range(4):
    print(docs[i].page_content)

# 初始化FlagReranker模型，用于对检索结果进行重排序
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)  # 使用更大的模型并开启混合精度加速

# 计算每个检索结果与查询之间的新评分
scores = reranker.compute_score([
    [query, docs[0].page_content],
    [query, docs[1].page_content],
    [query, docs[2].page_content],
    [query, docs[3].page_content]
])

# 找到评分最高的文档片段索引
value = max(scores)
score_index = scores.index(max(scores))

# 输出评分最高的文档片段内容
print(docs[score_index].page_content)

# 使用OpenAI客户端发起对话请求，模型为gpt-3.5-turbo
client = OpenAI(
    base_url="https://oneapi.xty.app/v1",
    api_key="sk...",
    http_client=httpx.Client(base_url="https://oneapi.xty.app/v1", follow_redirects=True),
)

# 构建对话历史，包含问候语、用户问题和找到的相关文档片段
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Hello!"},
        {"role": "user", "content": f"问题：{query}/n已知：{docs[score_index].page_content}"},
    ]
)

# 打印一些调试信息
print()
print(completion.choices)  # 打印模型生成的回复选项
