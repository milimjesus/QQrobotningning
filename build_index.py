#此文件作用为生成索引和文本块，只需要每次更新本地知识库时运行即可
import os, glob, ollama, faiss, numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

knowlodge_store = r"D:\information"#存放资料的txt文件所在文件夹
model = "qwen:7b" #语言模型
embed_model = "bge-m3"#识别查找的模型
index_file = "faiss.index"
loader_map = {
    "*.txt":  lambda p: TextLoader(p, encoding="utf-8"),
}

all_docs = []
for pattern, LoaderCls in loader_map.items():
    for file in glob.glob(os.path.join(knowlodge_store, pattern)):
        docs = LoaderCls(file).load()
        for d in docs:
            d.metadata["source"] = os.path.basename(file)
        all_docs.extend(docs)

#切块并保存文本块列表文件
texts = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
).split_documents(all_docs)
np.save("texts.npy" , texts)

#向量化
vecs = []
for t in texts:
    vec = ollama.embeddings(model=embed_model, prompt=t.page_content)["embedding"]
    vecs.append(np.array(vec, dtype=np.float32))
vecs = np.vstack(vecs)

#保存FAISS索引
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, index_file)
