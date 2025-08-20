​
参考：用python实现自动回复QQ消息——不到60行-CSDN博客

一直想做个qq自动回复的机械人来自娱自乐，网上有成熟方案，但不愿冒封号的危险，偶然看到这篇文章，来了灵感

## 1、需要安装的模块
我测试的python版本是3.10，模块可以安装到conda虚拟环境里避免和其它项目冲突

1.pyautogui：让程序自动控制鼠标和键盘

2.pyperclip：复制剪贴板里的内容，向剪贴板写入内容

3.psutil：获取运行的进程信息

4.langchain：一个用于开发由语言模型驱动的应用程序的框架，提供了模块化的组件和可定制化的用例链

5.langchain-community

6.ollama 进行与ollama部署的模型的交互

7.numpy

8.faiss：用来检索本地知识库

## 2、下载Ollama并部署模型
本项目主要用到了两个模型：

qwen:7B 轻量大语言模型，一般的独显应该都能部署

bge-m3 对多语言、长文本等进行检索的模型

Ollama安装包应使用命令安装，不然会下到C盘：

下载安装文件后，用命令行安装即可：
.\OllamaSetup.exe /DIR = "D:\Ollama"

安装完毕后在Ollama设置中将模型存放位置改到D盘

下载模型命令：

ollama pull qwen:7B
ollama pull bge-m3

等待下载完成（可能有些慢）

## 3、 本地知识库处理
创建一个build_index文件，作用为生成索引和文本块，只需要每次更新本地知识库时运行一次即可

导入模块：

```python
import os, glob, ollama, faiss, numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
```

设置变量：

```python
knowlodge_store = r"D:\information"#存放资料的txt文件所在文件夹
model = "qwen:7b" #语言模型
embed_model = "bge-m3"#识别查找的模型
index_file = "faiss.index"
loader_map = {
    "*.txt":  lambda p: TextLoader(p, encoding="utf-8"),
}#处理txt文件，可自行添加其他格式的处理工具
```

读取文件（演示代码里处理的知识库是txt文件，也可以自行改成其他格式的）：

```python
all_docs = []
for pattern, LoaderCls in loader_map.items():
    for file in glob.glob(os.path.join(knowlodge_store, pattern)):
        docs = LoaderCls(file).load()
        for d in docs:
            d.metadata["source"] = os.path.basename(file)
        all_docs.extend(docs)
```
切块并保存文本块列表文件：

```python
texts = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
).split_documents(all_docs)
np.save("texts.npy" , texts)
```
向量化从而可以使faiss读取内容并处理:

```python
vecs = []
for t in texts:
    vec = ollama.embeddings(model=embed_model, prompt=t.page_content)["embedding"]
    vecs.append(np.array(vec, dtype=np.float32))
vecs = np.vstack(vecs)
```
保存faiss索引文件：
```python
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, index_file)
```

文本块列表文件和索引文件生成在python文件同一文件夹下，如果更新了本地知识库，再次运行一遍上述代码即可。

  4、主代码
导入需要的模块：
```python
import re
import time
import psutil
import pyautogui
import pyperclip
from langchain.chains import ConversationChain
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
import pathlib
import os, ollama, faiss, numpy as np
```
定义需要的变量：

ollama的模型端口默认为http://localhost:11434

```python
# 定义一些变量
system = (
    ""
)#人设，可自行填写或不写
qwen7b = ChatOllama(base_url="http://localhost:11434",model= "qwen:7b")
model = qwen7b
embed_model = "bge-m3"
QQ_path = r"D:\QQ\QQ.exe"
index_path = r"D:\PyCharm\pyproject\QQ\faiss.index"
loop = 2          # 每 2 秒询问一次
history_file = pathlib.Path(__file__).with_name("chat_history.json")#存储记忆以前的对话内容
history = FileChatMessageHistory(str(history_file))
index_file = "faiss.index"  # FAISS 索引文件
texts_file = "texts.npy"    # 文本块列表文件
qq_message_pic = r"D:\pics.png"
```
接下来是定义函数，首先是load_index_and_texts函数，作用是读取我们上面生成的本地知识库索引和文本块文件：
```python
def load_index_and_texts():
    #加载 FAISS 索引
    index = faiss.read_index(index_file)
    #加载文本块列表
    texts = np.load(texts_file, allow_pickle=True)
    return index, texts.tolist()
```
然后是strip_between函数，作用是去除思考过程（如果有的话）：
```python
def strip_between(text: str, start: str, end: str) -> str:
    #去掉思考过程
    pattern = re.escape(start) + r".*?" + re.escape(end)
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()
```
接着是两个用以判断qq是否运行的函数：
```python
def is_qq_running() -> bool:
    #利用psutil判断 QQ.exe 是否存在进程
    return any(p.name().lower() == "qq.exe" for p in psutil.process_iter())

def open_qq():
    #如未运行则启动 QQ，并等待登录完成
    if not is_qq_running():
        print("未检测到 QQ，正在启动…")
        os.startfile(QQ_path)
        # 等待登录界面出现（可根据自己电脑调整）
        time.sleep(15)
    else:
        print("QQ运行中")
```
接着是两个用来判断是否有新消息的函数，原理是利用有新消息时qq图标变红，从而被pyautogui识别，有新消息时的图标图片请自行截图：
```python
def locate_and_click(image_path: str, confidence=0.99, timeout=10):
    #在屏幕上找到qq图标并单击，需要opencv
    start = time.time()
    while time.time() - start < timeout:
        pos = pyautogui.locateOnScreen(image_path, confidence=confidence)
        if pos:
            center = pyautogui.center(pos)
            pyautogui.click(center)
            return center
        time.sleep(0.5)
    raise RuntimeError(f"未找到图片: {image_path}")

def find_qq():
    #找到qq并点击
    try:
        pos = locate_and_click(qq_message_pic, timeout=2)
        return pos
    except RuntimeError:
        return None
```
接着是获取并发送消息的函数，原理是qq电脑版打开时最新消息和发送框位置固定，也可以用opencv识别，我这里直接用了坐标，可自行修改：
```python
def get_latest_msg() -> str:
    #打开会话后，通过坐标定位最新消息，通过Ctrl+C 复制聊天内容
    time.sleep(0.1)
    pyautogui.moveTo(563, 1154, duration=1)
    pyautogui.click()
    pyautogui.hotkey("ctrl", "c")
    time.sleep(0.2)
    text = pyperclip.paste()
    return text

def send_msg(text: str):
    #把text粘到剪切板
    pyperclip.copy(text)
    time.sleep(0.2)
    #找到对话框，发送ai回复的数据
    pyautogui.moveTo(1015, 1341, duration=1)
    pyautogui.click()
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)
    pyautogui.press("enter")

def close_session():
    """关闭当前聊天窗口（Ctrl+W）"""
    pyautogui.hotkey("ctrl", "w")
    time.sleep(0.5)
```
接着是对接ai的函数，原理是将剪切板中复制的消息发送给ai，ai会首先在本地资源库的文本块中查找，如果匹配度达到阈值，则将文本输入预设从而回答问题，如果未达到阈值，则直接调用大模型本身知识回答。
```python
def ask(query: str, threshold: float = 0.45):
    index, texts = load_index_and_texts()
    index = faiss.read_index(index_path)
    q_vec = np.array(
        ollama.embeddings(model=embed_model, prompt=query)["embedding"],
        dtype=np.float32
    ).reshape(1, -1)
    #让FAISS返回分数+下标
    D, I = index.search(q_vec, 3)      #D是相似度分数
    best_score = D[0][0]                   #最高分
    #阈值判断：低于阈值 = 本地知识库没相关内容
    if best_score < threshold:
        prom = f""
    else:
        ctx = "\n".join(texts[i].page_content for i in I[0])
        prom = f"根据以下资料回答：\n{ctx}\n\n问题"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system + prom),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | model
    with_memory = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="messages"
    )
    m = with_memory.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"session_id": "user_001"}}
    )
    return m
```
最后是运行流程的主要函数：
```python
def main():
    open_qq()
    while True:
        try:
            #检测是否有消息，如没有则等待
            pos = find_qq()
            if not pos:
                time.sleep(loop)
                continue

            #找到发消息的人
            time.sleep(0.1)
            pyautogui.moveTo(271, 115, duration=1)
            pyautogui.click()

            #读取最新消息
            msg = get_latest_msg()
            if not msg:
                close_session()
                continue

            #调用本地大模型回答
            mes = ask(msg)
            word = mes.content

            #去掉思考过程
            word = strip_between(word, "<think>", "</think>")
            word = word.replace("\n" and "*","")
            reply = word

            #发送回复
            send_msg(reply)

            #最小化窗口
            pyautogui.hotkey("alt", "esc")

        except Exception as e:
            print("等待回复中", e)
            time.sleep(loop)
```
循环代码：
```python
if __name__ == "__main__":
    time.sleep(loop)
    main()
```
'''
运行时确保Ollama和QQ均开启，如果要调整速度，可以修改sleep的时间

​
