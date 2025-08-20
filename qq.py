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

# 定义一些变量
system = (
    "你是绫地宁宁，一名魔女，性格温柔，乐于助人"
)#人设
qwen7b = ChatOllama(base_url="http://localhost:11434",model= "qwen:7b")
model = qwen7b
embed_model = "bge-m3"
QQ_path = r"D:\QQ\QQ.exe"
index_path = r"D:\PyCharm\pyproject\QQ\faiss.index"
loop = 2          # 每 2 秒询问一次
history_file = pathlib.Path(__file__).with_name("chat_history.json")
history = FileChatMessageHistory(str(history_file))
index_file = "faiss.index"  # FAISS 索引文件
texts_file = "texts.npy"    # 文本块列表文件
qq_message_pic = r"D:\pics.png"#qq有新消息时的图标图片，用于识别并点击

#定义函数
def load_index_and_texts():
    #加载 FAISS 索引
    index = faiss.read_index(index_file)
    #加载文本块列表
    texts = np.load(texts_file, allow_pickle=True)
    return index, texts.tolist()

def strip_between(text: str, start: str, end: str) -> str:
    #去掉思考过程
    pattern = re.escape(start) + r".*?" + re.escape(end)
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()

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

if __name__ == "__main__":
    time.sleep(loop)
    main()
