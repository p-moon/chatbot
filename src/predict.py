from skeleton.bot_model import ChatBot

MODEL_PATH = "../model/text_generator"

if __name__ == '__main__':
    chatbot = ChatBot.load(MODEL_PATH)
    while True:
        user_input = input("请输入内容：")
        generated_text = chatbot.generate_text(user_input)
        print("生成的内容为：" + generated_text)
