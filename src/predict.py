from skeleton.bot_model import ChatBot

if __name__ == '__main__':
    chatbot = ChatBot("../model/chatbot")
    chatbot.load_model()
    while True:
        user_input = input("请输入内容：")
        if user_input == 'q':
            break
        generated_text = chatbot.generate_text(user_input)
        print("生成的内容为：" + generated_text)
