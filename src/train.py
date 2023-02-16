# This is a sample Python script.
from skeleton.bot_model import ChatBot
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    chatbot = ChatBot("../model/chatbot")
    chatbot.load_novel(novel_path="../material/红楼梦.txt")
    chatbot.preprocess_text()
    chatbot.build_model()
    chatbot.compile_model()
    chatbot.train_model(epochs=100, num_threads=15)
    chatbot.save_model()
    chatbot.load_model()
    generated_text = chatbot.generate_text("宝玉")
    print(generated_text)
