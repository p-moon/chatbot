# This is a sample Python script.
from skeleton.bot_model import ChatBot
import nltk
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # nltk.download('punkt')
    # nltk.download('stopwords')
    chatbot = ChatBot("../model/chatbot")
    chatbot.load_novel(novel_path="../material/红楼梦")
    chatbot.preprocess_text()
    chatbot.build_model()
    chatbot.compile_model()
    chatbot.train_model(epochs=100)
    chatbot.save_model()
    generated_text = chatbot.generate_text("The man")
    print(generated_text)
