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
    chatbot.add_material_from_file('../material/demo')
    chatbot.compile_model()
    chatbot.train()
    chatbot.save_model()
    # chatbot.load_model()
    resp = chatbot.predict("哈哈")
    print(resp)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
