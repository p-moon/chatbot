from skeleton.bot_model import ChatBot

MODEL_PATH = "../model/text_generator"

if __name__ == '__main__':
    chatbot = ChatBot(model_dir=MODEL_PATH, filename="../material/红楼梦.txt")
    chatbot.load_text()
    chatbot.train(epochs=100, num_threads=12)
    chatbot.save()
    chatbot = ChatBot.load(MODEL_PATH)
    generated_text = chatbot.generate_text("宝玉")
    print(generated_text)
