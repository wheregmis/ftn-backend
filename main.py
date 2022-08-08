from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fastapi import FastAPI
import random
import json


# from pytorch_chat.model import NeuralNet
# from pytorch_chat.nltk_utils import bag_of_words, tokenize

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/sentiment/{text}")
def read_item(text: str):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    status = ""
    if score['neg'] > score['pos']:
        status = "Negative Sentiment"
        print(score['neg'])
        print(score['pos'])
    elif score['neg'] < score['pos']:
        status = "Positive Sentiment"
        print(score['neg'])
        print(score['pos'])
    else:
        status = "Negative Sentiment"
        print(score['neg'])
        print(score['pos'])

    if status == "Positive Sentiment":
        return {"status": status, "score": score['pos']}
    else:
        return {"status": status, "score": score['neg']}

#
# @app.get("/chat/{message}")
# def get_chat_response(message: str):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     with open('pytorch_chat/intents.json', 'r') as json_data:
#         intents = json.load(json_data)
#
#     FILE = "pytorch_chat/data.pth"
#     data = torch.load(FILE)
#
#     input_size = data["input_size"]
#     hidden_size = data["hidden_size"]
#     output_size = data["output_size"]
#     all_words = data['all_words']
#     tags = data['tags']
#     model_state = data["model_state"]
#
#     model = NeuralNet(input_size, hidden_size, output_size).to(device)
#     model.load_state_dict(model_state)
#     model.eval()
#
#     sentence = tokenize(message)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)
#
#     output = model(X)
#     _, predicted = torch.max(output, dim=1)
#
#     tag = tags[predicted.item()]
#
#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 return {"chat_response": {random.choice(intent['responses'])}}
#             else:
#                 return {"chat_response": "I do not understand..."}
