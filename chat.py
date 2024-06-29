import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from spellchecker import SpellChecker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

spell = SpellChecker()

bot_name = "Jahanzaib"

def correct_text(text):
    words = text.split()
    corrected_words = [spell.correction(word) or word for word in words]
    return ' '.join(corrected_words)

def get_response(msg):
    corrected_msg = correct_text(msg)
    sentence = tokenize(corrected_msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == 'name':
                    return random.choice(intent['responses']).format(bot_name=bot_name)
                elif tag == 'qualification':
                    response = random.choice(intent['responses'])
                    response += f"I have completed a BE in Software Engineering and a Web Designing and Development Course."
                    return response
                elif tag == 'expertise':
                    response = random.choice(intent['responses'])
                    response += " My expertise lies in developing WordPress websites, crafting custom-coded solutions, and creating personalized web experiences. I am proficient in HTML, CSS, Bootstrap, JavaScript, PHP, MySQL, React JS, MongoDB, C#, and C."
                    return response
                else:
                    return random.choice(intent['responses'])

    return "I do not understand..."

if __name__ == "__main__":
    print(f"{bot_name}: Hello! I'm {bot_name}, your chatbot assistant. Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(f"{bot_name}: {resp}")
