# import files
from flask import Flask, render_template, request
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

text_file = open("./context.txt", "r")
data = text_file.read()
text_file.close()
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    QA_input = {
    'question': f'{userText}',
    'context': data
    }
    res = nlp(QA_input)
    return str(res['answer'])

if __name__ == "__main__":
    app.run()
