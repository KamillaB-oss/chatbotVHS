from flask import Flask, render_template, request, jsonify

from main import get_response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_response(input)

if __name__ == '__main__':
    app.run()
