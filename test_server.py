from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World! Flask is working."

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='127.0.0.1')
