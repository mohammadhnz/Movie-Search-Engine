from flask import Flask, request, render_template

from query import get_rag

rag = get_rag()

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def get_query():
    input_text = request.form['input_text']
    output = rag.query(input_text)
    return render_template('index.html', output=output)


def run_view():
    app.run()

