from flask import Flask, request, render_template, redirect, url_for
from information_retrieval import *
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///files_database.db'
db = SQLAlchemy(app)

home_dir = os.getcwd()
UPLOAD_FOLDER = os.path.join(home_dir, "uploaded_files")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class FileStructure(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    data = db.Column(db.LargeBinary)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['inputFile']
    newFile = FileStructure(name=file.filename, data=file.read())
    file.stream.seek(0)
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        with open(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), 'r', encoding='utf-8') as f:
            text = f.read()
            add_doc_to_original(text)
            processed_txt = preprocess_text(text)
            add_doc_to_processed(processed_txt)
            db.session.add(newFile)
            db.session.commit()
		#return 'File saved successfully'
        return redirect(url_for('index'))


@app.route('/retrieve', methods=['POST'])
def retrieve():
    query = request.form.get('userquery')
    processed_query = preprocess_text(query)
    output = get_similar_documents(processed_query)
    return render_template('results.html', results=output)

	
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)