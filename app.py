from flask import Flask, render_template, request, redirect, url_for, session
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM
from haystack.reader.farm import FARMReader
from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from transformers import TFAutoModelWithLMHead, AutoTokenizer
from haystack import Finder
from fastpunct import FastPunct
from flask_mysqldb import MySQL 
import MySQLdb.cursors 
import torch
import en_coref_md
import re
import os

app = Flask(__name__)

app.secret_key = 'Hello'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Rethink!2'
app.config['MYSQL_DB'] = 'geeklogin'
  
mysql = MySQL(app) 

mode = -1
step = 0
chat_history_ids = []
dir_path = os.getcwd()
print("Please select a model you would like to use: \n 1. ALBERT Large V2 \n 2. Uncased BERT Large \n 3. XLNet")
selector = input("Enter model number here: ")
if selector == '1':
    reader = FARMReader(model_name_or_path=os.path.join(dir_path, 'SaveAlbert'), use_gpu=False, num_processes=1)
elif selector == '2':
    reader = FARMReader(model_name_or_path=os.path.join(dir_path, 'SaveBERT'), use_gpu=False, num_processes=1)
elif selector == '3':
    reader = FARMReader(model_name_or_path="./SaveXLNet", use_gpu=False, num_processes=1)
    
nlp = en_coref_md.load()
model_sum = TFAutoModelWithLMHead.from_pretrained(os.path.join(dir_path, 'summarizer'), return_dict=True)
tokenizer_sum = AutoTokenizer.from_pretrained(os.path.join(dir_path, 'summarizer'))
classifier = pipeline("zero-shot-classification", model=os.path.join(dir_path, 'zero-shot-classifier'))
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="rethinkl_test1")
retriever = DensePassageRetriever(document_store=document_store)
tokenizer_converse = AutoTokenizer.from_pretrained(os.path.join(dir_path, 'DialoGPT-large'))
model_converse = AutoModelForCausalLM.from_pretrained(os.path.join(dir_path, 'DialoGPT-large'))
finder = Finder(reader, retriever)
fastpunct = FastPunct('en')

@app.route('/') 
@app.route('/login', methods =['GET', 'POST']) 
def login(): 
    msg = '' 
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form: 
        username = request.form['username'] 
        password = request.form['password'] 
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor) 
        cursor.execute('SELECT * FROM accounts WHERE username = % s AND password = % s', (username, password, )) 
        account = cursor.fetchone() 
        if account: 
            session['loggedin'] = True
            session['id'] = account['id'] 
            session['username'] = account['username'] 
            msg = 'Logged in successfully !'
        #if username=='user@gmail.com' and password=='user1234':
            return render_template('index.html', msg = msg) 
        else: 
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg = msg)


@app.route('/logout') 
def logout(): 
    session.pop('loggedin', None) 
    session.pop('id', None) 
    session.pop('username', None) 
    return redirect(url_for('login'))

@app.route('/dummy', methods=['GET', 'POST'])
def dummy():
   message = None
   if request.method == 'POST':
        datafromjs = request.form['mydata']
        result = datafromjs +"return this"

        #resp = make_response('{"response": '+result+'}')
        #resp.headers['Content-Type'] = "application/json"
        #print(resp)
        #return resp[response]
        #time.sleep(2)
        return result
        #return render_template('index.html', message='')


@app.route('/register', methods =['GET', 'POST']) 
def register(): 
    msg = '' 
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form : 
        username = request.form['username'] 
        password = request.form['password'] 
        email = request.form['email'] 
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor) 
        cursor.execute('SELECT * FROM accounts WHERE username = % s', (username, )) 
        account = cursor.fetchone() 
        if account: 
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email): 
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username): 
            msg = 'Username must contain only characters and numbers !'
        if not username or not password or not email: 
            msg = 'Please fill out the form !'
        else: 
            cursor.execute('INSERT INTO accounts VALUES (NULL, % s, % s, % s)', (username, password, email, )) 
            mysql.connection.commit() 
            msg = 'You have successfully registered !'
    elif request.method == 'POST': 
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg) 


@app.route('/command', methods=['POST', 'GET'])
def command():
    if request.method == 'POST':
        global mode
        command_given = request.form['mydata']
        if mode == 0:
            mode = 1
            answer = predict(command_given)
            if answer == "I'm sorry, but I dont't think I understand what you mean.":
                mode = 3
                return answer + ' Would you like to see condensed version of the documents that I think are relvant to your search?'
            return answer + ' Does that answer your question?'
        elif mode == 1:
            answer = classifier(command_given, ['yes', 'no'])
            if answer['labels'][0] == 'yes':
                mode = 2
                return 'Do you have anymore questions?'
            elif answer['labels'][0] == 'no':
                mode = 3
                return 'Would you like to see a condensed version of the documents that I think are relevant to your question?'
        elif mode == 2:
            answer = classifier(command_given, ['yes', 'no'])
            if answer['labels'][0] == 'yes':
                mode = 0
                return 'Go ahead, ask me a question!'
            elif answer['labels'][0] == 'no':
                mode = -1
                return 'I hope I cleared all your queries, have a good day!'
        elif mode == 3:
            answer = classifier(command_given, ['yes', 'no'])
            if answer['labels'][0] == 'yes':
                mode = 4
                with open('prev_questions.txt', 'r') as f:
                    ques = [line.rstrip('\n') for line in f]
                context_list = retriever.retrieve(query=ques[-1], top_k=1)
                for i in range(len(context_list)):
                    context_list[i] = context_list[i].to_dict()['text'].replace('\n', ' ')
                context = ''
                for i in range(len(context_list)):
                    context += context_list[i] + ' '
                inputs = tokenizer_sum.encode("summarize: " + context, return_tensors="tf", max_length=1024)
                outputs = model_sum.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=False)
                answer = tokenizer_sum.decode(outputs[0])
                return fastpunct.punct([answer])[0] + ' Would you like to contact support for more info?'
            elif answer['labels'][0] == 'no':
                mode = 4
                return 'Would you like to contact support for more info?'
        elif mode == 4:
            answer = classifier(command_given, ['yes', 'no'])
            if answer['labels'][0] == 'yes':
                mode = 2
                return 'The website is https://0.0.0.0.0. Would you like to ask more questions?'
            elif answer['labels'][0] == 'no':
                mode = 2
                return 'Do you have anymore questions that you would like to ask me?'
        elif mode == -1:
            answer = classifier(command_given, ['question', 'converse', 'conversational question'])
            if answer['labels'][0] == 'question':
                answer = classifier(command_given, ['question', 'ask a question'])
                if answer['labels'][0] == 'question':
                    mode = 1
                    return predict(command_given) + ' Does that answer your question?'
                mode = 0
                return 'Go ahead, ask me about our products'
            return conversational(command_given)
    

@app.route('/suggestion') 
def suggestion(): 
    return render_template('suggestion.html') 


@app.route('/go') 
def go():  
    return render_template('index.html')


def predict(question_list):
    global selector
    question_list = [question_list]
    if question_list[0].split()[0].lower() == 'summarize':
        context_list = retriever.retrieve(query=question_list[0], top_k=1)
        for i in range(len(context_list)):
            context_list[i] = context_list[i].to_dict()['text'].replace('\n', ' ')
        context = ''
        for i in range(len(context_list)):
            context += context_list[i] + ' '
        inputs = tokenizer_sum.encode("summarize: " + context, return_tensors="tf", max_length=512)
        outputs = model_sum.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        answer = tokenizer_sum.decode(outputs[0])
        return fastpunct.punct([answer])[0]
    with open('prev_questions.txt', 'r') as f:
        ques = [line.rstrip('\n') for line in f]
    ques.append(question_list[0])
    if len(ques) != 1:
        final_str = ''
        for i in ques[-2:]:
            if final_str == '':
                final_str += i
                continue
            final_str += ' ' + i
        doc = nlp(final_str)
        print(doc._.has_coref)
        print(final_str)
        if doc._.has_coref:
            final_str = doc._.coref_resolved
            print(final_str)
        for index, value in enumerate(final_str[::-1]):
            if value == '?' and index != 0:
                final_index = index
                break
            elif index == len(final_str) - 1:
                final_index = len(final_str) + 1
        question = final_str[len(final_str)-final_index+1:]
    else:
        question = ques[0]
    prediction = finder.get_answers(question=question, top_k_retriever=3, top_k_reader=1)
    if prediction['answers'][0]['probability'] > 0.65:
        answer = prediction['answers'][0]['answer']
    ques[-1] = question
    with open('prev_questions.txt', 'w') as f:
        for s in ques:
            f.write(s + '\n')
    try:
        return fastpunct.punct([answer])[0]
    except:
        mode = 0
        return "I'm sorry, but I dont't think I understand what you mean."


def conversational(user_inp):
    global step
    global chat_history_ids
    new_user_input_ids = tokenizer_converse.encode(user_inp + tokenizer_converse.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    chat_history_ids = model_converse.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer_converse.eos_token_id)
    return tokenizer_converse.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

def find_best_span(start_scores, end_scores):
    assert len(start_scores) == len(end_scores)
    iscore, i = start_scores[0], 0
    best_ij, best_ij_score = (-1,-1), None
    for j, jscore in enumerate(end_scores):
        if j==0:
            continue
        if best_ij_score is None or iscore+jscore > best_ij_score:
            best_ij_score = iscore+jscore
            best_ij = (i, j)
        if start_scores[j] > iscore:
            iscore = start_scores[j]
            i = j
    return float(best_ij_score)

if __name__ == '__main__':
    app.run(debug=False)
