import random
import json
from flask import Flask, render_template, jsonify, request
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import wikipedia
import pandas as pd 
import webbrowser  

import mysql.connector
from datetime import datetime


   
# making boolean series for a team name
  
app = Flask(__name__)
# display
   


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data,strict=False)

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
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/chatbot', methods=["POST"])

def chatbot_msg():
  
    if request.method == "POST":
        
        user_data = request.json
        sentence = user_data['msg']
        
        
        if "ajouter :" in sentence or "ajouter:" in sentence:
              mydb = mysql.connector.connect(
              host="localhost",
              user="root",
              password="alaa1234",
              database="mydatabase"
            )
              mycursor = mydb.cursor()
            
            
              sql = "INSERT INTO medicament (med,commentaire) VALUES (%s, %s)"
              a=sentence.split(":")
              if len(a)<3:
                  return jsonify(msg="erreur de commande")
              b=a[1]
              c=a[2]
             
              val =(b,c)
              mycursor.execute(sql,val)
              mydb.commit()
              return jsonify(msg="le medicament est ajouté à la liste")
        if "afficher:" in sentence or "afficher :"in sentence:
                  
                mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password="alaa1234",
                database="mydatabase"
            )
               
            
              
                mycursor = mydb.cursor()
                mycursor.execute("SELECT * FROM medicament")
                myresult = mycursor.fetchall()
                k="la liste des medicaments est:"
                for x in myresult:
                   k=k+"<br>"+x[0]+" / "+x[1]
                
                return jsonify(msg=k) 
        if "supprimer:" in sentence or"supprimer :" in sentence:
              mydb = mysql.connector.connect(
              host="localhost",
              user="root",
              password="alaa1234",
              database="mydatabase"
            )
              mycursor = mydb.cursor()
              mycursor.execute("TRUNCATE TABLE medicament")
            
              return jsonify(msg="la liste est supprimée")
        if "je sens :" in sentence or "je sens:" in sentence:
              mydb = mysql.connector.connect(
              host="localhost",
              user="root",
              password="alaa1234",
              database="mydatabase"
            )
              mycursor = mydb.cursor()
            
            
              sql = "INSERT INTO etat (date,commentaire) VALUES (%s, %s)"
              now = datetime.now()
              formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
              a=sentence.split(":")
              b=a[1]
              val =(formatted_date,b)
              mycursor.execute(sql,val)
              mydb.commit()
              return jsonify(msg="Votre etat est transferé à votre medecin")
              
            
        if "wiki:"in sentence or "wiki :" in sentence :
            wikipedia.set_lang("fr")
            search=sentence.split(":")
            
            try:
                 m=wikipedia.search(search[1])
                 k=wikipedia.summary(m[0],sentences=4)  
                 return jsonify(msg=k)
            except wikipedia.DisambiguationError as e:
                 return jsonify(msg="Essayer de chercher avec un autre mot")
           
        if "pharma:"in sentence or "pharma :" in sentence :
            
            search=sentence.split(":")
            data = pd.read_csv("pharma.csv")
           
            select= data["site"].loc[data["ville"] == search[1].lower()]
           
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', -1)
           
           
            if len(select)==0:
                return jsonify(msg="cette ville n'est pas disponible;voici la liste des villes:<br>agadir <br>beni mellal <br>berkane<br>berrechid<br>casablanca<br>chefchaouen <br>errachidia<br>fkih ben salah <br>guelmim<br>khouribga<br>larache <br>marrakech <br>meknes <br>mohammedia <br>nador<br>ouarzazate <br>oujda<br>rabat <br>safi <br>sale <br>settat <br>tan tan <br>tanger <br>temara <br>tetouan ")
            webbrowser.open(select.to_string(index=False), new=1, autoraise=True)
            return jsonify(msg="liste des pharmacies de garde : <br>"+select.to_string(index=False))
        
        if "hopital:"in sentence or "hopital :" in sentence :
            search=sentence.split(":")
            data = pd.read_csv("hopital.csv")
            
            select= data[["hopital","num"]].loc[data["ville"] == search[1].lower()]
            print(select.to_string(header=False,index=False))
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', -1)
            if len(select)==0:
                 return jsonify(msg="cette ville n'est pas disponible;voici la liste des villes:<br>agadir <br>casablanca<br>marrakech  <br>mohammedia <br>nador <br>oujda<br>rabat <br>safi <br>kenitra <br>tanger <br>tetouan ")
                
            return jsonify(msg="liste des hopitaux:<br>"+select.to_string(header=False,index=False))
    
        sentence = tokenize(sentence)
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
                   return jsonify(msg=f"{random.choice(intent['responses'])}")
                  
        else:
            return jsonify(msg=f"J'ai pas compris...")
if __name__ == '__main__':
	app.run(host='localhost')