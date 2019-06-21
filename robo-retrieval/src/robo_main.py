'''
Created on Jan 14, 2019

@author: LupitaRios

First chatbot Retrieval Based Model
'''

# Libraries 
import nltk
import numpy as np
import random
import string # to process standard python strings

# corpus
# reading the data
f= open('chatbot.txt', 'r', errors = 'ignore')

raw = f.read()

raw = raw.lower() # coverts to lowercase

#nltk.download('punkt') # first-time
sent_tokens = nltk.sent_tokenize(raw) # converts to list of sentences
word_tokens = nltk.word_tokenize(raw) # convert to list of words

#print("sent_tokens = " + str(sent_tokens[0:2]))
#print("word_tokens = " + str(word_tokens[0:2]))
sent_tokens[:2]
word_tokens[:5]

# pre-processing the raw text
# LelTokens will take as input the tokens and return normalized tokens
lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punt_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormilize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punt_dict)))


# Keyword matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)

GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
# generating response
# the concep of document similarity will be used.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# function response to search the user's utterance for one or more known keywords, and return 
# one of several possible responses.

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    
    TfidrVec = TfidfVectorizer(tokenizer = LemNormilize, stop_words = 'english')
    tfidf = TfidrVec.fit_transform(sent_tokens)
    
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    
    req_tfidf = flat[-2]
    
    if(req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response
    
# finally, we'll feed what the boot say while starting and ending a conversation dependong upon user's input
flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")

while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("ROBO: You are welcome...")
        else:
            if(greeting(user_response) != None):
                print("ROBO: " + greeting(user_response))
            else:
                print("ROBO: ", end = "")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye! take care...")
    