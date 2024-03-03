import dotenv
dotenv.load_dotenv()
import os
import pickle
import pandas as pd
import numpy as np

import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

import re

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_retriever_input(params):
    return params["messages"][-1].content

class LoggerStrOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        logger.info(f"QUERY: {text}")
        return text

PATTERN = re.compile(r'Title: (.+)\nYear: (\d+)')
reviews = pd.read_csv('datasets/reviews.csv')

def add_comments(params):
    movies = []
    for movie in params['context']:
        content = movie.page_content
        movies.append(PATTERN.findall(content)[0])
        movies[-1] = (movies[-1][0], int(movies[-1][1]))
        content = content.strip()
        comments = reviews.loc[reviews['Found Title'] == movies[-1][0]]
        movie_ratings = []
        comment_ratings = []
        comment_sents = []
        for comment in comments.iterrows():
            movie_ratings.append(float(comment['Movie Rating']))
            if comment['Comment Rating']:
                comment_ratings.append(int(comment['Comment Rating'][:-3]))
            sent = 0
            if comment['Content'] == 'positive':
                sent = 1
            elif comment['Content'] == 'negative':
                sent = -1
            comment_sents.append(sent)
        movie_rating = np.array(movie_ratings).mean()
        comment_rating = np.array(comment_ratings).mean()
        comment_sent = np.array(comment_sents).mean()
        if comment_sent < 0.25 and comment_sent > -0.25:
            comment_sent = 'Neutral'
        elif comment_sent > 0.25 and comment_sent < 0.5:
            comment_sent = 'Mostly Positive'
        elif comment_sent > 0.5:
            comment_sent = 'Very Positive'
        elif comment_sent < -0.25 and comment_sent > -0.5:
            comment_sent = 'Mostly Negative'
        else:
            comment_sent = 'Very Negative'
        content += f'\nMovie Rating: {movie_rating}\nComment Rating: {comment_rating}\nComments: {comment_sent}'
        movie.page_content = content
    print(movies)
    return params

class History:
    def __init__(self):
        self.messages = []

    def add_assistant_message(self, message):
        self.messages.append(('assistant', message))

    def add_user_message(self, message):
        self.messages.append(('user', message))

    def get_messages(self):
        result = ''
        for message in self.messages:
            result += f'<|{message[0]}|>\n'
            result += f'{message[1]}\n'
        return result.strip()

class Chat:
    def __init__(self):
        self.history = {}
        self.__init_llm()
        self.__init_chain()
        self.__init_retriever()
        self.__init_transformation_chain()
        self.retrieval_chain = (
            RunnablePassthrough.assign(
                context=self.query_transforming_retriever_chain,
            ).assign(
                answer=self.chain,
            )
        )

    def __init_llm(self):
        LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, quantization_config=bnb_config, device_map="cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        READER_LLM = pipeline(
            model=model,
            tokenizer=tokenizer,
            pad_token_id=50256,
            task="text-generation",
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1024,
            top_k=30,
        )
        self.llm = HuggingFacePipeline(
            pipeline=READER_LLM,
        )
    
    def __init_chain(self):
        prompt = PromptTemplate(
            input_variables=["context", "messages"],
            template="""<|system|>
You are a professional movie reviewer. You have been asked to suggest a movie to a user based on a given request.
You are given a list of movies to choose from which are similar to the user request.
You have to choose the best movie (with a little bias towards the popular movies) matching the user request, summarize the plot and give a review of the movie.
Choose a movie only from the list of movies.
Choose the movie closest to the user's description. Choose movies with positive reviews.
You must not include release year as a separate line. Include it in the title line.
Your review should be precise and to the point about the movie and why it is a good match for the user request.
Your review should be a little emotional and should make the user excited to watch the movie.
Your summary should be a little detailed and should include the important aspects of the movie that the user needs to know and major plot points of the movie.

You MUST give the results exactly in the following format:

Title: `title` (`year`)
Genre: `genre`
Plot: `summary of the plot according to the user request`
Review: `review of the movie according to the user request`

Here are the movies you can choose from:

{context}
-----------------
{messages}
<|assistant|>""")
        self.chain = add_comments | create_stuff_documents_chain(self.llm, prompt)
    
    def __init_retriever(self):
        with open("datasets/vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
        self.retriever = vectorstore.as_retriever(k=5)
        logger.info("retriever loaded")
    
    def __init_transformation_chain(self):
        query_transform_prompt = PromptTemplate(
            input_variables=["messages"],
            template="""<|system|>You are a helpful assistant.
You are given a conversation between the user and the assistant about requesting a movie.
Finally when the user asks, generate a search query in order to get the relevant movie to the conversation.
The search query should mostly be about the genre, theme, and the general plot of the movie and not specific about the actors or the characters in the movie unless they are very famous and known all over the world.
Only respond with the query, nothing else.
{messages}
<|user|>
give me the movie search query about the above conversation.
<|assistant|>"""
        )
        self.query_transforming_retriever_chain = query_transform_prompt | self.llm | LoggerStrOutputParser() | self.retriever

    def add_user(self, user):
        self.history[user] = History()
        logger.info(f"user {user} added")

    def has_user(self, user):
        return user in self.history

    def chat(self, user, message):
        if not self.has_user(user):
            self.add_user(user)
        logger.info(f"Q: {message}")
        self.history[user].add_user_message(message)
        response = self.retrieval_chain.invoke({"messages": self.history[user].get_messages()})
        logger.info(f"A: {response}")
        self.history[user].add_assistant_message(response['answer'])
        return response['answer']

    def count_messages(self, user):
        return len(self.history[user])