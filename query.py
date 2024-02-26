from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import pickle
import dotenv
import os

dotenv.load_dotenv()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

def get_llm():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1024,
        top_k=30,
    )
    llm = HuggingFacePipeline(
        pipeline=READER_LLM,
    )
    return llm

def get_prompt():
    PROMPT_TEMPLATE = """<|system|>
You are a professional movie reviewer. You have been asked to suggest a movie to a user based on a given request.
You are given 5 movies to choose from which are similar to the user request.
You have to choose the best movie matching the user reques, summarize the plot and give a review of the movie.
Don't include year of release as a seperate line. Include it in the title line.
Your review should be precise and to the point about the movie and why it is a good match for the user request.
Your review should be a little emotional and should make the user excited to watch the movie.
Your summary should be a little detailed and should include the important aspects of the movie that the user needs to know and major plot points of the movie.
Give the answer in the following format:
Title: `title` (`year`)
Genre: `genre`
Plot: `summary of the plot customised for the user request`
Review: `review of the movie according to the user request. You should include the important aspects of the movie that the user needs to know and major plot points of the movie`
<|user|>
Here are the 5 movies you can choose from:
{movies}
----------------
Now here is the movie request you have to suggest a movie for:
{request}
<|assistant|>"""
    prompt = PromptTemplate(
        input_variables=["movies", "request"],
        template=PROMPT_TEMPLATE,
    )
    return prompt

def get_llm_chain(prompt, llm):
    llm_chain = prompt | llm | StrOutputParser()
    return llm_chain

def get_retriever():
    with open("datasets/vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

def get_rag_chain(llm_chain, retriever):
    rag_chain = {"movies": retriever, "request": RunnablePassthrough()} | llm_chain
    return rag_chain

def get_rag():
    llm = get_llm()
    prompt = get_prompt()
    llm_chain = get_llm_chain(prompt, llm)
    retriever = get_retriever()
    rag_chain = get_rag_chain(llm_chain, retriever)
    return rag_chain

def query(rag_chain, request):
    return rag_chain.invoke(request)