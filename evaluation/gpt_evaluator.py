from openai import OpenAI
import json

class GPTAPIHandler:
    def __init__(self, api_key, api_base_url):
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key,
            base_url=api_base_url
        )
        self.set_context()

    def set_context(self):
        self.context = """
        To utilize a Generative Adversarial Network (GAN) for a retrieval task like determining the
         relevance of a query to a list of movies based on title and plot is unconventional,
          as GANs are typically used for generative tasks rather than retrieval. However, assuming
           you have somehow designed a GAN-based system that can score the relevance of a query to a
            given movie, here is a prompt you could use to describe the task to a human or a more traditional
             machine learning model designed for this purpose:
        Input:
        Query, Movie List
        Output:
        0 indicates that the movie is not relevant to the query.
        1 indicates that the movie is relevant to the query.
        Task:
        Your task is to assess the relevance of each movie in the provided list to 
        the given query using the GAN-based retrieval system. For each movie, 
        apply the system to compare the query with the movie's title and plot, 
        and then produce a binary relevance score (0 or 1). give output in python list format for example
        example1:
        {'query': blo blo'."}
        [{'Title': 'Movie1', 'Plot': 'zxy'}, {'Title': 'Movie2', 'Plot': 'blo blo'}]
        output:
        [0, 0]
        example2:
        {'query': blo blo'."}
        [{'Title': 'Movie1', 'Plot': 'zxy'}, {'Title': 'Movie2', 'Plot': 'blo blo'}, {'Title': 'Movie2', 'Plot': 'blo blo'}]
        output:
        [0, 1, 1]
        """

    def ask_question(self, question):
        """Asks a question based on the current context."""
        try:
            prompt = f"context:{self.context}\n\n{question}\n"
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


api_key = ''
base_url = ''
gpt_handler = GPTAPIHandler(api_key, base_url)

with open('evaluation_queries.json', 'r') as file:
    queries = json.load(file)

with open('fasttext_results.json', 'r') as file:
    fasttext_responses = json.load(file)

with open('gan_results.json', 'r') as file:
    gan_responses = json.load(file)

message = ""
counter = 1
responses = []
for query, movies in zip(queries, fasttext_responses):
    print(message)
    message += str(counter) + ".\n" + str(query) + "\n" + str(movies) + "\n\n"
    response = gpt_handler.ask_question(message)
    responses.append(response)
    counter += 1
    print(counter)
    print(response)

with open('fasttext_eval.json', 'w') as file:
    file.write(json.dumps(responses))
