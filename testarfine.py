from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:personal::BcPgmbYx",
    messages=[
        {"role": "system", "content": "Especialista em cannabis medicinal"},
        {"role": "user", "content": "Qual a diferença entre fitocanabinoides e endocanabinoides?"}
    ]
)
print(completion.choices[0].message)