import openai
import os
import json

from dotenv import load_dotenv # Para carregar a key da openAI do arquivo .env

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Caminho do arquivo dos dados para treinamento.
TRAINING_FILE_PATH = "dadoscannabis.jsonl" 

# Nome do modelo base do fine-tuning
BASE_MODEL = "gpt-3.5-turbo"

# Valida a formatação do arquivo .jsonl para fine-tuning.
def valida_jsonl(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if "messages" not in data or not isinstance(data["messages"], list):
                    print(f"Erro na linha {i+1}: 'messages' chave ausente ou não é uma lista.")
                    return False
                for message in data["messages"]:
                    if "role" not in message or "content" not in message:
                        print(f"Erro na linha {i+1}: Mensagem mal formatada (faltando 'role' ou 'content').")
                        return False
            except json.JSONDecodeError:
                print(f"Erro na linha {i+1}: Linha não é um JSON válido.")
                return False
    print(f"Arquivo '{file_path}' validado com sucesso!")
    return True

# Envia o arquivo de treinamento para a OpenAI.
def carrega_dado_treino(file_path):
        
    print(f"Fazendo upload do arquivo '{file_path}' para a OpenAI...")
    try:
        with open(file_path, "rb") as f:
            response = openai.files.create(
                file=f,
                purpose="fine-tune"
            )
        file_id = response.id
        print(f"Arquivo enviado com sucesso! ID do arquivo: {file_id}")
        return file_id
    except openai.APIError as e:
        print(f"Erro ao fazer upload do arquivo: {e}")
        return None

# Cria o fine-tuning.
def cria_finetuning(training_file_id, base_model):
    
    print(f"Criando Fine-Tuning para o modelo '{base_model}' com o arquivo '{training_file_id}'...")
    try:
        response = openai.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=base_model
        )
        job_id = response.id
        print(f"Fine-Tuning criado com sucesso! ID do Fine-Tuning: {job_id}")
        return job_id
    except openai.APIError as e:
        print(f"Erro ao criar Fine-Tuning: {e}")
        return None

# Monitora o status do fine-tuning.
def monitora_finetuning(job_id):
    
    print(f"Monitorando o status criação do Fine-Tuning (ID: {job_id})...")
    while True:
        try:
            job_status = openai.fine_tuning.jobs.retrieve(job_id)
            print(f"Status atual: {job_status.status}")

            if job_status.status == "succeeded":
                fine_tuned_model_id = job_status.fine_tuned_model
                print(f"Fine-Tuning concluído com sucesso! ID do modelo ajustado: {fine_tuned_model_id}")
                return fine_tuned_model_id
            elif job_status.status == "failed":
                print(f"O trabalho de Fine-Tuning falhou. Erro: {job_status.error}")
                return None
            elif job_status.status == "cancelled":
                print("O trabalho de Fine-Tuning foi cancelado.")
                return None

            # Espera um pouco antes de verificar novamente
            import time
            time.sleep(60) # Verifique a cada 60 segundos
        except openai.APIError as e:
            print(f"Erro ao recuperar status do trabalho: {e}")
            return None

# Usa fine-tuned para gerar uma resposta.
def usa_finetuning(model_id, prompt):
    
    print(f"\nUsando o modelo ajustado '{model_id}' para gerar uma resposta...")
    try:
        response = openai.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print("Resposta do modelo:")
        print(response.choices[0].message.content)
    except openai.APIError as e:
        print(f"Erro ao usar o modelo ajustado: {e}")

# Corpo do programa
if __name__ == "__main__":
    # 1- Validando o arquivo .json!
    if not valida_jsonl(TRAINING_FILE_PATH):
        print("Será necessário corrigir o .json para prosseguir!")
        exit()

    # 2- Enviando arquivo de treinameno para openAI!
    training_file_id = carrega_dado_treino(TRAINING_FILE_PATH)
    if not training_file_id:
        exit()

    # 3- Criando o fine-tuning!
    job_id = cria_finetuning(training_file_id, BASE_MODEL)
    if not job_id:
        exit()

    # 4- Monitora o fine-tuning até ser concluído!
    fine_tuned_model_id = monitora_finetuning(job_id)
    if not fine_tuned_model_id:
        print("Fine-tuning não foi concluído com sucesso.")
        exit()

    # 5- Usando o modelo ajustado!
    """
    test_prompt = "Quero saber sobre as características do produto X."
    usa_finetuning(fine_tuned_model_id, test_prompt)
    """

    print("\nProcesso de fine-tuning finalizado!")