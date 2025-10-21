import requests

API_URL = "http://127.0.0.1:8080/ask"
question = "What is the CE marking process?"

def ask_question(query: str):
    response = requests.get(API_URL, params={"query": query})
    if response.status_code == 200:
        print("🧠 Answer:")
        print(response.json()["answer"])
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    ask_question(question)