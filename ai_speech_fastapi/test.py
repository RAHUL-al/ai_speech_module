# # import os
# # import requests
# # from dotenv import load_dotenv

# # load_dotenv()

# # def speech_to_text() -> str:
# #         API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
# #         headers = {
# #             "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
# #             "Content-Type": "audio/wav"
# #         }

# #         if not os.path.exists("harvard.wav"):
# #             print(f"[Error] Audio file not found: ")
# #             return ""

# #         try:
# #             with open("harvard.wav", "rb") as f:
# #                 data = f.read()

# #             response = requests.post(API_URL, headers=headers, data=data)
# #             response.raise_for_status()

# #             result = response.json()
# #             text = result.get("text", "").strip()
# #             print(f"Transcribed [{os.path.basename("harvard.wav")}]: {text}")
# #             return text

# #         except Exception as e:
# #             print(f"[Error] Failed to transcribe harvard.wav: {e}")
# #             return ""
        
# # speech_to_text()


# from langchain_openai import ChatOpenAI
# from langchain.schema import SystemMessage, HumanMessage
# import os
# import threading
# from dotenv import load_dotenv

# load_dotenv()

# class TinyLlamaChat:
#     def __init__(self):
#         self.llm = ChatOpenAI(
#             model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#             openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#             openai_api_base="https://openrouter.ai/api/v1",
#             temperature=0.7
#         )


#     def topic_data_model(self, username: str, prompt: str) -> str:
#         try:
#             messages = [
#                 SystemMessage(content="You are a helpful assistant."),
#                 HumanMessage(content=prompt)
#             ]
#             response = self.llm.invoke(messages)
#             text_output = response.content
#             return text_output
#         except Exception as e:
#             print(f"[LangChain TinyLlama API Error] {e}")
#             return "TinyLlama model failed to generate a response."


# test = TinyLlamaChat()
# data = test.topic_data_model("Prince","tell me the capial of india?")


# import torch
# from transformers import pipeline

# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# messages = [
#     {
#         "role": "system",
#         "content": "You are a for help in text generation.",
#     },
#     {"role": "user", "content": "tell me the capial of india"},
# ]
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# print(outputs[0]["generated_text"])


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)

print("content:", content)


# def topic_data_model_for_Qwen(prompt: str) -> str:
#         try:
#             api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen1.5-7B-Chat"
#             headers = {
#                 "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}",
#                 "Content-Type": "application/json"
#             }

#             payload = {
#                 "inputs": [
#                     {"role": "user", "content": prompt}
#                 ],
#                 "parameters": {
#                     "do_sample": True,
#                     "max_new_tokens": 512,
#                     "return_full_text": False
#                 }
#             }

#             response = requests.post(api_url, headers=headers, json=payload)
#             response.raise_for_status()
#             result = response.json()

#             if isinstance(result, dict) and "error" in result:
#                 print(f"[HuggingFace Qwen Error] {result['error']}")
#                 return "Qwen API returned an error."

#             if isinstance(result, list):
#                 output_text = result[0]["generated_text"]
#             else:
#                 output_text = str(result)

#             cleaned_text = re.sub(r'\*.*?\*', '', output_text)
#             cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')
#             cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

#             return cleaned_text

#         except Exception as e:
#             print(f"[Qwen API Error] {e}")
#             return "Qwen model failed to generate a response."
        
        
# result = topic_data_model_for_Qwen("tell me the capital of india")
# print(result)
