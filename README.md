# **download local llm form huggingface**

## **Download TinyLlama (0.8GB)**
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" -OutFile "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```
## **Download Mistral 7B (4.1GB)**
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf" -OutFile "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

## **Phi-2 Download (1.7GB)**
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf" -OutFile "phi-2.Q4_K_M.gguf"



#📁 After Download - Move to Models Folder
## Create models directory
mkdir models

## Move downloaded model
move *.gguf models/
