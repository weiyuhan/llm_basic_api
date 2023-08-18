from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Request
from transformers import AutoTokenizer
import uvicorn, json
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer

    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')

    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to(CUDA_DEVICE)
    pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
    answer = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

    torch_gc()
    return answer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", device_map="auto", trust_remote_code=True)
    
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
