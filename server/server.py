import sys
from threading import Thread
from typing import List, Literal, Optional

import torch
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


app = FastAPI()


class MessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None


class MessageOutput(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class Choice(BaseModel):
    message: MessageOutput


class Request(BaseModel):
    messages: List[MessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 1024
    repetition_penalty: Optional[float] = 1.0


class Response(BaseModel):
    model: str
    choices: List[Choice]


@app.post("/v1/chat/completions", response_model=Response)
async def create_chat_completion(request: Request):
    global model, tokenizer

    print(datetime.now())
    print("\033[91m--received_request\033[0m", request)

    messages = [message.model_dump() for message in request.messages]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=128000, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    print(datetime.now())
    print("\033[91m--generated_text\033[0m", result)

    message = MessageOutput(
        role="assistant",
        content=result,
    )
    choice = Choice(
        message=message,
    )
    response = Response(model=sys.argv[1].split("/")[-1].lower(), choices=[choice])
    return response


torch.cuda.empty_cache()

if __name__ == "__main__":
    MODEL_PATH = sys.argv[1]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).cuda()

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
