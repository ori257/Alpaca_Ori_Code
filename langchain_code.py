from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from transformers import BitsAndBytesConfig

from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import torch

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

base_model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    quantization_config=quantization_config,
    # load_in_8bit=True,
    # device_map='auto',
)

pipe = pipeline(
    "text-generation",
    model=base_model, 
    tokenizer=tokenizer, 
    max_length=256,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: 
{instruction}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["instruction"])


llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )

question = "What is the capital of India?"
print(llm_chain.run(question))


window_memory = ConversationBufferWindowMemory(k=7)

conversation = ConversationChain(
    llm=local_llm, 
    verbose=True, 
    memory=window_memory
)

print(conversation.prompt.template)

conversation.prompt.template = '''The following is a friendly conversation between a human and an AI called Alpaca. 

Current conversation:
{history}
Human: {input}
AI:'''

conversation.predict(input="What is your name?")

