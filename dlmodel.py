from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'elyza/Llama-3-ELYZA-JP-8B'

model = AutoModelForCausalLM.from_pretrained(model_id)
model.save_pretrained('./assets/l3-elyza')
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained('./assets/l3-elyza')
