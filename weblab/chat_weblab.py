import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import Fore, Back, Style, init

init(autoreset=True)

tokenizer = AutoTokenizer.from_pretrained("matsuo-lab/weblab-10b-instruction-sft")
model = AutoModelForCausalLM.from_pretrained("matsuo-lab/weblab-10b-instruction-sft", torch_dtype=torch.float16, device_map="auto")

if torch.cuda.is_available():
    model = model.to("cuda")

while True:
    input_text = input('> ')
    token_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            # pad_token_id=tokenizer.pad_token_id,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0])
    print(Fore.YELLOW + 'weblab: ' + output)

