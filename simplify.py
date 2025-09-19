from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained("./medicalsimplifiermodel").to(device)
tokenizer = T5Tokenizer.from_pretrained("./medicalsimplifiermodel")

def simplify_text(text):
    input_text = "simplify in English: " + text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=6,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.2,
        repetition_penalty=1.5,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("Enter medical text or type 'quit' to exit:")
    while True:
        text = input("> ")
        if text.lower() == "quit":
            break
        simplified = simplify_text(text)
        print(f"Simplified text: {simplified}\n")
