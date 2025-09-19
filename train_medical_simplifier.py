import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Load the CSV with correct column names
df = pd.read_csv('medical_dataset.csv')

# Use the updated columns
dataset = Dataset.from_pandas(df)

# Split into train and test sets
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test['train']
eval_dataset = train_test['test']

# Use t5-base for better capacity
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocess function with reduced max_length for memory
def preprocess_function(examples):
    inputs = ["simplify in English: " + ex for ex in examples["original_text"]]
    model_inputs = tokenizer(inputs, max_length=96, padding='max_length', truncation=True)  # reduced length
    labels = tokenizer(examples["simplified_text"], max_length=96, padding='max_length', truncation=True)

    labels_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in l]
        for l in labels["input_ids"]
    ]
    model_inputs["labels"] = labels_ids
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # More epochs for better training
    per_device_train_batch_size=1,  # Reduced batch size for memory saving
    per_device_eval_batch_size=1,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    # Removed evaluation_strategy if version doesn't support it
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./medicalsimplifiermodel")
tokenizer.save_pretrained("./medicalsimplifiermodel")

print("Training complete! Model saved to ./medicalsimplifiermodel")
