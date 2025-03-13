from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import os

# Define where to save the model
output_dir = "./quantum_math_gpt2_model"
os.makedirs(output_dir, exist_ok=True)

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure there's a padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

# Sample dataset of quantum mechanics and probability-related problems
quantum_math_data = [
    "Quantum mechanics explains the behavior of subatomic particles.",
    "Superposition means a particle exists in multiple states at once.",
    "The Schrödinger equation governs quantum state evolution: iℏ∂Ψ/∂t = ĤΨ.",
    "The uncertainty principle: ΔxΔp ≥ ℏ/2.",
    "Q: Calculate the de Broglie wavelength of an electron with p = 1.5×10⁻²⁴ kg·m/s.\nA: λ = h/p = 4.42×10⁻¹⁰ m.",
    "Q: Find the energy of a photon with λ = 500 nm.\nA: E = hc/λ = 3.98×10⁻¹⁹ J.",
    "Q: If P(A) = 0.3 and P(B) = 0.4 (independent), find P(A and B).\nA: 0.3 × 0.4 = 0.12.",
]

# Custom dataset class for tokenized quantum mechanics data


class QuantumMathDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                   max_length=max_length, return_tensors="pt")

    def __getitem__(self, idx):
        # Extract data for batch training
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()  # Labels for training
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


# Prepare dataset and DataLoader
dataset = QuantumMathDataset(quantum_math_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Move model to available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# Optimizer setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop setup
num_epochs = 5
print(f"Starting training on {device}...")

for epoch in range(num_epochs):
    total_loss = 0

    for batch in dataloader:
        # Move batch to GPU/CPU
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()  # Reset gradients
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()  # Backpropagation step
        optimizer.step()  # Update model weights

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

# Save fine-tuned model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Chat function for interactive conversation


def chat():
    print("\nQuantum Physics Chatbot (type 'exit' to quit)")

    # Load the fine-tuned model (or fallback to base GPT-2)
    try:
        model = GPT2LMHeadModel.from_pretrained(output_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
        print("Loaded fine-tuned model.")
    except:
        print("Fine-tuned model not found. Using base GPT-2.")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model.to(device)
    model.eval()

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        input_ids = tokenizer.encode(
            user_input, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=200,
                num_return_sequences=1,
                top_p=0.92,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"AI: {response[len(user_input):].strip()}")


if __name__ == "__main__":
    chat()
