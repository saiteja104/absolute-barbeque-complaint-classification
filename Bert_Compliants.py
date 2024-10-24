import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Load your dataset
# df = pd.read_csv('your_dataset.csv')  # Uncomment and modify this line to load your dataset
# For demonstration, create a sample DataFrame

df = pd.read_csv("Final_AB_Complaints.csv")

# Split the data into training and testing sets
X = df['Cleaned_Review']
y = df[['Service Issue', 'Technical Issue', 'Food Quality', 'Atmosphere', 'Value for Money', 'Others', 'Hygiene']]
y = y.values  # Convert DataFrame to numpy array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input text
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

# Create a PyTorch dataset
class ComplaintDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)  # Use float for binary classification
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = ComplaintDataset(train_encodings, y_train)
test_dataset = ComplaintDataset(test_encodings, y_test)

# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=y.shape[1])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate the model
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

# Get classification report
print(classification_report(y_test, preds, target_names=[
    'Service Issue', 'Technical Issue', 'Food Quality', 
    'Atmosphere', 'Value for Money', 'Others', 'Hygiene'
]))
