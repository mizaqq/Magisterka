import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertForTokenClassification
import pytorch_lightning as pl

# Load and prepare data
df = pd.read_csv('/home/miza/Magisterka/src/data/gpt_generated_data.csv')
label_encoder = LabelEncoder()
df['Category_encoded'] = label_encoder.fit_transform(df['Category'])

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


def encode_text(text_list, tokenizer, max_length=64):
    return tokenizer(text_list, padding='max_length', truncation=False, max_length=max_length, return_tensors='pt')


inputs = encode_text(df['OCR_product'].tolist(), tokenizer)
labels_category = torch.tensor(df['Category_encoded'].values)
labels_cost = torch.tensor(df['Correct_cost'].values, dtype=torch.float32)

dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels_category, labels_cost)

# Explicit train/val/test split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)


# PyTorch Lightning Module
class BertMultiTaskModel(pl.LightningModule):
    def __init__(self, num_categories, learning_rate=1e-1):
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            'bert-base-multilingual-cased', num_labels=num_categories
        )
        self.dropout = nn.Dropout(0.3)
        self.cost_regressor = nn.Linear(self.bert.config.hidden_size, 1)
        self.criterion_category = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion_cost = nn.MSELoss()
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        category_logits = outputs.logits[:, 0, :]

        last_hidden_state = outputs.hidden_states[-1][:, 0, :]  # Shape: (batch_size, hidden_dim)

        # Pass through regression layer
        cost_prediction = self.cost_regressor(last_hidden_state).squeeze()

        return category_logits, cost_prediction

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels_cat, labels_cost = batch
        logits_cat, pred_cost = self(input_ids, attention_mask)
        loss_cat = self.criterion_category(logits_cat, labels_cat)
        loss_cost = self.criterion_cost(pred_cost, labels_cost)
        loss = loss_cat + loss_cost
        self.train_losses.append(loss.item())
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels_cat, labels_cost = batch
        logits_cat, pred_cost = self(input_ids, attention_mask)
        loss_cat = self.criterion_category(logits_cat, labels_cat)
        loss_cost = self.criterion_cost(pred_cost, labels_cost)
        loss = loss_cat + loss_cost
        self.val_losses.append(loss.item())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# Initialize and train model
model = BertMultiTaskModel(num_categories=df['Category_encoded'].nunique())
trainer = pl.Trainer(max_epochs=4, accelerator='gpu', devices=1, num_sanity_val_steps=0)
trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint("bert_multitask_model.ckpt")
import matplotlib.pyplot as plt


plt.figure(figsize=(8, 5))
plt.plot(model.train_losses, label="Train Loss", marker="o")
plt.plot(model.val_losses, label="Validation Loss", marker="o", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

# Save the figure instead of showing it
plt.savefig("/home/miza/Magisterka/src/analysis/train_val_loss_curve.png")

print("Loss curve saved as train_val_loss_curve.png")

model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels_cat, _ = batch
        logits_cat, _ = model(input_ids, attention_mask)
        preds = torch.argmax(logits_cat, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels_cat.cpu().numpy())

print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))

# Inference
model.eval()
sample_text = "Ryż bialu 1kg 0 x4.29 34.32"
encoded_input = encode_text([sample_text], tokenizer)
input_ids = encoded_input['input_ids'].to(model.device)
attention_mask = encoded_input['attention_mask'].to(model.device)

with torch.no_grad():
    logits_cat, pred_cost = model(input_ids, attention_mask)
    predicted_category_id = torch.argmax(logits_cat, dim=1).item()
    predicted_category = label_encoder.inverse_transform([predicted_category_id])[0]
    predicted_cost = pred_cost.item()

print(f"Predicted Category: {predicted_category}, Predicted Cost: {predicted_cost:.2f}")
