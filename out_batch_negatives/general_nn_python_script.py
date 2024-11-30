from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pandas as pd
import gzip
import json
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

combined_samples = pd.read_csv("combined_samples.csv")

combined_samples = combined_samples.sample(frac=1).reset_index(drop=True)

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
brand_encoder = LabelEncoder()
combined_samples['user_id_encoded'] = user_encoder.fit_transform(combined_samples['reviewerID'])
combined_samples['item_id_encoded'] = item_encoder.fit_transform(combined_samples['asin'])
combined_samples['brand_id_encoded'] = brand_encoder.fit_transform(combined_samples['brand'])
combined_samples = combined_samples[['user_id_encoded', 'item_id_encoded', 'brand_id_encoded', 'price', 'description', 'label']]
num_users =  combined_samples["user_id_encoded"].nunique()
num_items = combined_samples["item_id_encoded"].nunique()
num_brands =  combined_samples["brand_id_encoded"].nunique()

X = combined_samples[['user_id_encoded', 'item_id_encoded', 'brand_id_encoded', 'price', 'description']].values
y = combined_samples['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

positive_indices = np.where(y_train == 1)[0]
X_train_positive = X_train[positive_indices]
y_train_positive = y_train[positive_indices]
positive_indices_test = np.where(y_test == 1)[0]
X_test_positive = X_test[positive_indices_test]
y_test_positive = y_test[positive_indices_test]

for i in range(10):
    X_train = np.concatenate((X_train, X_train_positive))
    y_train = np.concatenate((y_train, y_train_positive))
    X_test = np.concatenate((X_test, X_test_positive))
    y_test = np.concatenate((y_test, y_test_positive))

permutation = np.random.permutation(len(X_train))
X_train_shuffled = X_train[permutation]
y_train_shuffled = y_train[permutation]


X_train_user = X_train[:, 0].astype(np.int32)
X_train_item = X_train[:,1].astype(np.int32)
X_train_brand = X_train[:,2].astype(np.int32)
X_train_price = X_train[:,3].astype(np.float32)
X_train_description = X_train[:, 4].astype(np.str_)

X_test_user = X_test[:, 0].astype(np.int32)
X_test_item = X_test[:, 1].astype(np.int32)
X_test_brand = X_test[:, 2].astype(np.int32)
X_test_price = X_test[:, 3].astype(np.float32)
X_test_description = X_test[:, 4].astype(np.str_)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)
vocab = build_vocab_from_iterator(yield_tokens(X_train_description), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

X_train_description_processed = [text_pipeline(text) for text in X_train_description]
X_test_description_processed = [text_pipeline(text) for text in X_test_description]
max_length = max(len(desc) for desc in X_train_description_processed + X_test_description_processed)

def pad_sequence(seq):
    return seq + [vocab["<unk>"]] * (max_length - len(seq))

X_train_description_padded = torch.tensor([pad_sequence(desc) for desc in X_train_description_processed], dtype=torch.long)
X_test_description_padded = torch.tensor([pad_sequence(desc) for desc in X_test_description_processed], dtype=torch.long)

X_train_user = torch.tensor(X_train_user, dtype=torch.int32).to(device)
X_train_item = torch.tensor(X_train_item, dtype=torch.int32).to(device)
X_train_brand = torch.tensor(X_train_brand, dtype=torch.int32).to(device)
X_train_price = torch.tensor(X_train_price, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

X_test_user = torch.tensor(X_test_user, dtype=torch.int32).to(device)
X_test_item = torch.tensor(X_test_item, dtype=torch.int32).to(device)
X_test_brand = torch.tensor(X_test_brand, dtype=torch.int32).to(device)
X_test_price = torch.tensor(X_test_price, dtype=torch.float32).to(device)

y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


train_dataset = TensorDataset(X_train_user, X_train_item, X_train_brand, X_train_price, X_train_description_padded, y_train)
test_dataset = TensorDataset(X_test_user, X_test_item, X_test_brand, X_test_price, X_test_description_padded, y_test)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, num_brands, vocab_size, embedding_size, description_embedding_size):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.brand_embedding = nn.Embedding(num_brands, embedding_size)
        self.description_embedding = nn.Embedding(vocab_size, description_embedding_size)
        self.fc1 = nn.Linear(embedding_size * 3 + 1 + description_embedding_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user, item, brand, price, description):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        brand_vec = self.brand_embedding(brand)
        desc_vec = self.description_embedding(description).mean(dim=1)  
        x = torch.cat([user_vec, item_vec, brand_vec, price.unsqueeze(1), desc_vec], dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = torch.sigmoid(self.fc3(x))
        return x



vocab_size = len(vocab)
model = RecommendationModel(num_users=num_users, num_items=num_items, num_brands=num_brands, vocab_size=vocab_size, embedding_size=32, description_embedding_size=100).to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)


def calculate_accuracy(outputs, labels):
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    correct = (predictions == labels).float().sum()
    return correct / len(labels)

for epoch in range(50):
    model.train()
    all_outputs = []
    all_labels = []
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    for users, items, brands, prices, descriptions, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(users, items, brands, prices, descriptions)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_accuracy += calculate_accuracy(outputs.squeeze(), labels)
        num_batches += 1
        all_outputs.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    train_auc = roc_auc_score(all_labels, all_outputs)
    train_loss = total_loss / num_batches
    train_accuracy = total_accuracy / num_batches
    
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0
    num_val_batches = 0
    all_val_outputs = []
    all_val_labels = []
    
    with torch.no_grad():
        for users, items, brands, prices, descriptions, labels in test_loader:
            outputs = model(users, items, brands, prices, descriptions)
            val_loss = criterion(outputs.squeeze(), labels)
            total_val_loss += val_loss.item()
            total_val_accuracy += calculate_accuracy(outputs.squeeze(), labels)
            num_val_batches += 1
            all_val_outputs.extend(outputs.detach().cpu().numpy())
            all_val_labels.extend(labels.detach().cpu().numpy())
        
    val_loss = total_val_loss / num_val_batches
    val_accuracy = total_val_accuracy / num_val_batches
    model.train()
    val_auc = roc_auc_score(all_val_labels, all_val_outputs)
    print(f"Epoch {epoch+1}/{50}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Val AUC: {val_auc:.4f}")
    print("--------------------")
