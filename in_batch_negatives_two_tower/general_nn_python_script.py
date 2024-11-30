from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import gzip
import json
import pandas as pd
from pymongo import MongoClient
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os

# Set up GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

positive_samples = pd.read_csv("/Users/dhruvahuja/Downloads/Research/src/also_view_negatives/samples/positive_samples.csv")
positive_samples = positive_samples.sample(frac=1).reset_index(drop=True)


print("Samples read")


positive_samples = positive_samples[['user_id_encoded', 'item_id_encoded', 'brand_id_encoded', 'price', 'description', 'label']]
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
brand_encoder = LabelEncoder()
positive_samples['user_id_encoded'] = user_encoder.fit_transform(positive_samples['reviewerID'])
positive_samples['item_id_encoded'] = item_encoder.fit_transform(positive_samples['asin'])
positive_samples['brand_id_encoded'] = brand_encoder.fit_transform(positive_samples['brand'])
positive_samples = positive_samples[['user_id_encoded', 'item_id_encoded', 'brand_id_encoded', 'price', 'description', 'label']]
num_users =  positive_samples["user_id_encoded"].nunique()
num_items = positive_samples["item_id_encoded"].nunique()
num_brands =  positive_samples["brand_id_encoded"].nunique()


X = positive_samples[['user_id_encoded', 'item_id_encoded', 'brand_id_encoded', 'price', 'description']].values
y = positive_samples[['label']].values

print("Design matrix loaded")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

positive_indices = np.where(y_train == 1)[0]
X_train_positive = X_train[positive_indices]
y_train_positive = y_train[positive_indices]
positive_indices_test = np.where(y_test == 1)[0]
X_test_positive = X_test[positive_indices_test]
y_test_positive = y_test[positive_indices_test]

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

print("Data rocessed")

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

print("bag of words processing done")

class UserTower(nn.Module):
    def __init__(self, user_dim, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(user_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 64)
        
    def forward(self, x):
        x = self.embedding(x)
        return self.fc(x)


class ItemTower(nn.Module):
    def __init__(self, item_dim, brand_dim, vocab_size, embedding_dim, description_embedding_size):
        super().__init__()
        self.item_embedding = nn.Embedding(item_dim, embedding_dim)
        self.brand_embedding = nn.Embedding(brand_dim, embedding_dim)
        self.description_embedding = nn.Embedding(vocab_size, description_embedding_size)
        self.fc = nn.Linear(embedding_dim * 2 + 1 + description_embedding_size, 64)
        
    def forward(self, item, brand, price, description):
        item_emb = self.item_embedding(item)
        brand_emb = self.brand_embedding(brand)
        desc_vec = self.description_embedding(description).mean(dim=1)
        x = torch.cat([item_emb, brand_emb, desc_vec, price.unsqueeze(1)], dim=1)
        return self.fc(x)

class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, item_dim, brand_dim, vocab_size, embedding_dim, description_embedding_size):
        super().__init__()
        self.user_tower = UserTower(user_dim, embedding_dim)
        self.item_tower = ItemTower(item_dim, brand_dim, vocab_size=vocab_size, embedding_dim = embedding_dim,description_embedding_size=description_embedding_size)
        
    def forward(self, user, item, brand, price, description):
        user_emb = self.user_tower(user)
        item_emb = self.item_tower(item, brand, price, description)
        return (user_emb * item_emb).sum(dim=1)
        
def in_batch_negative_sampling(user, item, brand, price, description):
    batch_size = user.size(0)
    users = user.repeat_interleave(batch_size)
    items = item.repeat(batch_size)
    brands = brand.repeat(batch_size)
    prices = price.repeat(batch_size)
    descriptions = description.repeat(batch_size, 1)
    labels = torch.eye(batch_size).view(-1)
    positive_indices = [i * (batch_size + 1) for i in range(batch_size)]
    pos_users = users[positive_indices]
    pos_items = items[positive_indices]
    pos_brands = brands[positive_indices]
    pos_prices = prices[positive_indices]
    pos_description = descriptions[positive_indices]
    pos_labels = labels[positive_indices]
    neg_indices = []
    for i in range(batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        possible_neg = list(range(start, end))
        possible_neg.remove(start + i) 
        neg_indices.extend(random.sample(possible_neg, 1))
    users = users[neg_indices]
    items = items[neg_indices]
    brands = brands[neg_indices]
    prices = prices[neg_indices]
    descriptions = descriptions[neg_indices]
    labels = labels[neg_indices]
    users = torch.cat([pos_users, users])
    items = torch.cat([pos_items, items])
    brands = torch.cat([pos_brands, brands])
    prices = torch.cat([pos_prices, prices])
    descriptions = torch.cat([pos_description, descriptions])
    labels = torch.cat([pos_labels, labels])
    return users, items, brands, prices, descriptions, labels


def calculate_accuracy(outputs, labels):
    predictions = (torch.sigmoid(outputs) > 0.5).float()
    correct = (predictions == labels).float().sum()
    return correct / len(labels)

def train_model(model, train_loader, test_loader, optimizer, epochs):
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        all_outputs = []
        all_labels = []
        for batch in train_loader:
            user, item, brand, price, description, labels = batch
            #print the shapes
            users, items, brands, prices, description, labels = in_batch_negative_sampling(user, item, brand, price, description)
            optimizer.zero_grad()
            outputs = model(users, items, brands, prices, description)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)
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
            for batch in test_loader:
                user, item, brand, price, description, _ = batch
                users, items, brands, prices, description, labels = in_batch_negative_sampling(user, item, brand, price, description)
                outputs = model(users, items, brands, prices, description)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                total_val_accuracy += calculate_accuracy(outputs, labels)
                num_val_batches += 1
                all_val_outputs.extend(outputs.detach().cpu().numpy())
                all_val_labels.extend(labels.detach().cpu().numpy())
        
        val_loss = total_val_loss / num_val_batches
        val_accuracy = total_val_accuracy / num_val_batches
        
        model.train()
        val_auc = roc_auc_score(all_val_labels, all_val_outputs)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f"Val AUC: {val_auc:.4f}")
        print("--------------------")

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

vocab_size = len(vocab)
model = TwoTowerModel(user_dim=num_users, item_dim=num_items, brand_dim=num_brands, vocab_size=vocab_size, embedding_dim=32, description_embedding_size=32)
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_model(model, train_loader, test_loader, optimizer, epochs=50)


