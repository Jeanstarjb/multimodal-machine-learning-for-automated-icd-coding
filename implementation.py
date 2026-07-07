import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

# Text model: A simple Text-CNN
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x1 = self.pool(torch.relu(self.conv1(x))).squeeze(-1)
        x2 = self.pool(torch.relu(self.conv2(x))).squeeze(-1)
        x = torch.cat([x1, x2], dim=1)
        return self.fc(x)

# Tabular model: A simple feedforward neural network
class TabularNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TabularNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Ensemble model
class MultimodalEnsemble(nn.Module):
    def __init__(self, text_model, tabular_model, num_classes):
        super(MultimodalEnsemble, self).__init__()
        self.text_model = text_model
        self.tabular_model = tabular_model
        self.fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, text_input, tabular_input):
        text_output = self.text_model(text_input)
        tabular_output = self.tabular_model(tabular_input)
        combined = torch.cat([text_output, tabular_output], dim=1)
        return self.fc(combined)

if __name__ == '__main__':
    # Dummy data
    vocab_size = 5000
    embed_dim = 50
    num_classes = 10
    text_input = torch.randint(0, vocab_size, (32, 100))  # Batch of 32, sequence length 100
    tabular_input = torch.rand(32, 20)  # Batch of 32, 20 features
    labels = torch.randint(0, 2, (32, num_classes)).float()  # Multilabel binary classification

    # Models
    text_model = TextCNN(vocab_size, embed_dim, num_classes)
    tabular_model = TabularNN(20, num_classes)
    ensemble_model = MultimodalEnsemble(text_model, tabular_model, num_classes)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(ensemble_model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = ensemble_model(text_input, tabular_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Evaluation
    with torch.no_grad():
        outputs = torch.sigmoid(ensemble_model(text_input, tabular_input))
        preds = (outputs > 0.5).float()
        labels_np = labels.numpy()
        preds_np = preds.numpy()
        micro_f1 = f1_score(labels_np, preds_np, average='micro')
        micro_auc = roc_auc_score(labels_np, outputs.numpy(), average='micro')
        print(f"Micro-F1: {micro_f1}, Micro-AUC: {micro_auc}")