import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# ========== Task 1: Load and Preprocess the Dataset ==========
file_path = "C:\\Users\\Chris\\Downloads\\league_of_legends_data_large.csv"
data = pd.read_csv(file_path)

print("Columns in dataset:", data.columns.tolist())

# Use 'win' as target column
target_column = 'win'

X = data.drop(columns=[target_column])
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# ========== Task 2: Implement Logistic Regression Model ==========
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


input_dim = X_train_tensor.shape[1]
model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ========== Task 3: Train the Model ==========
epochs = 100
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            preds = model(X_test_tensor)
            preds_class = (preds > 0.5).float()
            accuracy = (preds_class == y_test_tensor).float().mean()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

# ========== Task 4: Add L2 Regularization ==========
model_l2 = LogisticRegressionModel(input_dim)
optimizer_l2 = optim.SGD(model_l2.parameters(), lr=0.01, weight_decay=0.1)  # L2 regularization

for epoch in range(epochs):
    model_l2.train()
    outputs = model_l2(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer_l2.zero_grad()
    loss.backward()
    optimizer_l2.step()

with torch.no_grad():
    preds = model_l2(X_test_tensor)
    preds_class = (preds > 0.5).float()
    accuracy_l2 = (preds_class == y_test_tensor).float().mean()
    print(f"L2 Regularized Test Accuracy: {accuracy_l2:.4f}")

# ========== Task 5: Visualize Performance ==========
with torch.no_grad():
    preds = model_l2(X_test_tensor)
    preds_class = (preds > 0.5).int()

cm = confusion_matrix(y_test_tensor, preds_class)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test_tensor, preds_class))

fpr, tpr, _ = roc_curve(y_test_tensor.numpy(), preds.numpy())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ========== Task 6: Save and Load Model ==========
torch.save(model_l2.state_dict(), "logistic_model.pth")

loaded_model = LogisticRegressionModel(input_dim)
loaded_model.load_state_dict(torch.load("logistic_model.pth"))
loaded_model.eval()

with torch.no_grad():
    preds_loaded = loaded_model(X_test_tensor)
    accuracy_loaded = ((preds_loaded > 0.5).float() == y_test_tensor).float().mean()
    print(f"Loaded Model Accuracy: {accuracy_loaded:.4f}")

# ========== Task 7: Hyperparameter Tuning ==========
learning_rates = [0.001, 0.01, 0.05, 0.1]
best_lr = 0
best_acc = 0

for lr in learning_rates:
    model_tune = LogisticRegressionModel(input_dim)
    optimizer_tune = optim.SGD(model_tune.parameters(), lr=lr)

    for epoch in range(50):
        model_tune.train()
        outputs = model_tune(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer_tune.zero_grad()
        loss.backward()
        optimizer_tune.step()

    with torch.no_grad():
        preds = model_tune(X_test_tensor)
        acc = ((preds > 0.5).float() == y_test_tensor).float().mean().item()
        print(f"Learning Rate {lr}: Test Accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_lr = lr

print(f"Best Learning Rate: {best_lr}, Accuracy: {best_acc:.4f}")

# ========== Task 8: Feature Importance ==========
weights = model_l2.linear.weight.detach().numpy().flatten()
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': weights
}).sort_values(by='Importance', key=abs, ascending=False)

print("\nTop Important Features:")
print(importance_df.head(10))

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Weight (Importance)")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
