import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import folium

# Load and preprocess data
data = pd.read_csv("San_Francisco.csv")
data['Datetime'] = pd.to_datetime(data['Datetime']).astype(int) / 10**9
scaler = MinMaxScaler()
data[['Datetime', 'Latitude', 'Longitude']] = scaler.fit_transform(data[['Datetime', 'Latitude', 'Longitude']])
encoder = LabelEncoder()
data['Category'] = encoder.fit_transform(data['Category'])

# Prepare dataset and dataloader
class CrimeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx, [0, 2, 3]].values, dtype=torch.float)
        label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long)
        return features, label

dataset = CrimeDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Split dataset into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare dataset and dataloader for train and test sets
train_dataset = CrimeDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CrimeDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM model
class CrimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CrimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



# Model parameters
input_size = 3
hidden_size = 64
num_layers = 3 
num_classes = len(data['Category'].unique())

# Initialize model, loss function, and optimizer
model = CrimeLSTM(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the model
num_epochs = 5

for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(dataloader):
        features = features.unsqueeze(1)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))

# Test the model
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for features, labels in test_dataloader:
        features = features.unsqueeze(1)
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

# Calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(all_labels, all_predictions)
conf_matrix = confusion_matrix(all_labels, all_predictions)
class_report = classification_report(all_labels, all_predictions, target_names=encoder.classes_, zero_division=1)


print("Accuracy: {:.2f}".format(accuracy))
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Create a function to inverse_transform the data
def inverse_transform_data(data, scaler, encoder):
    inv_data = data.copy()
    inv_data[['Datetime', 'Latitude', 'Longitude']] = scaler.inverse_transform(data[['Datetime', 'Latitude', 'Longitude']])
    inv_data['Category'] = encoder.inverse_transform(data['Category'])
    return inv_data

# Inverse_transform test_data
test_data_inv = inverse_transform_data(test_data, scaler, encoder)

# Save the trained model
model_path = "lstm_model.pt"
torch.save(model.state_dict(), model_path)


from folium.plugins import MarkerCluster

# ...

# Create a map centered at the mean latitude and longitude
m = folium.Map(location=[test_data_inv['Latitude'].mean(), test_data_inv['Longitude'].mean()], zoom_start=12)

# Create a MarkerCluster
marker_cluster = MarkerCluster().add_to(m)

counter = 0
for idx, row in test_data_inv.iterrows():
    if row['Category'] == encoder.inverse_transform([all_predictions[counter]])[0]:
        marker_color = 'green'
    else:
        marker_color = 'red'
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Category'], icon=folium.Icon(color=marker_color)).add_to(marker_cluster)
    counter += 1


# Save the map as an HTML file
m.save("crime_map.html")
