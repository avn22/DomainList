import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split


# Загрузка данных
data = pd.read_csv('data_with_printer_model.csv')  # Предполагается, что в файле уже есть колонка printer_model

# Преобразование данных в тензоры (включая кодирование моделей принтеров)
printer_models = pd.get_dummies(data['printer_model'])
inputs = torch.tensor(pd.concat([data[['pages_printed']], printer_models], axis=1).values, dtype=torch.float32)
targets = torch.tensor(data['days_until_empty'].values, dtype=torch.float32).view(-1, 1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

#---------------------------------------------------------------------------------------------------
#Загрузка Данных для обучения

data = pd.read_csv('data.csv')
inputs = torch.tensor(data[['pages_printed', 'ink_level']].values, dtype=torch.float32)
targets = torch.tensor(data['days_until_empty'].values, dtype=torch.float32).view(-1, 1)

#Определении модели

class CartridgeLifeModel(nn.Module):
    def __init__(self):
        super(CartridgeLifeModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = CartridgeLifeModel()
#Критерий потерь и оптимизатор:
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# Обучение
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete")

torch.save(model.state_dict(), 'cartridge_life_model.pth')
#model = CartridgeLifeModel()
#model.load_state_dict(torch.load('cartridge_life_model.pth'))
#model.eval()
