import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # Пример модели

# Пример данных для обучения (замени на свои)
X_train = np.random.rand(100, 8)  # Пример: 100 образцов с 8 признаками
y_train = np.random.randint(0, 2, 100)  # Пример: метки (0 или 1)

# Обучаем модель (например, DecisionTreeClassifier)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Сохраняем модель в файл
joblib.dump(model, 'model.pkl')

print("Модель сохранена успешно!")
