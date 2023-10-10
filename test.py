from UpliftTreeRegressor import UpliftTreeRegressor
from metrics import uplift_at_k
import numpy as np
import pandas as pd


# Load data from a CSV file
data = pd.read_csv('data/data.csv')

# Extract features and target variables
X_data = data.loc[:, data.columns.str.contains('feat')]
y_data = data['target']
treatment_data = data['treatment']

X = X_data.values
y = y_data.values
treatment = treatment_data.values

# X_train, X_test, treatment_train, treatment_test, y_train, y_test = train_test_split(X, treatment, y, test_size=0.2, random_state=10)


max_depth_values = [3] #[3, 5, 7]
min_samples_leaf_values = [6000] #[1000, 2000, 3000, 6400]
min_samples_leaf_treated_values = [2500] #[300, 500, 1000]
min_samples_leaf_control_values = [2500] # [300, 500, 1000]

best_uplift = float('-inf')
best_params = {}
best_uplift_model: UpliftTreeRegressor = None

# Перебираем все комбинации параметров
for max_depth in max_depth_values:
    for min_samples_leaf in min_samples_leaf_values:
        for min_samples_leaf_treated in min_samples_leaf_treated_values:
            for min_samples_leaf_control in min_samples_leaf_control_values:
                uplift_model = UpliftTreeRegressor(
                    max_depth,
                    min_samples_leaf,
                    min_samples_leaf_treated,
                    min_samples_leaf_control
                )
                uplift_model.fit(X, treatment, y)

                # Получаем предсказания uplift и вычисляем метрику uplift@k
                uplift_predictions = uplift_model.predict(X)
                metric = uplift_at_k(y, uplift_predictions, treatment, k=0.2)

                # Если текущая метрика лучше предыдущей, обновляем лучшие параметры
                if metric > best_uplift:
                    best_uplift = metric
                    best_params = {
                        'max_depth': max_depth,
                        'min_samples_leaf': min_samples_leaf,
                        'min_samples_leaf_treated': min_samples_leaf_treated,
                        'min_samples_leaf_control': min_samples_leaf_control
                    }
                    best_uplift_model = uplift_model


# Save the tree to a text file
with open('data/uplift_tree.txt', 'w') as file:
    best_uplift_model.save_tree_to_txt(best_uplift_model.tree_, file)

print(f'Best uplift@k: {best_uplift}')
print(f'Best parameters: {best_params}')






# # Загружаем пример предсказаний
# example_preds = np.load('data/example_preds.npy')

# # Вычисляем метрику для наших предсказаний
# my_score = uplift_at_k(y, uplift_predictions, treatment)

# # Вычисляем метрику для примера предсказаний
# example_score = uplift_at_k(y, example_preds, treatment)

# # Выводим оба значения
# print(f'My score: {my_score}')
# print(f'Example score: {example_score}')

# # Вычисляем и выводим разницу в процентах
# diff = np.abs((my_score - example_score)) / example_score * 100
# print(f'Difference: {diff} %')



# max_depth_values = [3] #[3, 5, 7]
# min_samples_leaf_values = [2] #[1000, 2000, 3000, 6400]
# min_samples_leaf_treated_values = [1] #[300, 500, 1000]
# min_samples_leaf_control_values = [1] # [300, 500, 1000]

# best_uplift = float('-inf')
# best_params = {}
# best_uplift_model: UpliftTreeRegressor = None

# # Перебираем все комбинации параметров
# for max_depth in max_depth_values:
#     for min_samples_leaf in min_samples_leaf_values:
#         for min_samples_leaf_treated in min_samples_leaf_treated_values:
#             for min_samples_leaf_control in min_samples_leaf_control_values:
#                 uplift_model = UpliftTreeRegressor(
#                     max_depth,
#                     min_samples_leaf,
#                     min_samples_leaf_treated,
#                     min_samples_leaf_control
#                 )
#                 uplift_model.fit(X[:10], treatment[:10], y[:10])

#                 # Получаем предсказания uplift и вычисляем метрику uplift@k
#                 uplift_predictions = uplift_model.predict(X[:10])
#                 metric = uplift_at_k(y[:10], uplift_predictions, treatment[:10], k=0.2)

#                 # Если текущая метрика лучше предыдущей, обновляем лучшие параметры
#                 if metric > best_uplift:
#                     best_uplift = metric
#                     best_params = {
#                         'max_depth': max_depth,
#                         'min_samples_leaf': min_samples_leaf,
#                         'min_samples_leaf_treated': min_samples_leaf_treated,
#                         'min_samples_leaf_control': min_samples_leaf_control
#                     }
#                     best_uplift_model = uplift_model


# # Save the tree to a text file
# with open('data/uplift_tree.txt', 'w') as file:
#     best_uplift_model.save_tree_to_txt(best_uplift_model.tree_, file)

# print(f'Best uplift@k: {best_uplift}')
# print(f'Best parameters: {best_params}')






# # Загружаем пример предсказаний
# example_preds = np.load('data/example_preds.npy')

# # Вычисляем метрику для наших предсказаний
# my_score = uplift_at_k(y, uplift_predictions, treatment)

# # Вычисляем метрику для примера предсказаний
# example_score = uplift_at_k(y, example_preds, treatment)

# # Выводим оба значения
# print(f'My score: {my_score}')
# print(f'Example score: {example_score}')

# # Вычисляем и выводим разницу в процентах
# diff = np.abs((my_score - example_score)) / example_score * 100
# print(f'Difference: {diff} %')