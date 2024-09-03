import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LassoCV, RidgeCV
import numpy as np


lasso = Lasso(random_state= 17)
ridge = Ridge(random_state= 17)
lasso_cv = LassoCV(random_state=17)
ridge_cv = RidgeCV()

df = pd.read_csv('/Users/annaisaeva/PycharmProjects/social_wasters/Time-Wasters on Social Media.csv')

X = df.drop(columns='Satisfaction')
y = df['Satisfaction']

numerical_features = X.select_dtypes(include=['int64']).columns
categorical_features = X.select_dtypes(include=['object', 'bool']).columns

# трансформер, который заполняет пропущенные числовые значения средним
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

# трансформер, который заполняет пропущенные категориальные значения модой
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# объединяю трансформеры в колонный трансформер
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features),
                                               ('cat', categorical_transformer, categorical_features)])


def train_validate_report(model, X_train, y_train, X_test, y_test, feature_names):
    # Создаем полный пайплайн
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    # Обучаем модель
    pipeline.fit(X_train, y_train)

    # Предсказания
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Оценка модели
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # рассчет коэффициента детерминации для оценки качества модели
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"MSE для модели {model.__class__.__name__}:")
    print(f"  - На обучающей выборке: {np.sqrt(mse_train)}")
    print(f"  - На тестовой выборке: {np.sqrt(mse_test)}")
    print(f"R² для модели {model.__class__.__name__}:")
    print(f"  - На обучающей выборке: {r2_train:.2f}")
    print(f"  - На тестовой выборке: {r2_test:.2f}")

    # print("Model coefficirnts:")
    # print(pd.DataFrame(model.coef_, feature_names, columns =['coef']).sort_values(by='coef',                                                                 ascending = False))
    # y_pred = model.predict(X_test)
    # for i, (real, pred) in enumerate(zip(y_test, y_pred)):
    #     error = real - pred
    #     print(f'{i:<10} {real:<20} {pred:<25} {error:<10}')



from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

categorical_data = []
for column_name in df.columns:
    if not (pd.api.types.is_integer_dtype(df[column_name]) or pd.api.types.is_integer(df[column_name])):
        categorical_data.append(column_name)

for column_name in df.columns:
    if column_name in categorical_data:
        df[column_name] = labelencoder.fit_transform((df[column_name]))
        print(labelencoder.classes_)

'''специально обхявляю еще раз после того, как закодировала половину значений'''
X = df.drop(columns='Satisfaction')
y = df['Satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Или можно использовать Ridge, пока не поняла, как выбирать оптимальный вариант
# model = Ridge(alpha=1.0, random_state=42)


models = [Lasso(random_state=17), Ridge(random_state=17), LassoCV(random_state=17), RidgeCV()]
for model in models:
    train_validate_report(model, X_train, y_train, X_test, y_test,
                          feature_names=numerical_features.tolist() + categorical_features.tolist())
'''model = Lasso(random_state=17)
clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
param_grid = {'model__alpha': [0.01, 0.1, 1, 10]}

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Test Mean Squared Error: {mse}')

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()
print(f'Cross-Validation Mean Squared Error: {cv_mse}')

for i, (real, pred) in enumerate(zip(y_test, y_pred)):
    error = real - pred
    print(f'{i:<10} {real:<20} {pred:<25} {error:<10}')

from sklearn.model_selection import train_test_split, cross_val_score
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error')
mean_cv_score = -cv_scores.mean()
print(f'Cross-Validated MSE: {mean_cv_score}')

for i, (real, pred) in enumerate(zip(y_test, y_pred)):
    error = real - pred'''

'''Univariate Feature Selection in scikit-learn'''



from sklearn.feature_selection import SelectKBest, chi2, f_classif
selector  = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_train, y_train)
print('\nUnivariate Feature Selection in scikit-learn\n X_selected:\n', X_selected)

'''Recursive Feature Elimination with Random Forests'''

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

selector = RFE(model, n_features_to_select=10)

X_selected = selector.fit_transform(X_train, y_train)

selected_features_bool = selector.support_
print(selected_features_bool)

print('\nRecursive Feature Elimination with Random Forests\n X_selected: \n', X_selected)

'''Using Feature Importance for Feature Selection Python'''

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

importances = model.feature_importances_
top_10_features = X_train.columns[importances > 0.1]
X_selected = X_train[top_10_features]

print('\nUsing Feature Importance for Feature Selection Python\n X_selected:\n', X_selected)