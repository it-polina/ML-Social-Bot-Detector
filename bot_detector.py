#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Social Media Bot Detection System
Курсовая работа: Поиск ботов в социальных сетях (ВКонтакте)
"""

# =============================================================================
# ИМПОРТЫ БИБЛИОТЕК
# =============================================================================

# Для работы с VK API
import vk_api
from vk_api import ApiError

# Для работы с данными
import csv
import json
from json import JSONDecodeError
import numpy as np
import pandas as pd

# Для машинного обучения
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Для нейронной сети
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras import utils

# =============================================================================
# 1. РАБОТА С VK API
# =============================================================================

def generateVKAuthToken():
    # TODO: стоит наладить автоматическую генерацию токенов
    # конструктор для получения токена: https://oauth.vk.com/authorize?cliend_id=51816400&redirect_uri=https://api.vk.com/blank.html&scope=offline,messages&response_type=token
    sessionToken = ('vk1.a.ITdcLmt5rvciJUSzMT2jdScgzuQMtSXhr5nMrsO5nXEbpMJg'
                    'WDe41xYnpVwFziXX-ZKAPpdtuvhqyKPtS17UzVMiu1PHy1rFSlc576OFs1_TvFTlSGqk3-'
                    'k7iTs1BVXKXgsFf3K0-rkmaf-o2BeHpQVgS9lx0qJXBPi3A6E-CyApnm89UFb_ZBqF5YoUOpIo_7pcY5tqi_EEO5zyVjQV3g')
    return sessionToken

# возвращает интерфейс для осуществления запросов
def getVKSession():
    vkSession = vk_api.VkApi(token=generateVKAuthToken())
    return vkSession.get_api()

# возвращает список метрик для пользователя с id==usr_id
def getUserMetrics(userId, vkAPI=getVKSession()):
    # предусловие: пользователь с id==usr_id существует
    userMetrics = [userId]
    userInfo = vkAPI.users.get(user_id=userId, fields='counters')[0]  # возвращает list из одного json, поэтому из списка берем 1 элемент ([0])
    
    if(None != userInfo.get('deactivated')):
        # В этом случае данные, которые должны предоставляться при валидной странице,
        # нам не доступны, соответственно мы оставляем значения для этих метрик пустыми.
        # при обработке датафрейма эти ячейки заполнятся усредненными по столбцам значениями
        userMetrics.append(None)    # friends
        userMetrics.append(None)    # followers
        userMetrics.append(None)    # subscriptions
        userMetrics.append(None)    # wall posts count
        userMetrics.append(1 if 'deleted' == userInfo.get('deactivated') else 0)
        userMetrics.append(1 if 'banned' == userInfo.get('deactivated') else 0)
        return userMetrics
    
    userMetrics.append(userInfo['counters'].get('friends'))
    userMetrics.append(userInfo['counters'].get('followers'))
    userMetrics.append(userInfo['counters'].get('subscriptions'))
    
    # Если страница пользователя приватная, то wall.get выбросит исключение (30), поэтому этот запрос обрабатываем в блоке try/except
    try:
        wallPostsCnt = vkAPI.wall.get(owner_id=userId)['count']
        userMetrics.append(wallPostsCnt)
    except ApiError:
        userMetrics.append(None)
    
    userMetrics.append(0)      # аккаунт не забанен
    userMetrics.append(0)      # аккаунт не заморожен
    
    return userMetrics

# =============================================================================
# 2. СБОР ДАННЫХ
# =============================================================================

def constructDataset(datasetName='my_dataset.csv', usersCnt=500):
    vkAPI = getVKSession()
    datasetInstanceName = datasetName
    dsFile = open(datasetInstanceName, 'a')
    dsInstanceWriter = csv.writer(dsFile)
    
    dsHeader = ['user_id', 'friends_cnt', 'followers_cnt', 'subscriptions_cnt', 
                'wall_posts_cnt', 'is_deleted', 'is_banned', 'is_bot']
    dsInstanceWriter.writerow(dsHeader)
    
    sourceRaw = 'embeddings_info.json'
    
    with open(sourceRaw) as file:
        i = 0
        for line in file:
            # если нужное количество пользователей собрали - завершаем сбор данных
            if usersCnt <= i:
                break
            try:
                jsonObj = json.loads(line)
                usrMetrics = getUserMetrics(int(jsonObj['id']), vkAPI)
                usrMetrics.append(int(jsonObj['label']))
                dsInstanceWriter.writerow(usrMetrics)
                i += 1
                print('--- ', i, ' users processed ---')
            except JSONDecodeError:
                continue
        file.close()
    
    dsFile.close()
    print('Набор данных готов')

# =============================================================================
# 3. ПРЕДОБРАБОТКА ДАННЫХ
# =============================================================================

# возвращает 'pandas' датафрейм
def getProcDF(dsName):
    df = pd.read_csv(dsName)
    
    # удаляем дубликаты строк
    df = df.drop_duplicates(subset=['user_id'])
    
    # удаляем колонки с большим кол-вом пустых метрик
    colsToDel = []
    threshold = 0.8
    for col in df.columns:
        if df[col].isna().sum() / df.shape[0] > threshold:
            colsToDel.append(col)
    df = df.drop(columns=colsToDel)
    
    # заполняем пустоты в значениях метрик
    si = SimpleImputer()
    
    si.fit(df[['followers_cnt']])
    transformed = si.transform(df[['followers_cnt']])
    df['followers_cnt'] = transformed
    
    si = si.fit(df[['wall_posts_cnt']])
    transformed = si.transform(df[['wall_posts_cnt']])
    df['wall_posts_cnt'] = transformed
    
    si = si.fit(df[['subscriptions_cnt']])
    transformed = si.transform(df[['subscriptions_cnt']])
    df['subscriptions_cnt'] = transformed
    
    si = si.fit(df[['friends_cnt']])
    transformed = si.transform(df[['friends_cnt']])
    df['friends_cnt'] = transformed
    
    return df

# =============================================================================
# 4. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛЕЙ
# =============================================================================

# Загружаем обработанный датасет
botsInfoDF = getProcDF('bots_dataset.csv')

# Подготавливаем признаки и целевую переменную
x = botsInfoDF[['friends_cnt', 'followers_cnt', 'subscriptions_cnt', 
                'wall_posts_cnt', 'is_deleted', 'is_banned']]
y = botsInfoDF['is_bot']

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Словарь для хранения результатов
results = {}

# =============================================================================
# 5. ОБУЧЕНИЕ МОДЕЛЕЙ
# =============================================================================

# ---------- Дерево решений ----------
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)
predResult = model.predict(X_test)
desTreeRes = accuracy_score(y_test, predResult)
results['DecisionTree'] = desTreeRes
print('accuracy score of decision tree is ', desTreeRes)

# ---------- Случайный лес ----------
from sklearn.ensemble import RandomForestClassifier

randomForestModel = RandomForestClassifier(n_estimators=100)
randomForestModel = randomForestModel.fit(X_train, y_train)
RFPredResult = randomForestModel.predict(X_test)
RFRes = accuracy_score(y_test, RFPredResult)
results['RandomForest'] = RFRes
print('accuracy score of random forest classifier is ', RFRes)

# ---------- Наивный Байес ----------
from sklearn.naive_bayes import GaussianNB

NB_Model = GaussianNB()
NB_Model.fit(X_train, y_train)
NB_Pred = NB_Model.predict(X_test)
NB_Res = accuracy_score(y_test, NB_Pred)
results['NaiveBayes'] = NB_Res
print(NB_Res)

# ---------- Градиентный бустинг ----------
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

gradient_boosting = GradientBoostingClassifier(n_estimators=100)
gradient_boosting.fit(X_train, y_train)
GBPred = gradient_boosting.predict(X_test)
GBRes = accuracy_score(y_test, GBPred)
results['GradientBoosting'] = GBRes
print('accuracy_score of gradient boosting classifier is ', GBRes)

# выводим матрицу ошибок и отчет о классификации
print('\nconfusion matrix:\n', confusion_matrix(y_test, GBPred))
print('\nclassification report:\n', classification_report(y_test, GBPred))

# ---------- Многослойный Перцептрон (нейронная сеть) ----------
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np

# Создание копий данных для нормализации (чтобы не изменять исходные)
X_train_nn = X_train.copy()
X_test_nn = X_test.copy()

max_friends_cnt = 10000
max_followers_cnt = max_friends_cnt
max_subs_cnt = max_friends_cnt
max_wall_posts_cnt = max_friends_cnt

# Нормализация данных
X_train_nn['friends_cnt'] /= max_friends_cnt
X_train_nn['followers_cnt'] /= max_followers_cnt
X_train_nn['subscriptions_cnt'] /= max_subs_cnt
X_train_nn['wall_posts_cnt'] /= max_wall_posts_cnt

X_test_nn['friends_cnt'] /= max_friends_cnt
X_test_nn['followers_cnt'] /= max_followers_cnt
X_test_nn['subscriptions_cnt'] /= max_subs_cnt
X_test_nn['wall_posts_cnt'] /= max_wall_posts_cnt

# Создание модели
test_model = keras.Sequential([
    keras.layers.Input(shape=(6,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

test_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
test_model.fit(X_train_nn, y_train, epochs=80, batch_size=100, verbose=False)

# Предсказание и оценка точности
NNRes = accuracy_score(y_test, np.rint(test_model.predict(X_test_nn)))
results['NeuralNetwork'] = NNRes
print('accuracy_score of neural network prediction is ', NNRes)

# =============================================================================
# 6. ВЫВОД ВСЕХ РЕЗУЛЬТАТОВ
# =============================================================================

print("\n" + "="*50)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
print("="*50)
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")
