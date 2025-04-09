import pandas as pd
import dill
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
with open('model/car_pipe.pkl', 'rb') as file:
    model = dill.load(file)
with open('model/future_d.pkl', 'rb') as file:
    future_d = dill.load(file)


class Form(BaseModel):
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str



class Prediction(BaseModel):
    id: str
    pred: str


def device_os_na(df):
    df_sessions = df.copy()
    device_os_values = set(df_sessions.device_os.unique())
    device_os_values.discard('(not set)')
    device_os_values.discard(np.nan)
    for os in device_os_values:
        all_device_Android = df_sessions[df_sessions.device_os == os].device_brand.unique().tolist()
        if "(not set)" in all_device_Android:
            all_device_Android.remove("(not set)")
        for phone in all_device_Android:
            df_sessions.loc[
                (df_sessions['device_os'].isna()) & 
                (df_sessions['device_brand'] == phone),
                'device_os'
            ] = os
    android = ['Instagram 208.0.0.32.135 Android', 'Android Webview',
       'Instagram 209.0.0.21.119 Android', 'Android', 'Instagram 199.1.0.34.119 Android',
       'Instagram 194.0.0.36.172 Android', 'Opera Mini', 'Puffin',
       'Instagram 202.0.0.37.123 Android', 'Samsung Internet', 'com.vk.vkclient',
       'Instagram 192.0.0.35.123 Android', 'Android Browser', 'UC Browser'
       'Instagram 158.0.0.30.123 Android', 'Android Runtime', 'Threads 202.0.0.23.119']
    df_sessions.loc[
        (df_sessions['device_os'].isna()) & 
        (df_sessions['device_browser'].isin(android)),
        'device_os'
    ] = 'Android'
    browsers_os = {
    'YaBrowser': 'Windows',  # Чаще используется на Windows, но есть версии для macOS и Android.
    'Chrome': 'Windows',  # Самый распространённый браузер, но предполагаем Windows.
    'Safari': 'iOS',  # Safari — браузер Apple, работает только на macOS и iOS.
    'Firefox': 'Windows',  # Firefox кроссплатформенный, но чаще встречается на Windows.
    'Opera': 'Windows',  # Opera работает везде, но в основном на Windows.
    'Edge': 'Windows',  # Браузер от Microsoft, стандартный для Windows.
    '(not set)': '(not set)',  # Нет информации о браузере, операционную систему определить нельзя.
    'Mozilla Compatible Agent': 'Windows',  # Это может быть что угодно, требует уточнения.
    'Coc Coc': 'Windows',  # Вьетнамский браузер на основе Chromium, доступен для Windows и macOS.
    '[FBAN': 'iOS',  # User-Agent Facebook App, чаще всего встречается на iOS.
    'Internet Explorer': 'Windows',  # Классический браузер Windows.
    'MRCHROME': 'Windows',  # Вероятно, специфическая версия Chrome.
    'UC Browser': 'Android',  # Браузер популярен среди пользователей Android.
    'SeaMonkey': 'Windows',  # Браузер на основе Mozilla, чаще используется на Windows.
    'Mozilla': 'Windows',  # Может быть разным, но чаще всего это Windows.
    'Maxthon': 'Windows',  # Браузер, работающий на разных платформах, но чаще на Windows.
    'Konqueror': 'Linux'  # Браузер KDE, используется в основном на Linux.
    }

    df_sessions.loc[
        (df_sessions['device_os'].isna()) & 
        (df_sessions['device_browser'].isin(browsers_os.keys())),
        'device_os'
    ] = df_sessions['device_browser'].map(browsers_os)
    return df_sessions


def device_brand_na(df):
    df_sessions = df.copy()
    df_sessions.loc[
    (df_sessions['device_brand'].isna()) & 
    (df_sessions['device_os'] == 'Macintosh'),
    'device_brand'
    ] = 'Apple'
    df_sessions.loc[
        (df_sessions['device_brand'].isna()) & 
        (df_sessions['device_os'] == 'Windows') &
        (df_sessions['device_category'] == 'desktop'),
        'device_brand'
        ] = '(not set)'
    df_sessions['device_brand'] = df_sessions['device_brand'].fillna('(not set)')
    return df_sessions


def feature_new(df):
    df_sessions = df.copy()
    
    # Создание новой колонки organic_traffic
    organic_traffic = ['organic', 'referral', '(none)']
    df_sessions['organic_traffic'] = df_sessions['utm_medium'].apply(lambda x: 1 if x in organic_traffic else 0)
    
    social_media = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs', 
                      'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    df_sessions['social_media_ad'] = df_sessions['utm_source'].apply(lambda x: 1 if x in social_media else 0)
    
    df_sessions['visit_date'] = pd.to_datetime(df_sessions['visit_date'])
    df_sessions['year'] = df_sessions['visit_date'].dt.year
    df_sessions['month'] = df_sessions['visit_date'].dt.month
    df_sessions['day'] = df_sessions['visit_date'].dt.day
    
    df_sessions['visit_time'] = pd.to_timedelta(df_sessions['visit_time'])

    # Извлекаем компоненты времени
    df_sessions['hour'] = (df_sessions['visit_time'].dt.total_seconds() // 3600).astype(int)
    df_sessions['minute'] = ((df_sessions['visit_time'].dt.total_seconds() % 3600) // 60).astype(int)
    df_sessions['second'] = (df_sessions['visit_time'].dt.total_seconds() % 60).astype(int)
    
    groups = {
    'ad_channel': ['banner', 'cpm', 'cpc', 'cpv', 'cpa', 'smartbanner', 'promo_sbol', 
                   'promo_sber', 'reach', 'desktop', 'tablet', 'static', 'CPM'],
    
    'organic_channel': ['organic', '(none)', '(not set)'],
    
    'referral_channel': ['referral', 'link', 'linktest', 'qr', 'qrcodevideo'],
    
    'social': ['smm', 'stories', 'blogger_channel', 'blogger_stories', 
               'blogger_header', 'vk_smm', 'fb_smm', 'ok_smm', 'tg', 'social'],
    
    'email_sms': ['email', 'sms', 'outlook'],
    
    'app_catalogue': ['app', 'sber_app', 'Sbol_catalog', 'catalogue', 
                      'landing_interests', 'web_polka', 'main_polka'],
    
    'other_channel': ['push', 'partner', 'post', 'article', 'users_msk', 
                      'cbaafe', 'dom_click', 'medium', 'last', 'clicks', 
                      'landing', 'info_text', 'nkp', 'google_cpc', 'yandex_cpc']
    }

    def get_group(value):
        for group, values in groups.items():
            if value in values:
                return group
        return None

    df_sessions['utm_medium_group'] = df_sessions['utm_medium'].apply(get_group)
    
    # Разбиваем строку по разделителю 'x' и создаём новые колонки
    df_sessions[['screen_width', 'screen_height']] = df_sessions['device_screen_resolution'].str.split('x', expand=True)

    # Преобразуем в числовые значения
    df_sessions['screen_width'] = pd.to_numeric(df_sessions['screen_width'], errors='coerce')
    df_sessions['screen_height'] = pd.to_numeric(df_sessions['screen_height'], errors='coerce')
    
    utm_mapping = {
    'banner': 'paid', 'cpm': 'paid', 'CPM': 'paid', 'cpv': 'paid', 'cpc': 'paid',
    'google_cpc': 'paid', 'yandex_cpc': 'paid', 'cpa': 'paid', 'partner': 'paid',

    'organic': 'organic', 'referral': 'organic', 'social': 'organic',

    'smm': 'social', 'fb_smm': 'social', 'vk_smm': 'social', 'ok_smm': 'social',
    'tg': 'social', 'stories': 'social', 'blogger_channel': 'social',
    'blogger_stories': 'social', 'blogger_header': 'social', 'post': 'social',

    'email': 'direct', 'sms': 'direct', 'push': 'direct',

    '(none)': 'unknown', '(not set)': 'unknown'
    }

    df_sessions['utm_category'] = df_sessions['utm_medium'].map(lambda x: utm_mapping.get(x, 'other'))
    
    
    category = ['utm_source', 'utm_medium', 'utm_campaign',
       'utm_adcontent', 'utm_keyword', 'device_category', 'device_os',
       'device_brand', 'device_model', 'device_screen_resolution',
       'device_browser', 'geo_country', 'geo_city', 'utm_medium_group',
       'utm_category']

    for feature in category:
        df_sessions[feature +'_numeric'] = df_sessions[feature].map(lambda x: future_d[feature].get(x) if pd.notna(x) else 0)
    df_sessions = df_sessions.drop(columns=category)    
    df_sessions['screen_width'] = df_sessions['screen_width'].fillna(df_sessions['screen_width'].median())
    df_sessions['screen_height'] = df_sessions['screen_height'].fillna(df_sessions['screen_height'].median())

    df_sessions = df_sessions.drop(['visit_date', 'visit_time', 'client_id'], axis=1)
    
    return df_sessions


def preprosessing(df):
    return feature_new(device_brand_na(device_os_na(df)))


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = preprosessing(pd.DataFrame.from_dict([form.dict()]))
    y = model['model'].predict(df)
    return {
        "id": form.client_id,
        "pred": str(y[0])
    }
