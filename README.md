# BTS Parkinson Web

Веб-сервис для анализа мимики при болезни Паркинсона.

## Описание работы

Пользователь загружает два коротких видео: улыбку и зажмуривание. Сервис покадрово обрабатывает видео через MediaPipe Face Mesh, считает индексы EAR для глаз и MAR для рта, нормирует временные ряды и извлекает шесть скалярных признаков. На собранных данных обучаются три классификатора с LOOCV-валидацией. По каждой сессии генерируется отчёт с графиками и заключением.

## Стек

| Слой | Чем сделано |
|------|-------------|
| Бэкенд | FastAPI, SQLAlchemy 2.0, SQLite |
| Видео | OpenCV, MediaPipe Face Mesh |
| ML | scikit-learn: Random Forest, SVM RBF, Logistic Regression, LOOCV |
| Аналитика | NumPy, SciPy, pandas, matplotlib, seaborn |
| Аутентификация | JWT через python-jose, bcrypt |

## Структура

```
bts_parkinson_web/
├── app/
│   ├── main.py                   точка входа и роуты
│   ├── auth.py                   JWT, хеши паролей
│   ├── config.py                 настройки
│   ├── crud.py                   операции с БД
│   ├── db.py                     подключение к SQLite
│   ├── models.py                 SQLAlchemy-модели
│   ├── schemas.py                pydantic-схемы
│   ├── processing/
│   │   ├── pipeline.py           оркестрация
│   │   ├── indices.py            EAR, MAR
│   │   ├── normalize.py          нормализация
│   │   ├── features.py           извлечение признаков
│   │   ├── stats.py              статистические тесты
│   │   ├── ml.py                 классификация
│   │   ├── plots.py              графики
│   │   ├── report.py             генерация отчёта
│   │   └── conclusion.py         формулировка вывода
│   └── static/                   фронтенд
├── notebooks/
│   └── analysis.ipynb            аналитика: статистика, ML, графики
├── scripts/                      вспомогательные скрипты
├── requirements.txt
└── .env.example
```

## Запуск

Нужен Python 3.9+.

```bash
git clone <url> bts_parkinson_web
cd bts_parkinson_web
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Поправьте `SECRET_KEY` в `.env`, затем:

```bash
uvicorn app.main:app --reload
```

Сервис поднимется на `http://127.0.0.1:8000`. Первого администратора создаём через bootstrap:

```bash
curl -X POST http://127.0.0.1:8000/auth/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"...","full_name":"..."}'
```

## Аналитика

`notebooks/analysis.ipynb` содержит полный разбор: описательная статистика, боксплоты, тест Манна-Уитни, LOOCV с тремя классификаторами, confusion matrices, ROC и PR-кривые, важность признаков. GitHub рендерит ноутбук с графиками без запуска.

Чтобы прогнать на своих данных:

```bash
jupyter notebook notebooks/analysis.ipynb
```
