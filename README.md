# MVP LightGBM Price Project

Проект для предсказания цены единицы продукции с использованием модели LightGBM.

## Обучение модели

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Настройка переменных окружения

**Linux/Mac (bash):**
```bash
export MLFLOW_TRACKING_URI=file:./mlruns
export MLFLOW_EXPERIMENT=mvp_lightgbm_price
# Опционально: export MLFLOW_REGISTER_MODEL=mvp-lightgbm-price
```

**Windows (PowerShell):**
```powershell
$env:MLFLOW_TRACKING_URI="file:./mlruns"
$env:MLFLOW_EXPERIMENT="mvp_lightgbm_price"
# Опционально: $env:MLFLOW_REGISTER_MODEL="mvp-lightgbm-price"
```

**Windows (CMD):**
```cmd
set MLFLOW_TRACKING_URI=file:./mlruns
set MLFLOW_EXPERIMENT=mvp_lightgbm_price
REM Опционально: set MLFLOW_REGISTER_MODEL=mvp-lightgbm-price
```

### Запуск обучения

```bash
python -m ml.train --csv data/raw/mvp_quotes.csv
```

#### Дополнительные параметры обучения

**Настройка количества испытаний для подбора гиперпараметров:**
```bash
python -m ml.train --csv data/raw/mvp_quotes.csv --trials 100
```

**Использование гиперпараметров из предыдущего запуска:**
```bash
python -m ml.train --csv data/raw/mvp_quotes.csv --use-run-id <run_id>
```

После обучения модель будет сохранена в директории `mlruns/` с метриками (RMSE, MAE, MAPE, R²).

## Сборка Docker-контейнера

### Вариант 1: Использование скриптов (Рекомендуется)

**Linux/Mac:**
```bash
./docker/build.sh
```

**Windows PowerShell:**
```powershell
.\docker\build.ps1
```

### Вариант 2: Ручная сборка

```bash
docker build -t model-api:latest .
```

### Вариант 3: Использование Docker Compose

```bash
docker-compose build
```

## Запуск Docker-контейнера

### Вариант 1: Использование скриптов (Рекомендуется)

**Linux/Mac:**
```bash
./docker/run.sh
```

**Windows PowerShell:**
```powershell
.\docker\run.ps1
```

### Вариант 2: Ручной запуск

**Автоматическое определение последней модели:**
```bash
docker run -p 8000:8000 model-api:latest
```

**Указание конкретной модели:**
```bash
docker run -p 8000:8000 \
  -e MODEL_URI="runs:/<run_id>/model" \
  -e MLFLOW_TRACKING_URI="file:./mlruns" \
  model-api:latest
```

### Вариант 3: Использование Docker Compose

```bash
docker-compose up
```

Или с пересборкой:
```bash
docker-compose up --build
```

### Проверка работы API

После запуска контейнера API будет доступен по адресу `http://localhost:8000`

**Проверка здоровья сервиса:**
```bash
curl http://localhost:8000/healthz
```

**Документация API:**
Откройте в браузере: `http://localhost:8000/docs`

**Пример запроса на предсказание:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "rfq_id": 1,
      "customer_tier": "A",
      "material": "steel",
      "thickness_mm": 5.0,
      "length_mm": 1000.0,
      "width_mm": 500.0,
      "material_cost_rub": 1000.0,
      "labor_minutes_per_unit": 10.0,
      "labor_cost_rub": 500.0,
      "qty": 10
    }
  ]'
```

## Особенности Docker-контейнера

- **Автоматическое определение модели**: Если `MODEL_URI` не установлен, контейнер автоматически найдет последнюю обученную модель
- **Проверка здоровья**: Встроенный эндпоинт для проверки здоровья на `/healthz`
- **Оптимизированный образ**: Включает только последнюю модель для уменьшения размера образа
- **Готовность к продакшену**: Включает обработку ошибок и логирование

## Переменные окружения

- `MODEL_URI`: URI модели MLflow (например, `runs:/<run_id>/model`). Если не установлен, скрипт entrypoint попытается автоматически найти последнюю модель
- `MLFLOW_TRACKING_URI`: URI для отслеживания MLflow (по умолчанию `file:./mlruns`)
- `PYTHONPATH`: Установлен в `/app` по умолчанию
- `PORT`: Внутренний порт (по умолчанию 8000)

## Устранение неполадок

**Контейнер не запускается:**
- Убедитесь, что директория `mlruns/` существует и содержит данные модели
- Проверьте, что `MODEL_URI` установлен правильно или что скрипт entrypoint может найти модель
- Проверьте логи контейнера: `docker logs <container_id>`

**Модель не найдена:**
- Убедитесь, что последний запуск модели включен в Docker-образ
- Вручную установите переменную окружения `MODEL_URI`

**Порт уже используется:**
- Измените маппинг порта: `docker run -p 8001:8000 ...`
- Или остановите существующий контейнер, использующий порт 8000

