# Offline ASR на Faster-Whisper: speech_api + demo

Многоязычное офлайн-распознавание речи на базе Faster-Whisper (CTranslate2). Репозиторий содержит:

- библиотечный модуль speech_api.py — чистый и переиспользуемый API,
- CLI-утилиту demo.py — удобный интерфейс командной строки для проверки и использования.

Подходит для распознавания аудио из файлов, с микрофона и из NumPy-массивов, с поддержкой авто-детекта языка и слов с таймкодами.

Содержание:

- Описание функционала
- Принципы работы и архитектура
- API-справочник (функции, классы, структуры)
- Установка и зависимости
- Использование CLI: флаги, примеры, коды возврата
- Примеры использования API в Python
- Рекомендации по производительности
- Устранение неполадок (FAQ)

---

Возможности

- Многоязычное офлайн ASR на Faster-Whisper (CTranslate2)
- Источники звука:
  - аудиофайлы (через soundfile/librosa или напрямую через ffmpeg),
  - микрофон (через sounddevice),
  - NumPy-массив (1D float32, 16 кГц, моно)
- Язык:
  - авто-детект (по умолчанию),
  - фиксированный язык по коду (ru, en, …)
- Слова с таймкодами и вероятностями (word-level timestamps)
- Настройки качества/скорости:
  - устройство и тип вычислений: CPU/GPU, compute_type (int8, float16, float32 и др.)
  - ширина бима beam_size
- Логи:
  - в файл speech_api.log с ротацией,
  - в консоль (через demo.py)
- Чистый API без CLI-зависимостей, пригодный для интеграции в приложения

---

Принципы работы и архитектура

- Два слоя:
  1. speech_api.py — библиотека:
     - инкапсулирует загрузку модели и распознавание;
     - нормализует аудио, обрабатывает исключения, пишет логи;
     - предоставляет удобные функции для файлов/микрофона/массивов.
  2. demo.py — CLI:
     - принимает флаги и выводит результат с логами и форматированными словами/таймкодами.
- Аудиопайплайн:
  - Для файлов:
    - если установлен soundfile — читаем локально;
      - если sr=16000 — сразу в модель;
      - если sr!=16000 и есть librosa — локально ресемплируем в 16 кГц;
      - иначе — делегируем чтение и ресемплинг в ffmpeg (через Faster-Whisper).
    - если soundfile нет — сразу делегируем в ffmpeg.
  - Для микрофона:
    - записываем 16 кГц моно float32 через sounddevice,
    - фильтр «тишины»: если пусто/очень тихо — возвращаем пустой результат и language="und",
    - нормализация по пику (масштабирование до 1.0),
    - распознаём.
  - Для массивов:
    - ожидается 1D float32 моно 16 кГц (WHISPER_SAMPLE_RATE=16000).
- Логирование:
  - модуль speech_api пишет в speech_api.log (ротация до 2 МБ, 3 бэкапа),
  - demo.py включает вывод логов в консоль (корневой логгер + принудительный уровень для speech_api).
- Обработка ошибок:
  - единый тип исключений SpeechApiError с кодом ErrorCode (OK, INVALID_ARG, IO, MODEL_NOT_FOUND, AUDIO_DEVICE, INTERNAL и др.),
  - CLI возвращает коды завершения: 0 — успех; 1 — неверные аргументы/общая ошибка; 2 — ошибка уровня API.

---

API-справочник

Основные константы и структуры:

- WHISPER_SAMPLE_RATE = 16000
  - модель ожидает массивы 1D float32, моно, 16 кГц.
- class ErrorCode(IntEnum):
  - OK, INVALID_ARG, IO, MODEL_NOT_FOUND, AUDIO_DEVICE, INTERNAL, UNSUPPORTED_FORMAT, TIMEOUT, NOT_INITIALIZED
- class SpeechApiError(Exception):
  - Исключение API с полем .code (ErrorCode), удобное для обработки в приложениях.
- @dataclass Word:
  - start: float — начало слова (сек),
  - end: float — конец слова (сек),
  - word: str — текст слова,
  - conf: float — вероятность слова.
- @dataclass Transcription:
  - text: str — распознанный текст,
  - words: List[Word] — список слов с таймкодами/вероятностями,
  - language: str — код языка,
  - duration_s: float — оценённая длительность аудио,
  - raw: object — «сырой» info от модели (например, language_probability).

Класс Recognizer

- Recognizer(model_name_or_path, device="cpu", compute_type="int8")
  - Загружает модель Whisper (локальный путь к модели или имя — tiny/base/small/medium/large-v3, и т.п.).
  - Параметры:
    - device: "cpu" или "cuda"
    - compute_type: тип вычислений движка CTranslate2:
      - CPU: обычно "int8" (по умолчанию) или "float32"
      - GPU: обычно "float16" (рекомендуется)
- transcribe(audio, enable_words=True, language=None, beam_size=5, \*\*kwargs) -> Transcription
  - audio: путь к файлу (str) или 1D numpy.ndarray (float32, 16 кГц, моно).
  - language: None — авто-детект; строка — фиксированный язык (ru, en, …).
  - enable_words: включает словарные таймкоды.
  - beam_size: компромисс качество/скорость.
  - kwargs: пробрасывается в WhisperModel.transcribe (например, vad_filter=True и др.).
- close() -> None
  - Освобождение ресурсов (снятие ссылки на модель).

Публичные функции библиотеки

- init_recognizer(model_name_or_path, device="cpu", compute_type="int8") -> Recognizer
  - Инициализация распознавателя.
- free_recognizer(recognizer) -> None
  - Освобождение ресурсов.
- transcribe_wav(recognizer, wav_path, enable_words=True, language=None, beam_size=5, \*\*kwargs) -> Transcription
  - Распознавание из файла с логикой локального чтения и безопасным fallback’ом в ffmpeg.
  - При двойной ошибке (локальное чтение и затем ffmpeg) поднимается SpeechApiError(INTERNAL) с «сцеплённым» контекстом исключений.
- transcribe_microphone(recognizer, duration_s=5.0, device=None, enable_words=True, language=None, beam_size=5, \*\*kwargs) -> Transcription
  - Запись 16 кГц моно через sounddevice с последующим распознаванием.
  - При пустой/тихой записи возвращает пустой текст и language="und".
- list_audio_devices() -> List[dict]
  - Список доступных аудиоустройств ввода (приведённый к list[dict] для стабильной обработки .get()).
- version() -> str
  - Строка версии API, удобна для логов/диагностики.

---

Установка и зависимости

Требования

- Python 3.8+
- ОС: Linux, macOS, Windows
- Рекомендуется установленный ffmpeg в PATH (для чтения большинства форматов файлов через Faster-Whisper).

Минимальный набор

- pip install faster-whisper numpy

Дополнительно (опционально)

- Микрофон:
  - pip install sounddevice
  - Платформенные зависимости:
    - Linux: sudo apt-get install libportaudio2
    - macOS (Homebrew): brew install portaudio
    - Windows: обычно бинарные колёса уже содержат зависимости
- Локальное чтение аудио (без ffmpeg) и ресемплинг:
  - pip install soundfile librosa
  - Платформенные зависимости:
    - Linux/macOS: libsndfile обычно подтягивается автоматически
- ffmpeg:
  - Linux: sudo apt-get install ffmpeg
  - macOS: brew install ffmpeg
  - Windows: установить сборку и добавить ffmpeg.exe в PATH

GPU (опционально)

- Совместимая NVIDIA GPU, драйверы и CUDA (для faster-whisper/ctranslate2).
- Рекомендуемые параметры: --device cuda --compute-type float16.

---

Использование CLI (demo.py)

Справка

- python demo.py -h

Основные флаги

- Источники:
  - --wav PATH — распознать аудиофайл (WAV, MP3, …; любые форматы, поддерживаемые ffmpeg).
  - --mic SECONDS — записать с микрофона указанное число секунд и распознать.
  - --list-devices — вывести входные аудиоустройства.
- Модели:
  - --model NAME_OR_PATH — имя модели (tiny/base/small/medium/large-v3, …) или путь к локальной модели.
  - Логика выбора: явное --model > локальная папка models/faster-whisper-small > скачать "small".
- Вычисления:
  - --device cpu|cuda — устройство вычислений.
  - --compute-type TYPE — int8 (CPU), float16 (GPU), float32 и т.п.
- Распознавание:
  - --lang CODE|none|auto|detect — язык (None включает авто-детект; по умолчанию авто-детект).
  - --beam-size N — ширина бима.
  - --no-words — не вычислять word-level timestamps (быстрее).
  - --words-mode words|times|both — формат вывода слов:
    - words — только слова;
    - times — таймкоды и вероятности;
    - both — и список, и таймкоды.
- Логи:
  - --log-level critical|error|warning|info|debug — уровень логов в консоль (по умолчанию info).

Коды возврата

- 0 — успех
- 1 — неверные аргументы/общая ошибка
- 2 — ошибка уровня SpeechApiError

Примеры

- Распознать файл с авто-детектом языка:
  - python demo.py --wav sample.wav
- Принудительно задать язык (более стабильно на коротких фразах):
  - python demo.py --wav sample.wav --lang ru
- Вывести слова с таймкодами и вероятностями:
  - python demo.py --wav sample.wav --words-mode times
- Отключить word-level (быстрее):
  - python demo.py --wav sample.wav --no-words
- Использовать GPU:
  - python demo.py --wav sample.wav --device cuda --compute-type float16
- Увеличить beam_size (чуть медленнее, чуть качественнее):
  - python demo.py --wav sample.wav --beam-size 8
- Запись с микрофона (7 секунд):
  - python demo.py --mic 7
- Показать список входных устройств:
  - python demo.py --list-devices
- Явно указать модель:
  - python demo.py --wav sample.wav --model small
  - python demo.py --wav sample.wav --model models/faster-whisper-small

Вывод языка

- При авто-детекте: печатается «Язык (определён): <code> (p=…)».
- При фиксированном языке: «Язык (задан): <code>».
- Примечание: на коротких/нейтральных фразах авто-детект может ошибаться.

---

Примеры использования API (Python)

Загрузка модели и распознавание файла

```python
from speech_api import init_recognizer, transcribe_wav, free_recognizer

rec = init_recognizer("small", device="cpu", compute_type="int8")
try:
    tr = transcribe_wav(rec, "sample.wav", enable_words=True, language=None, beam_size=5)
    print("Language:", tr.language)
    print("Text:", tr.text)
    for w in tr.words[:5]:
        print(f"{w.start:.2f}-{w.end:.2f}: {w.word} (p={w.conf:.2f})")
finally:
    free_recognizer(rec)
```

Распознавание NumPy-массива (1D float32, 16 кГц, моно)

```python
import numpy as np
import soundfile as sf
import librosa
from speech_api import init_recognizer, Recognizer, WHISPER_SAMPLE_RATE

# Читаем произвольное аудио, приводим к 16 кГц моно float32:
audio, sr = sf.read("sample_any_format.flac", dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)  # моно
if sr != WHISPER_SAMPLE_RATE:
    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=WHISPER_SAMPLE_RATE).astype("float32")

rec = init_recognizer("small")
tr = rec.transcribe(audio, enable_words=True, language=None)
print(tr.text)
```

Распознавание записи с микрофона (через API)

```python
from speech_api import init_recognizer, transcribe_microphone, free_recognizer

rec = init_recognizer("small", device="cuda", compute_type="float16")
try:
    tr = transcribe_microphone(rec, duration_s=5.0, language=None, enable_words=True)
    if tr.language == "und":
        print("Слишком тихая/пустая запись. Попробуйте снова.")
    else:
        print(tr.text)
finally:
    free_recognizer(rec)
```

Перебор аудиоустройств ввода

```python
from speech_api import list_audio_devices

for i, d in enumerate(list_audio_devices()):
    print(i, d.get("name"), d.get("max_input_channels"))
```

---

Рекомендации по производительности

- Используйте GPU при наличии:
  - --device cuda --compute-type float16
- Выбирайте размер модели по задаче:
  - tiny/base/small — быстрее, хуже качество,
  - medium/large-v3 — лучше качество, медленнее и ресурсоёмко.
- Уменьшайте вычислительную нагрузку:
  - отключайте слово-таймкоды, если они не нужны: --no-words,
  - подбирайте beam_size (5–8 — разумный диапазон; меньше — быстрее).
- Подавайте уже подготовленный 16 кГц моно float32 массив (минует ffmpeg и часть конвертаций).

---

Устранение неполадок (FAQ)

Модель не загружается (MODEL_NOT_FOUND)

- Проверьте имя/путь к модели (--model).
- Если модель скачивается автоматически, нужна сеть. Для офлайн-режима заранее скачайте модель и укажите путь.
- Убедитесь, что путь models/faster-whisper-small существует, если рассчитываете на локальную копию.

Ошибка аудиоустройства (AUDIO_DEVICE)

- Убедитесь, что установлен sounddevice и системные зависимости:
  - Linux: sudo apt-get install libportaudio2
  - macOS: brew install portaudio (и разрешите доступ к микрофону в настройках ОС)
  - Windows: проверьте разрешения на микрофон.
- Посмотрите список устройств: python demo.py --list-devices
- В API можно указать индекс устройства: transcribe_microphone(..., device=IDX)

Пустой/очень тихий микрофон: язык "und"

- Говорите громче и ближе к микрофону.
- Увеличьте duration_s для более длинной записи.
- Проверьте выбранное устройство ввода.

Файл не читается/ошибка ffmpeg (INTERNAL)

- Установите ffmpeg и убедитесь, что он в PATH.
- Попробуйте конвертировать файл в WAV 16 кГц моно.
- В журнал speech_api.log пишется подробный стек.

Авто-детект языка ошибается

- Для коротких/нейтральных фраз используйте фиксированный язык: --lang ru (или en).
- Используйте более длинные фразы, более крупную модель или GPU.

Низкая скорость на CPU

- Используйте compute_type=int8 (по умолчанию).
- Отключите word-level timestamps: --no-words.
- Рассмотрите GPU с float16.

---

Примечания

- list_audio_devices всегда возвращает list[dict], чтобы безопасно использовать .get().
- В transcribe_microphone язык для пустой/тихой записи всегда "und" — это намеренно, для однозначной обработки.
- В transcribe_wav улучшена диагностика: при двойной ошибке (soundfile и затем ffmpeg) сообщение укажет, что упали на ffmpeg, а контекст первой ошибки сохранится.
- Массивы, передаваемые в Recognizer.transcribe, должны быть 1D float32, моно и 16 кГц. Если это не так — используйте путь к файлу, и Faster-Whisper выполнит конвертацию через ffmpeg.

---

Лицензия и благодарности

- Faster-Whisper (CTranslate2) — спасибо авторам проекта.
- Данный модуль/CLI — пример интеграции для офлайн-распознавания речи.
