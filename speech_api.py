# -*- coding: utf-8 -*-
"""
speech_api.py — API распознавания речи на базе Faster-Whisper.

Возможности:
  - Распознавание из файла (любой формат, поддерживаемый ffmpeg) или из массива numpy (16 kHz, float32, моно).
  - Запись с микрофона и распознавание.
  - Словарные таймкоды (word_timestamps).
  - Конфигурируемая загрузка модели (CPU/GPU).

Зависимости (минимум):
  pip install faster-whisper numpy

Опционально:
  - Микрофон: pip install sounddevice
  - Локальное чтение файлов: pip install soundfile
  - Локальная передискретизация: pip install librosa

Примечание:
  - Если soundfile/librosa не установлены, можно передавать путь к файлу — Faster-Whisper использует ffmpeg для декодирования/ресемплинга.

Лог-файл: speech_api.log
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import IntEnum
from logging.handlers import RotatingFileHandler
from queue import Queue
from typing import List, Optional, Union

import numpy as np

# --- ОПЦИОНАЛЬНЫЕ ЗАВИСИМОСТИ ---
# sounddevice — для записи с микрофона.
try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # type: ignore

# soundfile — для локального чтения аудиофайлов.
try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None  # type: ignore

# librosa — для локальной передискретизации.
try:
    import librosa  # type: ignore
except Exception:
    librosa = None  # type: ignore

# faster-whisper — обязательная зависимость.
try:
    from faster_whisper import WhisperModel
except Exception as e:
    # Принципиальный момент:
    # - Явно падаем с понятным сообщением, если основной движок не установлен.
    # - Это ранняя проверка, чтобы не скрывать проблему глубже в стеке.
    raise RuntimeError(
        "Faster-Whisper не установлен. Выполните: pip install faster-whisper\nОригинальная ошибка: %s" % e
    )

# ----------------------- ЛОГИРОВАНИЕ -----------------------
# Принципиальный момент: выделенный логгер модуля с ротацией файла.
# - Все сообщения пишутся в speech_api.log (do 2 МБ, 3 бэкапа).
# - Сообщения также «поднимаются» к root-логгеру и могут быть отображены в консоли
#   при настройке в приложении (см. demo.py -> setup_logging()).
LOGGER = logging.getLogger("speech_api")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    _fh = RotatingFileHandler("speech_api.log", encoding="utf-8", maxBytes=2_000_000, backupCount=3)
    _fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    _fh.setFormatter(_fmt)
    LOGGER.addHandler(_fh)

# ---------------------- КОНСТАНТЫ --------------------------
# Whisper ожидает float32 моно 16 kHz при передаче массива numpy.
WHISPER_SAMPLE_RATE = 16000

# ---------------------- КЛАССЫ ОШИБОК И ДАННЫХ --------------------------

class ErrorCode(IntEnum):
    OK = 0
    INVALID_ARG = 1
    IO = 2
    MODEL_NOT_FOUND = 3
    AUDIO_DEVICE = 4
    INTERNAL = 5
    UNSUPPORTED_FORMAT = 6
    TIMEOUT = 7
    NOT_INITIALIZED = 8


class SpeechApiError(Exception):
    """
    Исключение API с кодом ошибки.
    Принципиальный момент: единый тип исключений упрощает обработку ошибок в CLI/приложениях.
    """
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Word:
    start: float
    end: float
    word: str
    conf: float


@dataclass
class Transcription:
    text: str
    words: List[Word]
    language: str
    duration_s: float
    raw: object  # Сырой ответ от Whisper (info)


class Recognizer:
    """
    Обёртка над Faster-Whisper:
      - Загружает модель.
      - Распознаёт как массив numpy (float32, 16 kHz, моно), так и путь к файлу.

    Принципиальные моменты:
      - Явное управление устройством/типом вычислений (CPU/GPU).
      - Любые внутренние ошибки движка заворачиваются в SpeechApiError(INTERNAL),
        чтобы вызывающая сторона могла единообразно реагировать.
    """
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        try:
            self._model = WhisperModel(model_name_or_path, device=device, compute_type=compute_type)
            LOGGER.info(
                "Модель Whisper загружена: %s (device=%s, compute_type=%s)",
                model_name_or_path, device, compute_type
            )
        except Exception as e:
            LOGGER.exception("Ошибка загрузки модели Whisper")
            raise SpeechApiError(ErrorCode.MODEL_NOT_FOUND, f"Ошибка загрузки модели '{model_name_or_path}': {e}")

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        enable_words: bool = True,
        language: Optional[str] = None,   # None => авто-детект
        beam_size: int = 5,
        **kwargs
    ) -> Transcription:
        """
        Распознаёт аудио:
          - audio: путь к файлу (str) ИЛИ массив numpy (float32, 16 kHz, моно).
          - enable_words: включить словарные таймкоды (word_timestamps).
          - language: код языка (например 'ru', 'en') или None для авто-детекта.
          - beam_size: ширина бима (качество/скорость).
          - **kwargs: дополнительные параметры Faster-Whisper (vad_filter и др.).

        Принципиальные моменты:
          - На вход можно передавать как путь к файлу, так и уже подготовленный массив.
          - Если language=None, движок выполнит авто-детект языка.
        """
        try:
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                word_timestamps=enable_words,
                **kwargs
            )

            full_text = ""
            all_words: List[Word] = []
            for segment in segments:
                full_text += segment.text + " "
                if enable_words and getattr(segment, "words", None):
                    for w in segment.words:
                        all_words.append(
                            Word(
                                start=float(w.start),
                                end=float(w.end),
                                word=str(w.word).strip(),
                                conf=float(getattr(w, "probability", 0.0)),
                            )
                        )

            return Transcription(
                text=full_text.strip(),
                words=all_words,
                language=str(getattr(info, "language", language or "")),
                duration_s=float(getattr(info, "duration", 0.0)),
                raw=info,
            )
        except Exception as e:
            LOGGER.exception("Ошибка во время распознавания Whisper")
            raise SpeechApiError(ErrorCode.INTERNAL, f"Ошибка распознавания: {e}")

    def close(self) -> None:
        # У Faster-Whisper нет явного метода освобождения,
        # держим ссылку на модель, чтобы сборщик мусора освободил память GPU/CPU.
        self._model = None


# ------------------------ ПУБЛИЧНЫЕ API ФУНКЦИИ -----------------------

def init_recognizer(
    model_name_or_path: str,
    device: str = "cpu",
    compute_type: str = "int8",
    **kwargs
) -> Recognizer:
    """Инициализирует распознаватель (загружает модель Whisper в память)."""
    return Recognizer(model_name_or_path=model_name_or_path, device=device, compute_type=compute_type)


def free_recognizer(recognizer: Optional[Recognizer]) -> None:
    """Освобождает ресурсы распознавателя."""
    if recognizer:
        recognizer.close()


def _to_mono_1d_float32(audio_data: np.ndarray) -> np.ndarray:
    """
    Приводит массив к формату 1D float32 моно.
    Принципиальный момент: модель ожидает 1-канальный поток 16 кГц float32 (см. WHISPER_SAMPLE_RATE).
    """
    if audio_data.ndim == 2:
        if audio_data.shape[1] > 1:
            # Смешивание каналов в моно: среднее по каналам.
            audio_data = np.mean(audio_data, axis=1)
        else:
            # Уплощение (N,1) -> (N,)
            audio_data = audio_data[:, 0]
    audio_data = np.asarray(audio_data, dtype=np.float32)
    return audio_data


def transcribe_wav(
    recognizer: Recognizer,
    wav_path: str,
    enable_words: bool = True,
    language: Optional[str] = None,
    beam_size: int = 5,
    **kwargs
) -> Transcription:
    """
    Распознаёт речь из аудиофайла.
    Поддерживаются любые форматы, которые умеет ffmpeg (через Faster-Whisper),
    а также прямое чтение через soundfile (если установлено).

    Логика:
      1) Если установлен soundfile — пытаемся прочитать локально.
         - Если sr == 16000 — отдаём массив в модель напрямую.
         - Если sr != 16000:
             * если установлен librosa — передискретизируем локально и отдаём массив;
             * иначе — используем прямой путь (делегируем в ffmpeg/Faster-Whisper).
      2) Если soundfile не установлен — сразу отдаём путь к файлу в модель.

    Принципиальные моменты:
      - «Оптимистичный» локальный путь экономит накладные расходы на ffmpeg.
      - Гарантированная деградация к ffmpeg даёт совместимость со множеством форматов.
      - В случае двойного падения (локальное чтение и затем ffmpeg) возвращаем корректное,
        «сочленённое» (chained) исключение для удобной диагностики.
    """
    if not os.path.isfile(wav_path):
        raise SpeechApiError(ErrorCode.IO, f"Файл не найден: {wav_path}")

    def _fallback_via_path() -> Transcription:
        LOGGER.info("Чтение файла делегировано Faster-Whisper (ffmpeg). Путь: %s", wav_path)
        return recognizer.transcribe(
            wav_path, enable_words=enable_words, language=language, beam_size=beam_size, **kwargs
        )

    if sf is None:
        # Нет soundfile — сразу используем ffmpeg через движок Faster-Whisper.
        return _fallback_via_path()

    try:
        audio_data, samplerate = sf.read(wav_path, dtype="float32")  # type: ignore
        audio_data = _to_mono_1d_float32(audio_data)

        if samplerate == WHISPER_SAMPLE_RATE:
            return recognizer.transcribe(
                audio_data, enable_words=enable_words, language=language, beam_size=beam_size, **kwargs
            )

        if librosa is not None:
            LOGGER.info(
                "Частота дискретизации файла %d Гц. Передискретизация в %d Гц через librosa...",
                samplerate, WHISPER_SAMPLE_RATE
            )
            # Принципиальный момент: локальная передискретизация избегает внешних зависимостей и ускоряет пайплайн.
            audio_data = librosa.resample(y=audio_data, orig_sr=samplerate, target_sr=WHISPER_SAMPLE_RATE)  # type: ignore
            LOGGER.info("Передискретизация завершена.")
            audio_data = _to_mono_1d_float32(audio_data)
            return recognizer.transcribe(
                audio_data, enable_words=enable_words, language=language, beam_size=beam_size, **kwargs
            )

        # Нет librosa — делегируем чтение/ресемплинг во внутрь модели (ffmpeg).
        return _fallback_via_path()

    except SpeechApiError:
        # Уже корректный тип исключения API — прокидываем без изменений.
        raise
    except Exception as e:
        # Локальное чтение упало — пробуем через ffmpeg.
        LOGGER.warning("Локальное чтение через soundfile не удалось (%s). Пытаемся через ffmpeg...", e)
        try:
            return _fallback_via_path()
        except Exception as e2:
            # Принципиальный момент:
            # - Сообщаем, что финально упали именно на шаге ffmpeg.
            # - Сохраняем исходное исключение 'e' как __cause__ (chained), чтобы не терять контекст первой ошибки.
            LOGGER.exception("Ошибка при распознавании файла")
            raise SpeechApiError(ErrorCode.INTERNAL, f"Ошибка распознавания файла (ffmpeg): {e2}") from e


def transcribe_microphone(
    recognizer: Recognizer,
    duration_s: float = 5.0,
    device: Optional[int] = None,
    enable_words: bool = True,
    language: Optional[str] = None,
    beam_size: int = 5,
    **kwargs
) -> Transcription:
    """
    Распознаёт речь с микрофона в течение `duration_s` секунд.

    Принципиальные моменты:
      - Запись ведётся с частотой 16 кГц, моно, float32 — сразу в «правильном» формате для модели.
      - Встроенный «фильтр тишины»: если запись пустая/слишком тихая — возвращаем пустой результат
        и язык 'und' (undefined), чтобы вызывающий код мог корректно реагировать.
      - Простая нормализация по пиковому уровню для улучшения устойчивости распознавания.
    """
    if sd is None:
        raise SpeechApiError(ErrorCode.AUDIO_DEVICE, "Модуль sounddevice не установлен")

    q: Queue = Queue()

    def _callback(indata, frames, time_info, status):
        if status:
            LOGGER.warning("Статус аудиопотока: %s", status)
        q.put(indata.copy())

    LOGGER.info("Начало записи с микрофона: %.2f сек", duration_s)
    try:
        # Важно: задаём samplerate/channels под модель, чтобы не делать конвертацию позже.
        with sd.InputStream(
            samplerate=WHISPER_SAMPLE_RATE, channels=1, dtype="float32",
            device=device, callback=_callback
        ):
            sd.sleep(int(duration_s * 1000))
    except Exception as e:
        LOGGER.exception("Ошибка доступа к аудиоустройству")
        raise SpeechApiError(ErrorCode.AUDIO_DEVICE, f"Ошибка аудиоустройства: {e}")

    LOGGER.info("Запись завершена, начинается распознавание...")

    # Собираем записанные фрагменты из очереди (фреймы из callback).
    audio_chunks = []
    while not q.empty():
        audio_chunks.append(q.get())

    if not audio_chunks:
        # Принципиальный момент: как и в описании — при тишине/пустоте язык строго 'und'.
        return Transcription(text="", words=[], language="und", duration_s=0.0, raw=None)

    audio_data = np.concatenate(audio_chunks, axis=0)  # shape: (N, 1)
    audio_data = _to_mono_1d_float32(audio_data)

    if audio_data.size == 0:
        return Transcription(text="", words=[], language="und", duration_s=duration_s, raw=None)

    # Оценка пикового уровня сигнала
    max_amplitude = float(np.max(np.abs(audio_data)))
    if max_amplitude < 1e-4:
        LOGGER.warning("Записанный звук слишком тихий, распознавание может быть неточным или пустым.")
        return Transcription(text="", words=[], language="und", duration_s=duration_s, raw=None)

    # Нормализация до пика 1.0
    audio_data = audio_data / max_amplitude
    LOGGER.info("Аудио нормализовано (пиковая громкость: %.4f).", max_amplitude)

    return recognizer.transcribe(
        audio_data, enable_words=enable_words, language=language, beam_size=beam_size, **kwargs
    )


def list_audio_devices() -> List[dict]:
    """
    Возвращает список доступных аудиоустройств (как dict от sounddevice).

    Принципиальный момент:
      - Нормализуем представление устройств к list[dict], чтобы downstream-код мог
        стабильно использовать .get(...) независимо от версии sounddevice/платформы.
    """
    if sd is None:
        raise SpeechApiError(ErrorCode.AUDIO_DEVICE, "Модуль sounddevice не установлен")
    try:
        devices = sd.query_devices()  # type: ignore
        # sd.query_devices может возвращать список структур/маппингов — приводим к "чистым" dict.
        return [dict(d) for d in devices]  # type: ignore
    except Exception as e:
        LOGGER.exception("Ошибка при запросе устройств")
        raise SpeechApiError(ErrorCode.AUDIO_DEVICE, f"Ошибка устройств: {e}")


def version() -> str:
    """Возвращает строку версии API (для быстрой диагностики и логов)."""
    return "speech_api/2.2 (Faster-Whisper)"