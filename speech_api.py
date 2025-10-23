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
from queue import Queue  # Потокобезопасная очередь для сбора аудиоданных с микрофона.
from typing import List, Optional, Union

import numpy as np

# --- ОПЦИОНАЛЬНЫЕ ЗАВИСИМОСТИ ---
# Основная функция (распознавание массива) будет доступна всегда, а дополнительные
# (микрофон, быстрое чтение файлов) — только при наличии соответствующих библиотек.

# sounddevice — для записи с микрофона.
try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None

# soundfile — для локального чтения аудиофайлов (быстрее, чем ffmpeg для WAV/FLAC).
try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None  # type: ignore

# librosa — для локальной передискретизации (изменения частоты дискретизации аудио).
try:
    import librosa  # type: ignore
except Exception:
    librosa = None  # type: ignore
try:
    from faster_whisper import WhisperModel
except Exception as e:
    raise RuntimeError(
        "Faster-Whisper не установлен. Выполните: pip install faster-whisper\nОригинальная ошибка: %s" % e
    )

# ----------------------- ЛОГИРОВАНИЕ -----------------------
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
# Модель Whisper была обучена на аудио с частотой 16000 Гц.
# Любое аудио должно быть приведено к нему для лучшего качества.
WHISPER_SAMPLE_RATE = 16000

# ---------------------- КЛАССЫ ОШИБОК И ДАННЫХ --------------------------

class ErrorCode(IntEnum):
    """Перечисление для кодов ошибок. Использование Enum вместо "магических чисел" (1, 2, 3...)
    делает код более читаемым и надежным."""
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
    Собственный класс исключений для API.
    Это позволяет вызывающему коду (например, demo.py) ловить конкретно ошибки нашего API
    и отличать их от других исключений Python.
    """
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code
@dataclass
class Word:
    """Структура для хранения информации об одном распознанном слове."""
    start: float  # Время начала слова в секундах.
    end: float    # Время конца слова в секундах.
    word: str     # Текст слова.
    conf: float   # Уверенность модели в этом слове (от 0.0 до 1.0).


@dataclass
class Transcription:
    """Структура для хранения полного результата распознавания."""
    text: str
    words: List[Word]
    language: str
    duration_s: float
    raw: object  # Сырой объект `info` от Faster-Whisper для доступа к доп. данным (например, вероятности языка).


class Recognizer:
    """
    Класс-обертка над моделью Faster-Whisper. Он инкапсулирует логику
    загрузки модели и выполнения распознавания.
    """
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Конструктор загружает модель в память. Это может быть долгой операцией.
        - model_name_or_path: "small", "medium" или путь к папке с моделью.
        - device: "cpu" или "cuda".
        - compute_type: тип вычислений ("int8" для CPU, "float16" для GPU).
        """
        try:
            # Основной вызов для загрузки модели из библиотеки faster-whisper.
            self._model = WhisperModel(model_name_or_path, device=device, compute_type=compute_type)
            LOGGER.info(
                "Модель Whisper загружена: %s (device=%s, compute_type=%s)",
                model_name_or_path, device, compute_type
            )
        except Exception as e:
            # Если что-то пошло не так (модель не найдена, нет CUDA),
            # логируем ошибку и выбрасываем наше собственное исключение SpeechApiError.
            LOGGER.exception("Ошибка загрузки модели Whisper")
            raise SpeechApiError(ErrorCode.MODEL_NOT_FOUND, f"Ошибка загрузки модели '{model_name_or_path}': {e}")

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        enable_words: bool = True,
        language: Optional[str] = None,   # Если None, включается авто-определение языка.
        beam_size: int = 5,
        **kwargs
    ) -> Transcription:
        """
        Основной метод, выполняющий распознавание.
        """
        try:
            # Вызов метода `transcribe` из faster-whisper. Он возвращает итератор сегментов и объект с информацией.
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                word_timestamps=enable_words,
                **kwargs
            )

            full_text = ""
            all_words: List[Word] = []
            # Результат может состоять из нескольких сегментов (частей). Их нужно объединить.
            for segment in segments:
                full_text += segment.text + " "
                # Безопасно проверяем, есть ли у сегмента атрибут 'words'.
                if enable_words and getattr(segment, "words", None):
                    for w in segment.words:
                        # Преобразуем данные из формата faster-whisper в наш собственный класс Word.
                        all_words.append(
                            Word(
                                start=float(w.start),
                                end=float(w.end),
                                word=str(w.word).strip(),
                                conf=float(getattr(w, "probability", 0.0)), # Уверенность может отсутствовать.
                            )
                        )

            # Упаковываем все результаты в наш стандартизированный объект Transcription.
            return Transcription(
                text=full_text.strip(),
                words=all_words,
                language=str(getattr(info, "language", language or "")),
                duration_s=float(getattr(info, "duration", 0.0)),
                raw=info,
            )
        except Exception as e:
            # Любую ошибку во время распознавания оборачиваем в SpeechApiError.
            LOGGER.exception("Ошибка во время распознавания Whisper")
            raise SpeechApiError(ErrorCode.INTERNAL, f"Ошибка распознавания: {e}")

    def close(self) -> None:
        """Освобождает ресурсы модели."""
        # У Faster-Whisper нет явного метода `close()`.
        # Присвоение None удаляет ссылку на объект модели, что позволяет сборщику мусора Python
        # освободить память, которую она занимала (особенно важно для GPU).
        self._model = None


# ------------------------ ПУБЛИЧНЫЕ API ФУНКЦИИ -----------------------
# Эти функции предоставляют более простой, процедурный интерфейс к возможностям класса Recognizer.

def init_recognizer(
    model_name_or_path: str,
    device: str = "cpu",
    compute_type: str = "int8",
    **kwargs
) -> Recognizer:
    """Функция-фабрика для создания и инициализации распознавателя."""
    return Recognizer(model_name_or_path=model_name_or_path, device=device, compute_type=compute_type)


def free_recognizer(recognizer: Optional[Recognizer]) -> None:
    """Освобождает ресурсы распознавателя, вызывая его метод close()."""
    if recognizer:
        recognizer.close()


def _to_mono_1d_float32(audio_data: np.ndarray) -> np.ndarray:
    """
    Вспомогательная функция для приведения аудио к формату, который ожидает модель.
    Модель Whisper требует на вход одномерный (моно) массив чисел с плавающей точкой (float32).
    """
    # Если массив двумерный, значит, у аудио несколько каналов (например, стерео).
    if audio_data.ndim == 2:
        # Если каналов больше одного (стерео), смешиваем их в один (моно) путем усреднения.
        if audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        else:
            # Если форма (N, 1), это моно, но в двумерном массиве. Уплощаем до (N,).
            audio_data = audio_data[:, 0]
    # Гарантируем, что тип данных - float32.
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
    Распознаёт речь из аудиофайла с использованием "умной" стратегии отката (fallback).
    """
    if not os.path.isfile(wav_path):
        raise SpeechApiError(ErrorCode.IO, f"Файл не найден: {wav_path}")

    # Вспомогательная функция для "запасного плана" — распознавания через ffmpeg.
    # Faster-Whisper сам вызовет ffmpeg, если ему передать путь к файлу.
    def _fallback_via_path() -> Transcription:
        LOGGER.info("Чтение файла делегировано Faster-Whisper (ffmpeg). Путь: %s", wav_path)
        return recognizer.transcribe(
            wav_path, enable_words=enable_words, language=language, beam_size=beam_size, **kwargs
        )

    # План А: если soundfile не установлен, сразу переходим к плану Б (ffmpeg).
    if sf is None:
        return _fallback_via_path()

    try:
        # Пытаемся прочитать файл локально через soundfile. Это быстрее, чем запускать ffmpeg.
        audio_data, samplerate = sf.read(wav_path, dtype="float32")  # type: ignore
        audio_data = _to_mono_1d_float32(audio_data)

        # Идеальный случай: частота дискретизации уже 16 кГц.
        if samplerate == WHISPER_SAMPLE_RATE:
            return recognizer.transcribe(
                audio_data, enable_words=enable_words, language=language, beam_size=beam_size, **kwargs
            )

        # Случай, когда нужна передискретизация. Проверяем, есть ли librosa.
        if librosa is not None:
            LOGGER.info(
                "Частота дискретизации файла %d Гц. Передискретизация в %d Гц через librosa...",
                samplerate, WHISPER_SAMPLE_RATE
            )
            # Выполняем передискретизацию локально.
            audio_data = librosa.resample(y=audio_data, orig_sr=samplerate, target_sr=WHISPER_SAMPLE_RATE)  # type: ignore
            LOGGER.info("Передискретизация завершена.")
            audio_data = _to_mono_1d_float32(audio_data)
            return recognizer.transcribe(
                audio_data, enable_words=enable_words, language=language, beam_size=beam_size, **kwargs
            )

        # Если librosa нет, мы не можем передискретизировать локально. Переходим к плану Б.
        return _fallback_via_path()

    except SpeechApiError:
        # Если ошибка уже нашего типа, просто пробрасываем ее дальше.
        raise
    except Exception as e:
        # Если локальное чтение не удалось, не сдаемся, а пробуем план Б.
        LOGGER.warning("Локальное чтение через soundfile не удалось (%s). Пытаемся через ffmpeg...", e)
        try:
            return _fallback_via_path()
        except Exception as e2:
            # Если и план Б не сработал, вот теперь выбрасываем ошибку.
            # `from e` — это "chained exception", оно сохраняет информацию об исходной ошибке `e`,
            # что очень помогает при отладке.
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
    Записывает аудио с микрофона и распознаёт его.
    """
    if sd is None:
        # Если sounddevice не установлен, работать с микрофоном невозможно.
        raise SpeechApiError(ErrorCode.AUDIO_DEVICE, "Модуль sounddevice не установлен")

    # Очередь для сбора аудио-фрагментов. Запись происходит в отдельном потоке,
    # и callback-функция будет складывать данные в эту очередь.
    q: Queue = Queue()

    # Эта функция будет вызываться библиотекой sounddevice для каждого нового блока аудиоданных.
    def _callback(indata, frames, time_info, status):
        if status:
            LOGGER.warning("Статус аудиопотока: %s", status)
        q.put(indata.copy()) # Кладем копию данных в очередь.

    LOGGER.info("Начало записи с микрофона: %.2f сек", duration_s)
    try:
        # `sd.InputStream` — это контекстный менеджер, который открывает аудиопоток и автоматически закрывает его.
        # Мы сразу просим записывать в формате, нужном модели (16кГц, 1 канал, float32), чтобы избежать конвертации.
        with sd.InputStream(
            samplerate=WHISPER_SAMPLE_RATE, channels=1, dtype="float32",
            device=device, callback=_callback
        ):
            sd.sleep(int(duration_s * 1000)) # Ждем указанное время, пока идет запись.
    except Exception as e:
        LOGGER.exception("Ошибка доступа к аудиоустройству")
        raise SpeechApiError(ErrorCode.AUDIO_DEVICE, f"Ошибка аудиоустройства: {e}")

    LOGGER.info("Запись завершена, начинается распознавание...")

    # Собираем все записанные фрагменты из очереди в один список.
    audio_chunks = []
    while not q.empty():
        audio_chunks.append(q.get())

    # Если ничего не записалось (например, микрофон отключен), возвращаем пустой результат.
    if not audio_chunks:
        return Transcription(text="", words=[], language="und", duration_s=0.0, raw=None)

    # Объединяем все фрагменты в один большой NumPy массив.
    audio_data = np.concatenate(audio_chunks, axis=0)
    audio_data = _to_mono_1d_float32(audio_data)

    if audio_data.size == 0:
        return Transcription(text="", words=[], language="und", duration_s=duration_s, raw=None)

    # "Фильтр тишины": проверяем максимальную громкость.
    max_amplitude = float(np.max(np.abs(audio_data)))
    # Если звук очень тихий, скорее всего, это тишина. Не тратим время на распознавание.
    if max_amplitude < 1e-4:
        LOGGER.warning("Записанный звук слишком тихий, распознавание может быть неточным или пустым.")
        return Transcription(text="", words=[], language="und", duration_s=duration_s, raw=None)

    # Нормализация: приводим громкость к стандартному уровню (пик = 1.0).
    # Это делает модель более устойчивой к разной громкости записи.
    audio_data = audio_data / max_amplitude
    LOGGER.info("Аудио нормализовано (пиковая громкость: %.4f).", max_amplitude)

    # Отправляем подготовленный массив на распознавание.
    return recognizer.transcribe(
        audio_data, enable_words=enable_words, language=language, beam_size=beam_size, **kwargs
    )


def list_audio_devices() -> List[dict]:
    """
    Возвращает список доступных аудиоустройств.
    """
    if sd is None:
        raise SpeechApiError(ErrorCode.AUDIO_DEVICE, "Модуль sounddevice не установлен")
    try:
        devices = sd.query_devices()  # type: ignore
        # `sd.query_devices` может возвращать не совсем dict.
        # Преобразуем каждый элемент в стандартный dict, чтобы сделать API более предсказуемым.
        return [dict(d) for d in devices]  # type: ignore
    except Exception as e:
        LOGGER.exception("Ошибка при запросе устройств")
        raise SpeechApiError(ErrorCode.AUDIO_DEVICE, f"Ошибка устройств: {e}")


def version() -> str:
    """Возвращает строку версии API для диагностики."""
    return "speech_api/2.2 (Faster-Whisper)"