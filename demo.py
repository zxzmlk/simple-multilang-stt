# -*- coding: utf-8 -*-
"""
demo.py — CLI-утилита для демонстрации работы speech_api (Faster-Whisper офлайн).

Принципиальные моменты:
  - Трёхступенчатый выбор модели: явная (--model) > локальная папка > скачивание DEFAULT_MODEL_NAME.
  - Логи: настраиваем корневой логгер и дополнительно принудительно выставляем уровень для логгера "speech_api",
    чтобы сообщения библиотеки были видны в консоли (они уже пишутся в speech_api.log).
  - Язык: по умолчанию — авто-детект; при явном указании — авто-детект отключается.
  - Формат слов: поддержка краткого списка и детальных таймкодов с вероятностями.
  - Коды возврата:
      0 — успех
      1 — неверные аргументы/общая ошибка
      2 — ошибка уровня SpeechApiError
"""

import argparse
import os
import sys
import time
import logging
from typing import Optional

from speech_api import (
    init_recognizer,
    free_recognizer,
    transcribe_wav,
    transcribe_microphone,
    list_audio_devices,
    SpeechApiError,
)

# Имя модели, которая будет скачана, если локальная не найдена.
DEFAULT_MODEL_NAME = "small"
# Путь по умолчанию, где программа будет искать локальную модель.
DEFAULT_LOCAL_MODEL_PATH = os.path.join("models", "faster-whisper-small")


def setup_logging(level_name: str = "info") -> None:
    """
    Настраивает логирование в консоль и пробрасывает логи из speech_api.

    Принципиальный момент:
      - basicConfig настраивает корневой логгер и формат для всей программы.
      - Явно выставляем уровень для логгера "speech_api", чтобы его сообщения отображались в консоли
        вместе с логами приложения. В speech_api уже настроен файл с ротацией, здесь добавляем консоль.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    api_logger = logging.getLogger("speech_api")
    api_logger.setLevel(level)


def _print_words(tr, mode: str, enable_words: bool) -> None:
    """
    Печать списка слов в разных форматах.

    Принципиальный момент:
      - Разные уровни детализации: только слова или подробные таймкоды с вероятностями.
      - Если слов нет (тишина/низкая уверенность), выводим информативное сообщение.
    """
    if not enable_words:
        print("\nСлова: отключены флагом --no-words")
        return

    if not tr.words:
        print("\nСлова: нет данных (возможно, тишина или низкая уверенность)")
        return

    if mode == "words":
        print(f"\nСлова ({len(tr.words)}):")
        print(" ".join(w.word for w in tr.words))
    elif mode == "times":
        print(f"\nСлова с таймкодами ({len(tr.words)}):")
        for w in tr.words:
            print(f"[{w.start:7.2f} .. {w.end:7.2f}] {w.word} (p={w.conf:.2f})")
    elif mode == "both":
        print(f"\nСлова ({len(tr.words)}):")
        print(" ".join(w.word for w in tr.words))
        print(f"\nСлова с таймкодами ({len(tr.words)}):")
        for w in tr.words:
            print(f"[{w.start:7.2f} .. {w.end:7.2f}] {w.word} (p={w.conf:.2f})")
    else:
        print(f"\n[Предупреждение] Неизвестный режим words-mode: {mode}")


def _normalize_lang(s: Optional[str]) -> Optional[str]:
    """
    Приводит значение аргумента --lang к None для авто-детекта,
    иначе возвращает исходную строку (без изменения регистра).

    Принципиальный момент:
      - Возврат None — это "включить авто-детект" на уровне движка.
      - Любое иное значение трактуется как фиксированный язык, авто-детект отключается.
    """
    if s is None:
        return None
    val = s.strip().lower()
    return None if val in ("none", "auto", "detect", "") else s


def main():
    parser = argparse.ArgumentParser(description="Демо русской речи (Faster-Whisper, офлайн)")
    parser.add_argument("--wav", type=str, help="Путь к аудиофайлу (WAV, MP3, и др.)")
    parser.add_argument("--mic", type=float, help="Длительность записи с микрофона, сек (например, 7)")
    # Убираем default, чтобы можно было отличить отсутствие аргумента от явного указания.
    parser.add_argument(
        "--model",
        type=str,
        help=f"Имя или путь к модели Whisper. По умолчанию: сначала ищется '{DEFAULT_LOCAL_MODEL_PATH}', затем скачивается '{DEFAULT_MODEL_NAME}'."
    )
    parser.add_argument("--list-devices", action="store_true", help="Показать устройства записи")

    # Параметры модели/распознавания
    parser.add_argument("--device", type=str, default="cpu", help="Устройство вычислений: cpu или cuda")
    parser.add_argument("--compute-type", type=str, default="int8", help="Тип вычислений (cpu:int8, cuda:float16)")
    parser.add_argument("--lang", type=str, default="none", help="Код языка (например: ru, en). 'none'/'auto'/'detect' — авто-детект")
    parser.add_argument("--beam-size", type=int, default=5, help="Ширина бима (качество/скорость)")
    parser.add_argument("--no-words", action="store_true", help="Отключить словарные таймкоды")

    # Режим вывода слов
    parser.add_argument(
        "--words-mode",
        choices=["words", "times", "both"],
        default="words",
        help="Режим вывода слов: words — только слова; times — таймкоды и вероятности; both — и то, и другое"
    )

    # Уровень логирования
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
        help="Уровень логирования в консоль (по умолчанию info)"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("demo")

    rec = None
    try:
        if args.list_devices:
            log.info("Запрашиваем список аудиоустройств...")
            try:
                devices = list_audio_devices()
                input_devices = 0
                for i, d in enumerate(devices):
                    name = d.get("name")
                    mic_in = d.get("max_input_channels")
                    if mic_in and mic_in > 0:
                        print(f"[{i}] {name} (max_input_channels={mic_in})")
                        input_devices += 1
                log.info("Найдено входных устройств: %d", input_devices)
            except SpeechApiError as e:
                log.error("Ошибка при запросе устройств: %s", e)
                print(f"Ошибка: {e}")
            return 0

        # Выбор модели: явная > локальная папка > скачивание.
        # Принципиальный момент: последовательность выбора гарантирует офлайн-режим при наличии локальной модели.
        if args.model:
            model_to_use = args.model
            log.info("Используем модель, указанную вручную: %s", model_to_use)
        elif os.path.exists(DEFAULT_LOCAL_MODEL_PATH):
            model_to_use = DEFAULT_LOCAL_MODEL_PATH
            log.info("Найдена локальная модель в папке проекта: %s", model_to_use)
        else:
            model_to_use = DEFAULT_MODEL_NAME
            log.warning(
                "Локальная модель в '%s' не найдена. Будет скачана модель '%s' (требуется сеть).",
                DEFAULT_LOCAL_MODEL_PATH, model_to_use
            )

        log.info(
            "Загрузка модели '%s' (device=%s, compute_type=%s)...",
            model_to_use, args.device, args.compute_type
        )
        t0 = time.perf_counter()
        rec = init_recognizer(model_to_use, device=args.device, compute_type=args.compute_type)
        log.info("Модель загружена за %.2f с.", time.perf_counter() - t0)

        language = _normalize_lang(args.lang)
        if language is None:
            log.info("Язык не задан — включён авто-детект.")
        else:
            log.info("Язык фиксирован: %s (авто-детект отключён).", language)

        enable_words = not args.no_words

        # Принципиальный момент:
        # - Используем проверку "is not None", чтобы явно заданное пустое значение или 0.0
        #   не трактовались как отсутствие аргумента.
        if args.wav is not None:
            wav_path = args.wav
            if not os.path.isfile(wav_path):
                log.error("Файл не найден: %s", wav_path)
                print("Файл не найден:", wav_path)
                return 1

            try:
                size_mb = os.path.getsize(wav_path) / (1024 * 1024)
                log.info("Распознавание файла: %s (%.2f MB)", wav_path, size_mb)
            except Exception:
                log.info("Распознавание файла: %s", wav_path)

            t1 = time.perf_counter()
            tr = transcribe_wav(
                rec,
                wav_path,
                enable_words=enable_words,
                language=language,    # None => авто-детект
                beam_size=args.beam_size
            )
            elapsed = time.perf_counter() - t1

            if language is None:
                log.info("Файл распознан за %.2f с. Язык определён: %s. Слов: %d",
                         elapsed, tr.language, len(tr.words))
            else:
                log.info("Файл распознан за %.2f с. Язык был задан: %s. Слов: %d",
                         elapsed, language, len(tr.words))

            print("\n--- Результат распознавания файла ---")
            if language is None:
                lang_prob = getattr(tr.raw, "language_probability", None)
                suffix = f" (p={lang_prob:.2f})" if lang_prob is not None else ""
                print(f"Язык (определён): {tr.language}{suffix}")
            else:
                print(f"Язык (задан): {language}")
            print(f"Длительность аудио: {tr.duration_s:.2f} c")
            print("Текст:")
            print(tr.text)
            _print_words(tr, args.words_mode, enable_words)

        elif args.mic is not None:
            dur = float(args.mic)
            log.info("Запись с микрофона на %.2f с и распознавание...", dur)
            t1 = time.perf_counter()
            tr = transcribe_microphone(
                rec,
                duration_s=dur,
                enable_words=enable_words,
                language=language,    # None => авто-детект
                beam_size=args.beam_size
            )
            elapsed = time.perf_counter() - t1

            if language is None:
                log.info("Распознавание записи с микрофона завершено за %.2f с. Язык определён: %s. Слов: %d",
                         elapsed, tr.language, len(tr.words))
            else:
                log.info("Распознавание записи с микрофона завершено за %.2f с. Язык был задан: %s. Слов: %d",
                         elapsed, language, len(tr.words))

            print("\n--- Результат распознавания с микрофона ---")
            if language is None:
                lang_prob = getattr(tr.raw, "language_probability", None)
                suffix = f" (p={lang_prob:.2f})" if lang_prob is not None else ""
                print(f"Язык (определён): {tr.language}{suffix}")
            else:
                print(f"Язык (задан): {language}")
            print(f"Длительность записи: {tr.duration_s:.2f} c")
            print("Текст:")
            print(tr.text)
            _print_words(tr, args.words_mode, enable_words)

        else:
            # Нет источника аудио — подсказка по использованию.
            parser.print_help()
            return 1

        return 0

    except SpeechApiError as e:
        # Единая обработка ошибок уровня API — печатаем код и сообщение.
        logging.getLogger("demo").exception("Исключение уровня API: %s", e)
        print(f"\n[ОШИБКА] ({e.code.name}) {e}")
        return 2
    except Exception as e:
        # Непредвиденные ошибки — возвращаем код 1.
        logging.getLogger("demo").exception("Неожиданная ошибка: %s", e)
        print(f"\n[НЕОЖИДАННАЯ ОШИБКА] {e}")
        return 1
    finally:
        # Освобождение ресурсов модели гарантировано даже при исключениях.
        if rec:
            log.info("Освобождение ресурсов модели...")
            free_recognizer(rec)
            log.info("Готово.")


if __name__ == "__main__":
    sys.exit(main())