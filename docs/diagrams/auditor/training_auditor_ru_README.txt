Материалы для аудиторской презентации (обучение моделей)
======================================================

Синтезированные журналы обучения (не из реального запуска):
  docs/training/scorer_v2_training_logs.png, .txt  — render_scorer_v2_training_logs.py
  docs/training/encoder_pretrain_logs.png, .txt    — render_encoder_pretrain_logs.py

Упрощённые схемы (только русские подписи, draw.io):
  docs/diagrams/auditor/training_encoder_pretrain_auditor_ru.drawio
  docs/diagrams/auditor/training_scorer_auditor_ru.drawio
  docs/diagrams/auditor/dataset_structure_auditor_ru.drawio

Полное описание процесса обучения оценщика: docs/training/scorer_v2_training_process_ru.md

Пересборка картинок журналов:
  .venv\Scripts\python docs\training\render_scorer_v2_training_logs.py
  .venv\Scripts\python docs\training\render_encoder_pretrain_logs.py
