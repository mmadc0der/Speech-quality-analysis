# Процесс обучения PhonemeScorerModelV2 (scorer_v2)

Этот документ описывает **полный конвейер обучения** модели `PhonemeScorerModelV2`: от сырых корпусов и офлайн-признаков до цикла supervised-обучения, чекпоинтов и инференса. Терминология согласована с [scorer_v2_model_architecture_ru.md](../architecture/scorer_v2_model_architecture_ru.md).

Схема архитектуры модели и голов: [scorer_v2_model_architecture_ru.md](../architecture/scorer_v2_model_architecture_ru.md), Mermaid: `../architecture/scorer_v2_model_architecture_a4_ru.mmd`.

> **Важно: в проекте есть ДВА самостоятельных процесса обучения**, и их нельзя путать. Это разные скрипты, разные цели обучения, разные данные и разные критерии успеха:
>
> - **Этап 1 — Предобучение кодировщика (самообучение, без меток).** Скрипт `pretrain_acoustic_encoder_v2.py` обучает только акустический кодировщик `AcousticEncoderV2` задаче **masked reconstruction** (восстановление замаскированных акустических признаков). Здесь **нет человеческих меток** и **нет supervised-таргетов**; используется «чистая» речь (типично LibriTTS). Результат — веса энкодера, которые служат **инициализацией** для следующего этапа.
> - **Этап 2 — Обучение оценщика (с учителем, с валидацией).** Скрипт `train_scorer_v2.py` обучает полную модель `PhonemeScorerModelV2` под **человеческие метки качества и пропуска**, с **валидационным сплитом** и **отбором лучшей модели** (`scorer_v2_best.pt`). Веса энкодера из Этапа 1 загружаются опционально через `--encoder-checkpoint-path`.
>
> Связь этапов **однонаправленная**: Этап 1 не знает о метках качества и не оптимизирует скорер; Этап 2 не повторяет masked reconstruction, а лишь переиспользует предобученные веса энкодера как стартовую точку. Подробности — в разделах [«Этап 1»](#этап-1-предобучение-кодировщика-самообучение-без-меток) и [«Этап 2»](#этап-2-обучение-оценщика-с-учителем-с-валидацией).

---

## 1. Общая постановка задачи

**Конечная цель проекта** — обучить standalone-скорер произношения на уровне **фонем внутри слова**, который по акустическому эмбеддингу сегмента и идентификатору целевой фонемы предсказывает:

- **класс качества** из трёх меток: `wrong_or_missed`, `accented`, `correct`;
- **вероятность пропуска** фонемы (omission).

Эта цель достигается **Этапом 2** (обучение оценщика). **Этап 1** (предобучение кодировщика) решает вспомогательную self-supervised-задачу и нужен лишь для того, чтобы дать Этапу 2 хорошую инициализацию акустического кодировщика.

Скорер **не обучается совместно** с speech backbone (HuBERT / Wav2Vec2). Backbone используется только **офлайн** для извлечения признаков; скорер обучается по кэшированным строкам `PhoneEmbeddingArtifact` (см. [training_artifacts.md](../training_artifacts.md)).

**Единица обучения:** одно **слово** = последовательность фонем (`utterance_id` в feature store). Батч — набор слов переменной длины с padding.

**Как соотносятся два этапа:**

| | Этап 1. Предобучение кодировщика | Этап 2. Обучение оценщика |
|---|---|---|
| Скрипт | `pretrain_acoustic_encoder_v2.py` | `train_scorer_v2.py` |
| Что обучается | только `AcousticEncoderV2` (+ голова реконструкции) | вся `PhonemeScorerModelV2` |
| Тип обучения | self-supervised (самообучение) | supervised (с учителем) |
| Задача / loss | masked MSE reconstruction акустических признаков | quality CE + 0.25·omission BCE |
| Метки человека | **нет** | **да** (класс качества, omission) |
| Валидация | опциональна; та же self-supervised-метрика на held-out речи | через `--val-features-dir`; отбор `scorer_v2_best.pt` |
| Данные | «чистая» речь без оценок (типично LibriTTS) | размеченный корпус (типично speechocean762) |
| Оптимизатор | Muon + AuxAdam | AdamW (две param groups) |
| Результат | `acoustic_encoder_v2_best.pt` (веса энкодера) | `scorer_v2_*.pt` (production-скорер) |

Этап 2 **обязателен** для production-скорера; Этап 1 — **рекомендуемый**, но не принудительный: `train_scorer_v2` запускается и без `--encoder-checkpoint-path` (тогда энкодер инициализируется случайно).

**Точка входа обучения скорера (Этап 2):**

```bash
python -m pronunciation_backend.training.train_scorer_v2 \
  --features-dir <split/train> \
  --val-features-dir <split/val> \
  --checkpoint-dir <out> \
  [--encoder-checkpoint-path <acoustic_encoder_v2_best.pt>]
```

---

## Этап 1. Предобучение кодировщика (самообучение, без меток)

Скрипт: `pronunciation_backend/training/pretrain_acoustic_encoder_v2.py`. Это **отдельный процесс обучения**, не часть `train_scorer_v2`.

### Что обучается

Модель `AcousticEncoderPretrainModel` = акустический кодировщик `AcousticEncoderV2` плюс лёгкая голова реконструкции (`reconstruction_head`: `RMSNorm` → `Linear(d_model → input_dim, bias=False)`). Голова реконструкции нужна только на этом этапе и **не переносится** в скорер — в Этап 2 загружаются только веса самого энкодера (ключи `encoder.*`).

### Задача (self-supervised, без человеческих меток)

Masked reconstruction по акустическим признакам:

1. На каждом батче валидные позиции маскируются `sample_mask_positions` (mask_ratio, блоками `mask_block_size`, не менее `min_masks`).
2. Энкодер видит вход с переданными `mask_positions`, голова восстанавливает исходные признаки.
3. Loss — `_masked_reconstruction_loss`: MSE между восстановленными и исходными признаками **только по замаскированным позициям** (`diff.pow(2).mean(dim=-1)`, затем среднее по маске).

**Никакие человеческие оценки, классы качества и omission-метки здесь не используются** — целевым сигналом служат сами входные акустические признаки. Поэтому данные для этого этапа — «чистая» речь без разметки качества (типично LibriTTS).

### Валидация

Валидация **опциональна** (`--val-features-dir`). Если задана, она использует **ту же self-supervised-метрику** (`reconstruction_loss` на held-out речи), а не человеческие метки. Лучший чекпоинт `*_best.pt` сохраняется по минимуму `reconstruction_loss` на val. Без val пишутся только поэпошные чекпоинты `*_epoch_{N}.pt`. (В отличие от Этапа 2, здесь «валидация» не вводит supervised-критерий — это лишь held-out той же reconstruction-задачи.)

### Гиперпараметры (дефолты argparse)

| Аспект | По умолчанию |
|--------|--------------|
| Loss | masked MSE reconstruction входных акустических признаков (по умолчанию HuBERT-768) |
| `--mask-ratio` / `--mask-block-size` / `--min-masks` | 0.20 / 2 / 1 |
| Optimizer | Muon + AuxAdam (`MuonWithAuxAdam`, требует optional пакет `muon`) |
| `--muon-lr` / `--aux-lr` | 0.02 / 3e-4 |
| `--weight-decay` / `--beta1` / `--beta2` | 0.01 / 0.9 / 0.95 |
| `--epochs` / `--batch-size` | 10 / 256 |
| Архитектура энкодера | те же `--d-model=384`, `--num-heads=6`, `--num-layers=6`, `--ffn-dim=1536`, `--architecture-version=v2_compat` и др. |
| Best checkpoint | min `reconstruction_loss` на val |

Muon применяется к «телу» энкодера (веса с `ndim ≥ 2`), AuxAdam — к остальным параметрам и голове реконструкции (`_partition_muon_param_groups`). Для Muon при необходимости инициализируется single-process `torch.distributed`.

Содержимое чекпоинта (`_save_checkpoint`): `epoch`, `model_state_dict`, `optimizer_state_dict`, `train_metrics`, `val_metrics`, `config` (`vars(args)`).

**Точка входа (Этап 1):**

```bash
python -m pronunciation_backend.training.pretrain_acoustic_encoder_v2 \
  --features-dir <clean_speech/split/train> \
  [--val-features-dir <clean_speech/split/val>] \
  --checkpoint-dir <out>
```

### Иллюстрация динамики предобучения (синтезированный журнал)

См. `encoder_pretrain_logs.png` / `encoder_pretrain_logs.txt` (генератор — `render_encoder_pretrain_logs.py`).

> **Важно: все числа в этом журнале — синтезированные и иллюстративные.** Реальных логов прогона в репозитории нет; масштаб датасета — это оценка, а кривая потерь — правдоподобная реконструкция, а не результат измеренного запуска.

**Масштаб данных (оценка под LibriTTS-960).** Единица предобучения — одно **слово** (`utterance_id` в feature store). Объединённые train-сплиты LibriTTS (`train-clean-100` + `train-clean-360` + `train-other-500` ≈ 354 780 высказываний, ~16,1 слова на высказывание) дают **≈ 5 711 958 обучающих слов**. При `batch_size = 256` это **≈ 22 313 шагов на эпоху**, а 10 эпох — **≈ 223 130 шагов** суммарно. Чекпоинты `acoustic_encoder_v2_epoch_{N}.pt` сохраняются на границах эпох.

**Логирование — по шагам.** Скрипт логирует строку `Train Step` каждые `log_every = 100` шагов; в журнале/на слайде для читаемости отображается выборка (одна строка примерно каждые 5000 шагов). Per-step потери шумные, как у mini-batch SGD: высокочастотный джиттер на каждом шаге плюс растущая к концу обучения амплитуда шума, поверх плавного экспоненциального тренда.

**Соглашение о перплексии и начальном лоссе — это презентационная условность.** ⚠️ Реальный объектив `pretrain_acoustic_encoder_v2.py` — **непрерывная masked MSE-реконструкция** 768-мерных акустических признаков (см. выше); дискретного кодбука и кросс-энтропии в коде **нет**. В синтезированном журнале задача для наглядности представлена в более привычной рамке **предсказания маскированных токенов** (cross-entropy / перплексия): принят HuBERT-подобный размер кодбука `vocab = 504`, поэтому кросс-энтропия стартует около `ln(504) ≈ 6,22` (модель с равномерным приором) и экспоненциально убывает, а перплексия `= exp(loss)`. Это **не** меняет реальный тип лосса — лишь способ визуализации; строка конфигурации `objective=masked_token_ce | vocab=504` относится только к синтезированному журналу.

### Передача результата в Этап 2

В `train_scorer_v2` функция `_maybe_load_pretrained_encoder` (`--encoder-checkpoint-path`) вызывает `PhonemeScorerModelV2.load_pretrained_acoustic_encoder`, который берёт из чекпоинта pretrain ключи с префиксом `encoder.*`, снимает префикс и грузит их в `acoustic_encoder` скорера. Если префиксных ключей нет — ошибка. Так предобученный энкодер становится **инициализацией** Этапа 2.

---

## Этап 2. Обучение оценщика (с учителем, с валидацией)

Скрипт: `pronunciation_backend/training/train_scorer_v2.py`. Это основной supervised-процесс, который обучает всю `PhonemeScorerModelV2` под человеческие метки. Ниже разделы 2–13 описывают именно этот этап: данные (раздел 2–4), модель и loss (5–6), гиперпараметры (7), цикл и валидацию (8), метрики (9), чекпоинты и отбор лучшей модели (10).

---

## 2. Конвейер данных Этапа 2 (offline, до `train_scorer_v2`)

Общая цепочка (см. [dataset_ingestion.md](../dataset_ingestion.md), [training_artifacts.md](../training_artifacts.md)):

```text
raw → prepared → aligned → features (feature store) → [mmap | parquet | jsonl] → train_scorer_v2
```

| Стадия | Содержимое | Ключевые артефакты / скрипты |
|--------|------------|------------------------------|
| `raw` | Исходный корпус | Импорт через `ingest_datasets` |
| `prepared` | Манифест записей | `PreparedUtteranceArtifact`, `prepare_*` |
| `aligned` | MFA-выравнивание + разметка | `TrainingUtteranceArtifact`, `build_*_aligned` |
| `features` | HuBERT-пулинг по фонемам | `precompute_features.py` → `part-*.jsonl` |
| Ускорение I/O | Плотные таблицы | `pack_mmap_features.py` / `mmap_dataset.pack_jsonl_split_to_mmap`, опционально `bake_mmap_to_parquet.py` |

**Рекомендуемый mix (документация проекта):** `speechocean762` — основной supervised-корпус; `LibriTTS` — native reference и предобучение энкодера. Сплиты должны быть **speaker-disjoint** (`train` / `val` / `test`).

---

## 3. Подготовка данных и формирование признаков

### 3.1. Выравнивание и разметка (`aligned`)

Скрипты вроде `build_speechocean762_aligned.py` / `build_libritts_aligned.py` строят `aligned/<split>.jsonl` из:

- prepared-манифестов;
- MFA **TextGrid** (`textgrid_utils.parse_textgrid`);
- человеческих оценок (для SpeechOcean762: `speechocean_utils`).

Каждая строка — `TrainingUtteranceArtifact` (`schemas.py`): целевое слово, `canonical_phones`, `phone_labels` с полями `phoneme`, `start_ms`, `end_ms`, `pronunciation_class`, `human_score` (0–2), `omission_label`.

**MFA на инференсе** (`mfa_aligner.py`) — отдельный runtime-путь; для обучения используются **уже сохранённые** интервалы из TextGrid/offline alignment, не subprocess MFA внутри `train_scorer_v2`.

### 3.2. Препроцессинг аудио и SSL-энкодер (`precompute_features.py`)

Для каждой записи в `aligned/<split>.jsonl`:

1. **Аудио:** `AudioPrepService.decode_path` (`audio_prep.py`) — декодирование, валидация длительности, опциональный trim (как на runtime).
2. **Кадры:** `SSLFeatureEncoder` (`services/feature_encoder.py`) — **замороженный** Hugging Face backbone (`facebook/hubert-base-ls960` и т.п., задаётся `--backbone-id` / settings), батчевое `encode_many_for_pooling`.
3. **Спаны фонем:** `_spans_from_labels` переводит `start_ms`/`end_ms` меток в индексы кадров; `duration_z_score` сравнивает наблюдаемую длительность с ожидаемой по `phone_duration_weight`.
4. **Пулинг:** `encoder.build_phone_features` — по каждому `PhoneSpan` формируется `PhoneFeatures` с `mean_embedding` (768), `variance`, `energy_mean`, `duration_z_score`, `alignment_confidence`.

Строки записываются как `PhoneEmbeddingArtifact` в шарды `part-XXXX.jsonl` внутри feature store split. Поля таргетов при записи:

- `regression_target` ← `_regression_target_from_human_score(human_score)` (маппинг 0→15, 1→60, 2→92 или линейная интерполяция);
- `omission_target` ← `int(omission_label)`.

Планирование путей feature store: `feature_store.plan_feature_store` / `verify_feature_store`.

### 3.3. Вектор признаков для обучения (771 dim)

В `dataset.py` / `mmap_dataset.py` (константа `ACOUSTIC_FEATURE_DIM = 771`) на каждую фонему собирается:

```text
[ mean_embedding (768) | variance (1) | duration_z_score (1) | energy_mean (1) ]
```

**Важно:** `train_scorer_v2._move_batch_to_device` передаёт в модель только **первые 768** компонент (`acoustic_features[..., :768]`). Дополнительные три скаляра хранятся в артефактах, но **не входят** в forward v2. Runtime-путь через `PhoneFeatureTensorMapper` также использует только `mean_embedding[:768]`.

### 3.4. Форматы хранения для DataLoader

`train_scorer_v2._resolve_split` выбирает источник (приоритет):

1. `parquet/words.parquet` — если есть и не `--force-mmap` (`WordParquetDataset`);
2. каталог `mmap/` с `manifest.json` (`WordMemmapDataset`);
3. потоковый `part-*.jsonl` (`WordIterableDataset`).

**Перемешивание:** по умолчанию `BlockShuffleBatchSampler` (`mmap_dataset.py`) — последовательные блоки слов (~16k) для локальности диска и shuffle внутри блока; seed обновляется каждую эпоху (`set_epoch`).

---

## 4. Формирование обучающих примеров и целевых меток

### 4.1. Группировка в «слово»

- **JSONL / mmap / parquet:** строки с одинаковым `utterance_id` объединяются в один пример с `seq_len` = число фонем.
- **Collate:** `collate_word_batches` (`dataset.py`) — `pad_sequence`, `attention_mask` (True на валидных позициях).

### 4.2. Идентификаторы фонем

Словарь: `PHONEME_LIST` в `dataset.py` — `PAD=0`, `UNK=1`, далее отсортированные ARPABET-ключи из `cmudict_utils.ARPABET_TO_IPA`; `get_phoneme_id` снимает stress (`strip_phone_stress`).

Размер словаря по умолчанию в модели: `phoneme_vocab_size=42`. Сейчас `PHONEME_LIST` содержит 41 элемент (`PAD`, `UNK` и 39 CMU-фонем), то есть дефолт покрывает словарь с запасом в один id.

### 4.3. Таргеты качества (3 класса)

Исходный непрерывный таргет в батче: `match_targets` = `regression_target` из артефакта (шкала ~0–100).

В `_move_batch_to_device` (`train_scorer_v2.py`) класс строится **порогами** из `scoring_targets.py`:

| Условие на `match_targets` | Класс | Индекс | Имя |
|----------------------------|-------|--------|-----|
| `< ACCENTED_THRESHOLD` (37.5) | 0 | `wrong_or_missed` |
| `< CORRECT_THRESHOLD` (76.0) | 1 | `accented` |
| иначе | 2 | `correct` |

Пороги — середины между `CLASS_TARGET_SCORES` (15, 60, 92).

Альтернативная функция `class_index_from_target_score` в `scoring_targets.py` даёт ту же логику для явных score; в train loop используется именно `torch.where` в `_move_batch_to_device`.

### 4.4. Таргет пропуска (omission)

- В артефакте: `omission_target` ∈ {0, 1}.
- В батче: `presence_targets = 1.0 - omission_target` (1 = фонема произнесена).
- Для loss: `omission_targets = 1.0 - presence_targets` (BCE по пропуску).

### 4.5. Поля, не участвующие в loss v2

`duration_targets` загружаются и дополняются padding в батче, но **`train_scorer_v2` их не использует** (в v1 `train_scorer.py` duration мог учитываться — см. note в `eval_scorer_checkpoint.py`). В JSONL- и mmap/parquet-путях `duration_targets` сейчас **дублируют** `regression_target` (`dataset.WordIterableDataset`, `mmap_dataset.pack_jsonl_split_to_mmap`).

---

## 5. Архитектура модели (кратко) и выходы

Полное описание: [scorer_v2_model_architecture_ru.md](../architecture/scorer_v2_model_architecture_ru.md). Реализация: `scorer_model_v2.py` — `PhonemeScorerModelV2`.

**Входы forward:**

- `acoustic_embeddings [B, S, 768]`;
- `phoneme_ids [B, S]`;
- `attention_mask [B, S]`.

**Поток:** `AcousticEncoderV2` → embedding целевой фонемы → fusion (`concat` state, diff, product → `4d` → `d`) → стек `AcousticEncoderBlock` (contextual scorer) → головы.

**Выходы (`ScorerV2Outputs`):**

| Выход | Размерность | Назначение при обучении |
|-------|-------------|-------------------------|
| `quality_logits` | `[B, S, 3]` | CrossEntropy → quality loss |
| `omission_logit` | `[B, S]` | BCEWithLogits → omission loss |
| `class_probs` | `[B, S, 3]` | softmax, диагностика |
| `expected_score` | `[B, S]` | `class_probs · [15, 60, 92]`; метрика MAE vs `score_targets` |
| `expected_human_score` | `[B, S]` | `class_probs · [0, 1, 2]`; **не** в loss |

Гиперпараметры архитектуры задаются CLI `train_scorer_v2` (см. §7) и сохраняются в `checkpoint["config"]` для `scorer_model_kwargs_from_config` при загрузке.

---

## 6. Функции потерь и их комбинирование

Реализация в `_run_epoch` (`train_scorer_v2.py`).

### 6.1. Quality loss

- `nn.CrossEntropyLoss(weight=class_weights, reduction="none")` по `quality_logits` и `class_targets`.
- `class_weights` вычисляются **один раз** до эпох: `_compute_class_weights` — подсчёт частот классов на train loader, формула `total / (num_classes * count)`, нормализация на среднее веса = 1.
- Этот проход заново итерирует `DataLoader`, но не «опустошает» обучение: map-style датасеты (`parquet`/`mmap`) переиспользуются через новый проход, а `WordIterableDataset` создаёт свежий итератор в `__iter__`. Цена — дополнительный полный I/O-проход перед первой эпохой.
- Усреднение: `_masked_mean` — среднее только по позициям с `attention_mask == True`.

### 6.2. Omission loss

- `nn.BCEWithLogitsLoss(reduction="none")` между `omission_logit` и `omission_targets`.
- Та же маскированная средняя.

### 6.3. Суммарный objective

```text
batch_total_loss = batch_quality_loss + omission_loss_weight * batch_omission_loss
```

По умолчанию `omission_loss_weight = 0.25` (`--omission-loss-weight`).

**Не входят в loss:** `expected_human_score`, `duration_targets`, явная регрессия по `match_targets` (MAE по `expected_score` только как **метрика**).

### 6.4. Регуляризация и стабилизация

- **AdamW** с разными LR для энкодера и остальных параметров (§7).
- **Gradient clipping:** `clip_grad_norm_(..., max_norm=1.0)` после `backward`.

---

## 7. Гиперпараметры обучения (`train_scorer_v2.py`)

Значения ниже — **дефолты argparse**; фактические production-запуски могут переопределять их аргументами CLI.

### 7.1. Оптимизация

| Параметр | CLI | По умолчанию |
|----------|-----|--------------|
| Эпохи | `--epochs` | 10 |
| Batch size (слов) | `--batch-size` | 128 |
| Learning rate (не-энкодер) | `--lr` | 3e-4 |
| LR scale для `acoustic_encoder` | `--encoder-lr-scale` | 0.2 → LR энкодера = 6e-5 |
| Weight decay | `--weight-decay` | 1e-2 |
| Optimizer | — | AdamW, две param groups |
| Scheduler | — | **нет в коде** |
| Freeze encoder | `--freeze-encoder-epochs` | 2 (первые эпохи `requires_grad=False` у `acoustic_encoder`) |
| Omission λ | `--omission-loss-weight` | 0.25 |
| Device | `--device` | `cuda` если доступен, иначе `cpu` |

При разморозке энкодера после `freeze_encoder_epochs` оптимизатор **пересобирается** (`_build_optimizer`).

### 7.2. DataLoader

| Параметр | По умолчанию |
|----------|--------------|
| `--num-workers` | 8 |
| `--prefetch-factor` | 4 |
| `--train-shuffle-mode` | `block` |
| `--shuffle-block-words` | 16384 |
| `--train-seed` / `--val-seed` | 1337 / 7331 |
| `pin_memory` | True |

### 7.3. Архитектура модели (CLI)

| Параметр | По умолчанию |
|----------|--------------|
| `--acoustic-input-dim` | 768 |
| `--d-model` | 384 |
| `--num-heads` | 6 |
| `--acoustic-layers` | 6 |
| `--scorer-layers` | 2 |
| `--ffn-dim` | 1536 |
| `--phoneme-vocab-size` | 42 |
| `--phoneme-embed-dim` | 48 |
| `--dropout` | 0.05 |
| `--rope-base` | 10000 |
| `--architecture-version` | `v2_compat` |

Дополнительные флаги v3-блока: `--block-layout`, `--norm-scheme`, `--attention-score-mode`, `--qk-norm-mode`, `--positional-mode`, `--disable-qk-norm` и др.

### 7.4. Связь с предобучением энкодера (Этап 1)

Гиперпараметры предобучения относятся к **Этапу 1** и подробно описаны в разделе [«Этап 1. Предобучение кодировщика»](#этап-1-предобучение-кодировщика-самообучение-без-меток). `pretrain_acoustic_encoder_v2.py` — **не** часть одного вызова `train_scorer_v2`; он связан с Этапом 2 только через `--encoder-checkpoint-path` (загрузка ключей `encoder.*` в `acoustic_encoder`).

---

## 8. Цикл обучения и валидация

### 8.1. `main()` в `train_scorer_v2.py`

1. Создать `PhonemeScorerModelV2`, опционально загрузить pretrained encoder, заморозить encoder если `freeze_encoder_epochs > 0`.
2. Построить train `DataLoader`, вычислить `class_weights` (полный дополнительный проход по train без сдвига первой эпохи).
3. Создать `AdamW` и функции потерь.
4. Если `--val-features-dir` — построить val loader и **закэшировать** все val-батчи в RAM (`_cache_batches`).
5. Для каждой эпохи:
   - обновить обучаемость энкодера;
   - `train_batch_sampler.set_epoch(epoch)` при block shuffle;
   - `_run_epoch(..., optimizer=optimizer, phase="train")`;
   - при наличии val — `_run_epoch(..., optimizer=None, phase="val")`;
   - сохранить `scorer_v2_epoch_{N}.pt`;
   - если val `quality_loss` улучшился — сохранить `scorer_v2_best.pt`.

### 8.2. Early stopping

**В коде нет** patience-based early stopping: цикл всегда выполняет `--epochs` эпох. «Отбор лучшей» модели — только snapshot при улучшении `val_metrics["quality_loss"]` (если val задан).

Если `--val-features-dir` **не** указан, `scorer_v2_best.pt` **не создаётся** автоматически — только epoch checkpoints.

### 8.3. Логирование

Каждые `--log-every` шагов (по умолчанию 100): quality loss, omission loss, score MAE, words/s. В конце эпохи — train/val summary (steps, words, tokens, losses, class accuracy, objective).

### 8.4. Иллюстрация динамики обучения (синтезированный журнал)

См. `scorer_v2_training_logs.png` / `scorer_v2_training_logs.txt` (генератор — `render_scorer_v2_training_logs.py`).

> **Важно: все числа в этом журнале — синтезированные и иллюстративные.** Реальных логов прогона в репозитории нет; кривые потерь подобраны вручную как правдоподобная реконструкция, а не результат измеренного запуска.

Сценарий: 10 эпох, `batch_size = 128`, ~312 шагов train и 36 шагов val на эпоху, заморозка энкодера на первые 2 эпохи (`encoder: trainable` появляется с эпохи 3), `omission_loss_weight = 0,25`. Метрики совпадают с реальными: `quality_loss`, `omission_loss`, `score_mae`, `class_accuracy`, `objective = quality + 0,25 · omission`.

**Что показывает кривая (мягкое переобучение по основной метрике `quality_loss`):**

- **Train `quality_loss` быстро убывает, затем выходит на плато** (выравнивается около ~0,40 примерно с эпохи 7).
- **Разрыв train/val расширяется** со временем: валидационная кривая держится явно выше тренировочной.
- **Локальный «горб» только на валидации в эпохах 7–8**: val `quality_loss` поднимается (0,584 на эпохе 7, 0,554 на эпохе 8), пока train уже на плато — типичный признак переобучения.
- Из-за этого горба **отбор лучшей модели (по val `quality_loss`) не выбирает эпохи 7–8**: они не побеждают предыдущий best с эпохи 6 (0,551), поэтому новый `scorer_v2_best.pt` появляется только на **эпохе 9** (0,526), а затем на эпохе 10 (0,522).
- `omission_loss`, `score_mae` и `class_accuracy` следуют более гладким трендам без выраженного горба.

Это согласуется с логикой §10.2: критерий best — **только** строгое улучшение val `quality_loss`, поэтому переобученные эпохи 7–8 закономерно пропускаются.

---

## 9. Метрики качества

### 9.1. Во время обучения (`_run_epoch`)

Для loss и MAE используется среднее по батчам (среднее средних батчей, не взвешенное по числу токенов). `class_accuracy` считается как доля правильно предсказанных валидных токенов за всю эпоху:

| Метрика | Определение |
|---------|-------------|
| `quality_loss` | masked CE |
| `omission_loss` | masked BCE |
| `score_mae` | mean \|expected_score − score_targets\| на mask |
| `class_accuracy` | доля argmax(quality_logits) == class_targets на валидных токенах |
| `objective_loss` | quality + λ·omission |

### 9.2. Офлайн-оценка чекпоинта

`eval_scorer_v2_checkpoint.py` — полный проход по split и агрегация в JSON:

- `score_mae`, `score_rmse`, `score_pearson`;
- `class_accuracy`, `omission_accuracy` (порог sigmoid 0.5);
- confusion matrix по классам, перцентили score, mean omission probability и диагностика схлопывания (например, все предсказания `correct`, слабая корреляция, почти одинаковые score для разных классов).

Использует те же `_build_dataloader`, `_move_batch_to_device`, что и обучение.

---

## 10. Чекпоинты и отбор лучшей модели

### 10.1. Содержимое checkpoint (`_save_checkpoint`)

```python
{
  "epoch": int,
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "train_metrics": dict,
  "val_metrics": dict | None,
  "class_weights": tensor (CPU),
  "config": vars(args),  # полный Namespace CLI
}
```

### 10.2. Файлы

| Файл | Когда пишется |
|------|----------------|
| `scorer_v2_epoch_{k}.pt` | Каждая эпоха |
| `scorer_v2_best.pt` | Val `quality_loss` строго меньше предыдущего best |

Критерий best: **только** `val_metrics["quality_loss"]`, не `objective_loss` и не `score_mae`.

### 10.3. Деплой

Runtime: `ScorerV2Runtime` (`services/scorer_v2_runtime.py`) загружает checkpoint из `PRONUNCIATION_SCORER_CHECKPOINT_PATH`, восстанавливает модель через `scorer_model_kwargs_from_config(config)` и переводит её в `model.eval()`.

---

## 11. Инференс после обучения (кратко)

**Отличие от train batch:** runtime строит тензоры из `PhoneFeatures` через `PhoneFeatureTensorMapper` — только `mean_embedding[:768]` и `get_phoneme_id`, без variance/duration_z/energy в модель.

**Pipeline API** (`pipeline.py`): `AudioPrepService` → `SSLFeatureEncoder` → `MfaForcedAligner` → `ScorerV2Runtime.score`.

**Выход на фонему** (`ScorerPhonePrediction`): `predicted_class`, `quality_class_probs`, `expected_score`, `expected_human_score`, `omission_probability`, тайминги, `alignment_confidence`.

Калибровка финальных client-facing scores (отдельный calibration layer) в **train_scorer_v2** не обучается — см. [training_artifacts.md](../training_artifacts.md).

---

## 12. Сводка скриптов и зависимостей

| Этап | Модуль |
|------|--------|
| Ingest / prepare / align | `ingest_datasets.py`, `prepare_speechocean762.py`, `build_speechocean762_aligned.py`, … |
| Feature precompute | `precompute_features.py` |
| Mmap pack | `pack_mmap_features.py`, `mmap_dataset.pack_jsonl_split_to_mmap` |
| Parquet bake | `bake_mmap_to_parquet.py` |
| Encoder pretrain | `pretrain_acoustic_encoder_v2.py` |
| **Scorer train** | **`train_scorer_v2.py`** |
| Eval | `eval_scorer_v2_checkpoint.py` |
| Serve | `scorer_v2_runtime.py`, `tensor_mapper.py` |

---

## 13. Неясности и пометки для верификации

| # | Тема | Статус |
|---|------|--------|
| 1 | Использование 3 доп. скаляров (771−768) в будущих версиях | В v2 в модель не подаются |
| 2 | LR scheduler, warmup, early stopping patience | **Не найдено** в `train_scorer_v2.py` |
| 3 | Фактические гиперпараметры production-запусков | В коде подтверждены только defaults; конкретные запуски нужно сверять с логами/скриптами запуска |
| 4 | Обязательность encoder pretrain перед scorer | Рекомендация docs, опционально в CLI |
| 5 | Критерий best только `quality_loss` vs business metric | Зафиксировано в коде |
| 6 | Поведение без val split (отсутствие `scorer_v2_best.pt`) | Зафиксировано в коде |
| 7 | Двойной проход train data при старте (class weights) | `_compute_class_weights` итерирует loader до эпох; первый train epoch получает новый проход |

---

*Документ сверен с исходным кодом репозитория и отредактирован для единообразия терминологии.*
