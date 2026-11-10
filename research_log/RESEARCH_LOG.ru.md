# RESEARCH LOG — DOGE FLIP BOT
*Дата старта: 2025 | Последнее обновление: 2026-11-05*

---

## МОДЕЛЬ

| Параметр | Значение |
|----------|----------|
| Архитектура | ProdTransformer (Transformer Encoder) |
| Таймфрейм | 1h |
| Горизонт предикта | 6 свечей вперёд (6h) |
| Стратегия | FLIP — держим пока не меняется предикт |
| Средняя длина сделки | ~40 часов |
| Признаков | 129 |
| seq_len | 80 |
| d_model | 256 |
| nhead | 8 |
| num_layers | 4 |
| dropout | 0.20–0.25 |
| lr | ~1.8e-04 |

---

## ПАРАМЕТРЫ МОДЕЛИ (ЧЕМПИОН)

| Параметр | Значение |
|----------|----------|
| long_th | 0.49 |
| short_th | 0.55 |
| temperature | 0.80 |
| Путь | `transformer2602/runs/champion_1h/champion_return.pt` |
| Продакшн-бандл | `prod_doge_1h_v1/model.pt` |

---

## ДАТАСЕТ

| | |
|--|--|
| Пара | DOGEUSDT (тест) / DOGEUSDC (прод) |
| Данные | Binance OHLCV: 1h, 4h, 1d + BTC/ETH |
| Дополнительно | Fear & Greed, Funding rate, SPY/QQQ/VIX macro |
| Обучение | до 2024-12-31 |
| Валидация | нет (всё в train) |
| Тест | 2025-01-01 → 2026-02-28 |

---

## AUDIT-2026-08-22 — 1m causal/live parity replay

**Цель:** проверить, можно ли воспроизвести реальные DOGEUSDC сделки `bot_binance_flip` из minute-by-minute Binance replay, прежде чем доверять исторической доходности.

**Данные:**
- DOGEUSDT/BTCUSDT/ETHUSDT Spot 1m и DOGEUSDC Futures 1m;
- `2025-01-01 00:00 → 2026-08-22 07:59 UTC`;
- по `861,600` строк на инструмент, пропусков `0`;
- 1h строится из `60` закрытых минут, 4h из `240`, 1d из `1440`;
- исполнение DOGEUSDC — первый 1m open после hourly decision, fee `4 bps/side`, slippage `2 bps/side`, funding и position fraction `93.1%`.

**Проверенные replay-варианты:**
1. completed-context causal 4h/1d;
2. live-style as-of partial 4h с пересборкой последних `80` feature rows на каждом решении;
3. bundle thresholds `0.49/0.55`;
4. production-candidate thresholds `0.48/0.52`.

**Результат historical parity с предоставленными реальными Last 10 trades:** ни один реконструированный вариант не воспроизвёл последовательность. Лучший вариант дал `0/10` совпадений дат, `6/10` сторон и `5/10` outcomes. Live-asof `0.48/0.52` создал `122` закрытия с апреля вместо редкой реальной FLIP-последовательности.

**Server audit (read-only):**
- SHA256 deployed `model.pt`, `trading_bot.py` и feature builder полностью совпадают с локальными; различие hash `runtime.py` оказалось только line endings, semantic diff отсутствует;
- фактический `.env`: DOGEUSDC, `DATA_START=2024-01-01`, thresholds `0.48/0.52`, fee `4 bps`, position `95%` с buffer `2%`;
- SQLite содержит `3438` hourly CANDLE ticks за `2026-04-01 → 2026-08-22 07:00`, `17` signal changes, `34` соответствующих close/open order events, FLAT signals `0`;
- exact latest-timestamp reproduction на скачанном server feature cache: server `p_up=0.67350375`, local `p_up=0.67345659`, абсолютная ошибка `<0.00005`, direction совпадает.

**Уточнённый вывод:** модель, код и текущий inference воспроизводятся. Historical mismatch вызван тем, что legacy `load_all` каждый час пересобирает mutable feature snapshot (текущие partial 4h, внешние daily sources и их cache state); финальные OHLC задним числом не восстанавливают тот feature matrix, который реально видел сервер. Minute OHLC достаточно для нового детерминированного causal pipeline, но старую модель нельзя честно прогнать назад без сохранённых historical feature snapshots. Server ticks являются достоверным forward OOS с марта 2026. Подбор параметров по реальным сделкам запрещён.

**Артефакты:**
- `replay_binance_flip_causal_1m.py`
- `tests/test_replay_binance_flip_causal_1m.py`
- `analysis_binance_flip_causal_1m/summary.json`
- `analysis_binance_flip_causal_1m/live_asof_production_replay_trades.csv`
- `analysis_binance_flip_causal_1m/server/ticks.csv`
- `analysis_binance_flip_causal_1m/server/trades.csv`

### Deterministic causal FLIP diagnostic, 2025–2026

Текущий checkpoint дополнительно прогнан на едином детерминированном causal pipeline: completed 1h/4h/1d, сигнал после close, исполнение на следующем DOGEUSDC minute open, thresholds `0.48/0.52`, dynamic temperature, position `93.1%`, fee `4 bps/side`, slippage `2 bps/side`, funding и minute mark-to-market открытых позиций.

**Механический результат (`2025-01-01 → 2026-08-22 07:59 UTC`):** return `+8131.07%`, MDD `-56.46%`, `420` trades, win rate `64.52%`, PF `1.555`, profitable months `15/20`.

**Разделение по честности:**
- 2025: `+6088.64%`, MDD `-56.46%` — **не OOS**: этот год использовался для выбора champion среди `100` seeds и grid thresholds;
- 2026 Jan–Mar: `+9.53%`, MDD `-39.48%` — не был сохранён как untouched test;
- 2026 Apr–Aug 22: `+21.43%`, MDD `-33.09%` — ближайший forward diagnostic, но не exact server replay, поскольку historical mutable server snapshots не сохранялись.

**Вердикт:** огромная полная доходность математически согласована с trade ledger, но selection-biased и не является оценкой достижимого OOS. Полезная информация — резкое ослабление edge в 2026 и высокий forward drawdown. Для честной будущей модели нужны causal training features, frozen checkpoint/thresholds до test и сохранение каждого live feature snapshot.

**Артефакты:**
- `run_flip_causal_2025_2026.py`
- `tests/test_run_flip_causal_2025_2026.py`
- `analysis_flip_causal_2025_2026/summary.json`
- `analysis_flip_causal_2025_2026/monthly_returns.csv`
- `analysis_flip_causal_2025_2026/trades.csv`
- `analysis_flip_causal_2025_2026/equity_1m.csv`

### EXP-053 — Causal FLIP Transformer retraining

**Дата:** 2026-08-22  
**Статус:** завершён, **NO-GO; провалены 5/5 preregistered checks**.

Полностью новый training pipeline устранил legacy look-ahead: completed 1h,
4h stamped на последнем constituent hour, previous-day daily/macro/sentiment,
без `bfill`, scaler только на fold train, окно включает текущую завершённую
свечу, target считается от следующего DOGEUSDT open до `close[t+6]`.

Протокол до обучения зафиксировал четыре expanding walk-forward folds, seeds
`11/29/47/71/89` и threshold grid `0.50…0.65`. Seed и threshold выбирались
только на validation; следующий test не участвовал в выборе. DOGEUSDC
исполнение включает position `93.1%`, fee `4 bps/side`, slippage `2 bps/side`
и funding.

**Selected test folds:**
- `2024H2`: `-10.08%`, MDD `-60.50%`, PF `0.918`;
- `2025H1`: `-33.18%`, MDD `-54.88%`, PF `0.716`;
- `2025H2`: `-17.93%`, MDD `-62.09%`, PF `0.898`;
- `2026 YTD`: `+94.80%`, MDD `-25.79%`, PF `3.713`.

**Stitched:** return `-3.95%`, MDD `-62.92%`, PF `0.989`, `166` trades,
positive selected folds `1/4`. Median пяти seed test returns по folds:
`-10.08%, -11.58%, -43.48%, +41.69%`; положительный median только `1/4`.

**Fixed-seed post-analysis:** seed `89`, лучший по aggregate validation,
получил stitched `-51.88%`, MDD `-78.78%`, PF `0.828`, positive folds `1/4`.
Seed `29`, выбранный post-hoc как лучший по aggregate test, получил `+83.68%`,
но MDD `-72.87%`, PF `1.144` и positive folds только `2/4`; folds:
`+46.84%, -11.58%, -51.39%, +191.06%`. Положительный итог почти полностью
создан 2026 YTD и не проходит risk/stability criteria. Использовать seed `29`
как production champion запрещено, поскольку его выбрали после просмотра test.

**Direct 2025–2026 correction-grid rerun:** чтобы проверить полный FLIP stack,
отдельно зафиксирован split train `≤2024-06-30`, grid `2024H2`, one-shot test
`2025-01-01 → 2026-08-22`. Для каждого из пяти seeds проверены `144`
fixed/dynamic temperature corrections × `49` LTH/STH pairs: `35,280`
validation configurations. Exact FLIP signal rule, DOGEUSDC next-open, fee,
slippage и funding включены.

Validation выбрал seed `47`, fixed `T=0.75`, LTH/STH `0.48/0.58`:
return `+191.44%`, MDD `-64.57%`, PF `1.823`. Dynamic correction участвовала,
но не победила. One-shot 2025–2026 test: return `-45.91%`, MDD `-73.94%`,
PF `0.878`, `231` trades; 2025 `-21.92%`, 2026 YTD `-30.72%`.
Следовательно, отсутствие correction grid не объясняло предыдущий провал:
совместно выбранные seed/temperature/LTH/STH также не переносятся с 2024H2.

**Optuna best-seed champion search:** выполнены `60` trials на pre-H2'25
development split; top-3 архитектуры затем обучены на данных включая 2025H1,
каждая по `20` seeds с полным correction grid на 2025H2. Все top architectures
сошлись к `horizon=24h`, `d_model=256`, `3` layers, `seq_len=40`,
direction-loss weight `0.9`. Historical champion: trial `53`, seed `0`,
dynamic volume `α=1.0`, `T=[0.70,1.40]`, LTH/STH `0.65/0.65`; 2025H2
selection `+36.15%`, MDD `-27.32%`, PF `1.250`, `131` trades.

Frozen 2026 diagnostic: `-11.66%`, MDD `-45.04%`, PF `0.887`, `156` trades.
Медиана 20 seeds у всех top-3 architectures на selection была отрицательной
(`-6%…-14%`), positive seed share только `35–40%`. Лучший seed подтверждает,
что архитектура может fit отдельный режим, но его edge не перенёсся на
следующий период; Optuna champion также **NO-GO**.

**Вывод:** 2026-режим выглядит сильным, но validation→test перенос провален в
первых трёх folds, seed variance экстремален, а stitched edge отсутствует.
Новый checkpoint не заменяет production FLIP. Post-hoc смена seeds,
thresholds или критериев на этих test windows запрещена.

**Артефакты:**
- `causal_flip/PROTOCOL.md`
- `causal_flip/features.py`
- `causal_flip/train_walkforward.py`
- `tests/test_causal_flip_training.py`
- `causal_flip/reports/walkforward_v1/REPORT.md`
- `causal_flip/reports/walkforward_v1/summary.json`
- `causal_flip/reports/FIXED_SEED_COMPARISON.md`
- `causal_flip/DIRECT_GRID_PROTOCOL.md`
- `causal_flip/reports/direct_grid_2025_2026/REPORT.md`
- `causal_flip/OPTUNA_CHAMPION_PROTOCOL.md`
- `causal_flip/reports/optuna_champion/REPORT.md`

---

## ЭКСПЕРИМЕНТЫ

---

### EXP-001 — Baseline FLIP стратегия (полный период)
**Дата:** ~2026-02  
**Период теста:** 2025-01-01 → 2026-02-28

| Метрика | Значение |
|---------|----------|
| Total Return | +8020% |
| MDD | -39.1% |
| Calmar | 205 |
| Trades | 253 |
| Win Rate | 66.4% |
| Profit Factor | 3.07 |

**Комментарий:** Базовая модель, 129 фичей, без фильтров.

---

### EXP-002 — Grid test порогов (шаг 0.01)
**Дата:** ~2026-02  
**Цель:** Найти оптимальные long_th / short_th  
**Результат:** long_th=0.49, short_th=0.55 — оптимальные  
**MDD:** не изменился существенно при вариации порогов

---

### EXP-003 — Температурная калибровка
**Дата:** ~2026-02  
**Параметр:** temperature=0.80 (сглаживание логитов)  
**Эффект:** улучшило качество сигналов, вошло в финальную конфигурацию

---

### EXP-004 — Помесячная доходность (Baseline)
**Период:** 2025-01 → 2026-02

| Месяц | Return |
|-------|--------|
| 2025-01 | сильно положительный (Bull run) |
| 2025-02 | сильно положительный |
| 2025-03 | положительный |
| 2025-04 | положительный |
| 2025-05 | положительный |
| 2025-06 | положительный |
| 2025-07 | слабо положительный |
| 2025-08 | отрицательный |
| 2025-09 | отрицательный |
| 2025-10 | слабо положительный |
| 2025-11 | слабо положительный |
| 2025-12 | слабо положительный |
| 2026-01 | отрицательный (Trump/политика) |
| 2026-02 | **сильно отрицательный** (главный убийца) |

**Вывод:** Февраль 2026 "съел" значительную часть прибыли. Причина — Трамп/политические события → непредсказуемые движения.

---

### EXP-005 — Анализ топ-10 худших сделок
**Дата:** ~2026-02  
**Метод:** Ранжировка по абсолютному убытку ($)

| # | Entry | Direction | Ret% | LossUSD |
|---|-------|-----------|------|---------|
| 1 | 2026-01-29 | LONG | -9.24% | $-96,504 |
| 2 | 2026-02-04 | LONG | -10.17% | $-94,161 |
| 3 | 2025-11-17 | LONG | -10.62% | $-84,141 |
| 4 | 2026-01-13 | SHORT | -5.89% | $-58,821 |
| 5 | 2025-09-06 | SHORT | -13.40% | $-55,890 |
| 6 | 2025-08-03 | SHORT | -13.98% | $-49,665 |
| 7 | 2025-10-01 | SHORT | -10.35% | $-48,916 |
| 8 | 2025-07-31 | LONG | -11.15% | $-44,583 |
| 9 | 2025-10-16 | LONG | -6.55% | $-44,168 |
| 10 | 2026-02-13 | SHORT | -4.94% | $-42,258 |

**Суммарный ущерб топ-10:** $619,107  
**Паттерны:**
- Jan–Feb 2026: Trump/политические события
- Aug–Sep 2025: короткий шорт против бычьего тренда
- Проблемная сделка 2026-01-29: повторяется во всех экспериментах

---

### EXP-006 — Permutation Importance (129 фичей)
**Дата:** ~2026-02  
**Метод:** Перемешиваем одну фичу, смотрим падение Return

**Топ-5 ВРЕДНЫХ фичей (снижают доходность):**

| # | Фича | Δ Return |
|---|------|---------|
| 1 | `btc_ema200_ratio` | сильно негативный |
| 2 | `btc_range_pct` | негативный |
| 3 | `eth_ma_ratio_24` | негативный |
| 4 | `btc_adx` | негативный |
| 5 | `macro_market_regime` | негативный |

**Всего вредных:** 13 фичей  
**Полезных:** ~80 фичей  
**Нейтральных:** остальные

---

### EXP-007 — Zero-out harmful features (без переобучения)
**Дата:** ~2026-02  
**Метод:** Обнуляем вредные фичи в инференсе, не переобучая

#### Результаты (2025-01 → 2026-02):

| Вариант | Return | MDD | Calmar | Trades | Win% | PF |
|---------|--------|-----|--------|--------|------|----|
| Baseline (129 фичей) | +8020% | -39.1% | +205 | 253 | 66.4% | 3.07 |
| Zero top-5 harmful | +7639% | -38.6% | +198 | 158 | 74.1% | 3.94 |
| Zero all 13 harmful | хуже baseline | — | — | — | — | — |

**Вывод:**
- Win rate вырос с 66% → 74% при зануление топ-5
- Доходность упала (−381%) — пропустили часть хороших сделок
- Зануление всех 13 фичей — хуже baseline (фичи дают полезные сигналы в Jan-Feb 2025)
- Топ-5 вредных фичей иногда дают правильный выход из позиции → просто "шумные", не бесполезные

**Топ-10 худших (Zero top-5) vs Baseline:**
- Исчезли: 2025-09-06, 2025-10-01, 2025-11-17, 2026-01-13, 2026-02-04, 2026-02-13
- Остались: 2025-07-31, 2025-08-03, 2025-10-16, 2026-01-29
- Новые: 2025-09-07, 2025-11-20, 2025-12-04, 2026-01-15, 2026-02-12, 2026-02-24
- `2026-01-29` стала хуже ($96K → $141K)

---

### EXP-008 — Ансамбль A+B (фильтр согласия)
**Дата:** 2026-03-05  
**Идея пользователя:** A и B могут хеджировать друг друга. Если A и B расходятся — сигнал ненадёжен, пропускаем.

**Расхождение моделей:** 18.3% баров (1818 из ~9900)

#### Результаты (2025-01-01 → 2026-02-28):

| Вариант | Return | MDD | Calmar | Trades | Win% | PF |
|---------|--------|-----|--------|--------|------|----|
| A baseline (129 фичей) | +8020% | -39.1% | +205 | 253 | 66.4% | 3.07 |
| A фильтр B (согласие) | +7639% | -38.6% | +198 | 158 | **74.1%** | **3.94** |
| B фильтр A (согласие) | +7639% | -38.6% | +198 | 158 | **74.1%** | **3.94** |
| **Avg ensemble (мягкий)** | **+8319%** | **-33.8%** | **+246** | 235 | 69.4% | 3.32 |

**Лучший вариант: Avg ensemble**
- MDD снизился: -39.1% → **-33.8%**
- Calmar вырос: 205 → **246**
- Доходность выше baseline: +8020% → **+8319%**
- Trades: 253 → 235

**Топ-3 худших сделок исчезли при фильтре согласия:**
- ✅ 2026-01-13 — исчезла
- ✅ 2026-02-04 — исчезла
- ✅ 2026-02-13 — исчезла
- ❌ 2026-01-29 — осталась (системная проблема, обе модели уверены в ней)

**Вывод:** `Avg ensemble` — наилучший вариант без переобучения. Готов к продакшну.

---

## ОТКЛОНЁННЫЕ СТРАТЕГИИ

| Стратегия | Причина отклонения |
|-----------|-------------------|
| Stop-loss (фиксированный %) | Ухудшает результат — срабатывает до разворота |
| Хеджирование встречной позицией | Снижает доходность, сложность управления |
| Stop & re-entry | Улучшает MDD, но снижает return |
| Win-rate circuit breaker | Неэффективен — просто пропускает хорошие сделки |
| Confidence-based sizing | Неэффективен для данной модели |
| Volume management (динамический размер) | Ухудшает результат |

---

## ИНФРАСТРУКТУРА

| Компонент | Путь | Статус |
|-----------|------|--------|
| Продакшн-бот | `bot_binance_flip/` | ✅ Работает на сервере |
| Модель бандл | `prod_doge_1h_v1/model.pt` | ✅ |
| Сервер | `<server>` (root) | ✅ Ubuntu 20.04 |
| Systemd сервис | `bot_binance_flip` | ✅ Автозапуск |
| Telegram бот | Команды: /stats /monthly /trades /pos /panel | ✅ |
| Пара | DOGEUSDC (Binance) | ✅ |

---

---

### EXP-009 — Переобучение v2 (новые фичи, без вредных)
**Дата:** 2026-03-05  
**Статус:** 🔄 В процессе (Optuna запущена)

**Изменения фичей:**

| Действие | Фичи |
|----------|------|
| ❌ Убраны (вредные по EXP-006) | `btc_ema200_ratio`, `btc_range_pct`, `btc_adx`, `btc_di_plus`, `btc_di_minus`, `eth_ma_ratio_24`, `macro_market_regime` |
| 🐛 Исправлен баг | `funding_rate` / `funding_ma8` — ключ был `DOGEUSDT_funding` (не существовал), исправлено на `funding` → теперь реальные данные |
| ✅ Добавлены новые | `doge_cvd_z24`, `doge_cvd_slope12` (Cumulative Volume Delta), `doge_vwap_dev24` (VWAP отклонение), `doge_vol_regime` (режим волатильности), `doge_candle_eff` (эффективность свечи), `doge_price_accel` (ускорение цены), `doge_ret_ac12/24` (автокорреляция доходности), `doge_btc_ratio_dev` (DOGE/BTC ratio) |

**Итоговый размер:** 131 фича (было 129)

**Сплит данных (строгий, без утечки):**
- Train: 2018-01-01 → 2024-09-30
- Val:   2024-10-01 → 2024-12-31
- Test:  2025-01-01 → 2025-12-31 (grid search FLIP)

**Команда:** `python -m transformer2602.optuna_1h_v2 --n_trials 60 --n_seeds 5 --device cuda`

---

---

### EXP-010 — Joint Grid: LTH × STH × Temperature (фиксированная)
**Дата:** 2026-03-06  
**Скрипт:** `grid_temp_thresholds.py`  
**Метод:** 441 комбинация (9T × 7LTH × 7STH) на тест-периоде 2025-01 → 2026-02

**Лучший по Calmar:**

| Параметр | Значение |
|----------|----------|
| T | 1.00 |
| LTH | 0.48 |
| STH | 0.52 |
| Return | +13,517% |
| MDD | -37.7% |
| Calmar | 358.65 |
| Trades | 245 |

**vs PROD (T=0.80, LTH=0.49, STH=0.55):** Return=+8020%, Calmar=205

**Наблюдения:**
- T=1.00 (без температурной коррекции) — лучшая во всех вариантах
- LTH=0.48 (чуть ниже текущего) — ловит больше шортов
- STH=0.52 вместо 0.55 — меньше пропущенных шортов
- **⚠️ Параметры отобраны на тест-периоде → нечестный результат**

---

### EXP-011 — Помесячное сравнение топ-конфигураций
**Дата:** 2026-03-06  
**Скрипт:** `grid_monthly_top3.py`

| Месяц | PROD | BASE T=1.00 | #1 eff α=3 | #3 body α=3 |
|-------|------|-------------|------------|------------|
| 2025-01 | +220% | +230% | **+259%** | +241% |
| 2025-02 | +103% | +116% | +113% | **+119%** |
| 2025-03 | +39% | +41% | **+70%** | +68% |
| 2025-06 | -8% | -7% | -10% | -5% |
| 2025-07 | +114% | +133% | +133% | **+141%** |
| 2025-11 | **+5%** | +1% | -2% | **+5%** |
| 2026-02 | -15% | -14% | **-9%** | -12% |

**Вывод:** BASE T=1.00 выигрывает в 8 из 14 месяцев. Февраль 2026: динамические варианты уменьшают убыток (-9% vs -15%).

---

### EXP-012 — Динамическая температура (per-bar)
**Дата:** 2026-03-06  
**Скрипты:** `dynamic_temp_test.py`, `check_dyntemp_median.py`

**Концепция:**
```
T_bar = T_max - strength[bar] × (T_max - T_min)
strength = sigmoid(α × z_score(candle_factor))

Сильная свеча (тело ≈ диапазон, нет теней) → T низкая → уверенный сигнал
Слабая свеча (дожи, длинные тени) → T высокая → осторожный сигнал
```

**Факторы:**
- `efficiency` = |close-open| / (high-low) — доля тела в диапазоне свечи
- `body` = |close-open| / open — размер тела
- `body+efficiency` — комбо
- `volume` — объём vs скользящее среднее 20 баров

**Ключевой результат (135 комбинаций на тест-периоде):**

| Метрика | Значение |
|---------|----------|
| Лучших PROD по Return | **135 из 135 (100%)** |
| Лучших PROD по Calmar | **135 из 135 (100%)** |
| Лучших PROD по MDD | 117 из 135 (87%) |
| Медиана Return | +14,602% (PROD: +8,020%) |
| Медиана Calmar | 387.77 (PROD: 205.27) |
| Медиана MDD | -37.1% (PROD: -39.1%) |
| Минимум Return | +9,682% |

**Лучший конфиг (топ-1):**
- factor=efficiency, α=3.0, T=[0.70–1.30]
- Return=+19,037%, MDD=**-27.2%**, Calmar=**700**

**Вывод:** 100% комбинаций бьют PROD → это не случайность, механизм работает.

---

### EXP-013 — Честный Walk-Forward тест
**Дата:** 2026-03-06  
**Скрипты:** `honest_grid.py`, `monthly_walkforward.py`

#### Тест A: Параметры с валидации → тест
- Валидация: 2024-07 → 2024-12 (6 мес, трамп-ралли)
- Тест: 2025-01 → 2026-02 (14 мес)
- Best-on-VAL → Test: **+5,399%** vs PROD **+8,020%** ❌

#### Тест B: Помесячный walk-forward
- Каждый месяц — лучшие параметры из предыдущего месяца
- WF итог: **+1,912%** vs PROD **+6,521%** ❌

**Почему не работает:**
1. 2024 — сильный бычий рынок (трамп-ралли в ноябре), несравним с 2025
2. Параметры предыдущего месяца не предсказывают следующий (рынок не авторегрессивен)
3. Overfit gap: 13,150% разницы между честным и нечестным лучшим

**Вывод:** Любая постобучаемая коррекция (WF, адаптивные параметры) хуже фиксированного PROD на данных режимах.

---

### EXP-014 — Тепловая карта динамической температуры
**Дата:** 2026-03-06  
**Скрипт:** `dyntemp_heatmap.py`  
**Размер грида:** 1536 комбинаций (4 фактора × 6 alpha × 8 T_min × 8 T_max)

**Стабильные зоны (Calmar > 350 у всех 6 alpha-значений):**

| Фактор | Лучшая зона T | Медиана Calmar |
|--------|--------------|----------------|
| efficiency | [0.40–1.30] | 568 |
| body | [0.40–1.30] | **576** |
| body+efficiency | [0.40–1.40] | 572 |
| volume | [0.40–1.50] | 562 |

**Топ-1 по Calmar:**
- efficiency, α=3.0, T=[0.70–1.30] → Calmar=**700**, Return=+19,037%, MDD=**-27.2%**

**MDD снизился с -37% до -27%** — именно это двигает Calmar вверх.

**Рекомендованные параметры для прода (консервативный выбор):**
```
factor  = efficiency
alpha   = 1.5   (средний, не экстремальный)
T_min   = 0.55  (не 0.40 — слишком агрессивно)
T_max   = 1.30
LTH     = 0.48
STH     = 0.52
```

---

### EXP-015 — Shock Cooldown Filter (anti-shock after extreme candles)
**Дата:** 2026-04-12  
**Скрипт:** `shock_cooldown_grid.py`  
**Идея:** после аномальной свечи (`z-score` диапазона `(high-low)/open`) не открывать новую позицию `N` баров.  
**Логика:** позицию можно закрыть по flip/exit, но повторный вход откладывается на `cooldown`.

**Периоды оценки:**
- Full: `2025-01-01 → 2026-02-28`
- 2025 отдельно: `2025-01-01 → 2025-12-31`
- 2026 отдельно: `2026-01-01 → 2026-02-28`

#### Бейзлайны

| Конфиг | Full Return | Full Calmar | 2025 Return | 2026 Return | 2026 MDD |
|--------|-------------|-------------|-------------|-------------|----------|
| FIXED T=1.00, 0.48/0.52 | +13,516.6% | 358.65 | +12,946.5% | -0.34% | -37.7% |
| DYN PROD α=1.5 [0.55–1.30], 0.48/0.52 | +15,428.9% | 401.56 | +14,077.4% | +4.59% | -38.4% |

#### Лучший мягкий trade-off

| Конфиг | Shock Z | Cooldown | Full Return | Full Calmar | 2025 Δ | 2026 Δ |
|--------|---------|----------|-------------|-------------|--------|--------|
| FIXED | 2.0 | 1 бар | +13,697.2% | 368.78 | -30.4% | +1.56% |

**Интерпретация:**
- Для `FIXED` простой anti-shock cooldown дал **очень маленькое** улучшение 2026 и небольшой рост общего Calmar.
- Для `DYN PROD` **ни один мягкий фильтр не улучшил 2026** без заметной потери сильного 2025.
- Агрессивные cooldown-фильтры (`12` баров) действительно улучшают 2026, но **ломают 2025** и сильно режут full-return.

**Вывод:**
- Простая эвристика `shock-bar -> pause entries` **не решает проблему 2026**.
- Для `DYN PROD` базовый режим уже лучше любого мягкого anti-shock cooldown.
- Если и продолжать эту ветку, то нужен **более умный shock filter**, а не просто cooldown после большой свечи.

**Артефакты:**
- `shock_cooldown_results.csv`

---

### EXP-016 — Shock Threshold Boost (anti-shock via stricter entry thresholds)
**Дата:** 2026-04-12  
**Скрипт:** `shock_threshold_boost_grid.py`  
**Идея:** на shock-барах не ставить паузу, а временно повышать пороги входа:
- `LTH = 0.48 + boost`
- `STH = 0.52 + boost`
- shock-режим включается при высоком `z-score` диапазона свечи

**Периоды оценки:**
- Full: `2025-01-01 → 2026-02-28`
- 2025 отдельно: `2025-01-01 → 2025-12-31`
- 2026 отдельно: `2026-01-01 → 2026-02-28`

#### Бейзлайны

| Конфиг | Full Return | Full Calmar | 2025 Return | 2026 Return | 2026 MDD |
|--------|-------------|-------------|-------------|-------------|----------|
| FIXED T=1.00, 0.48/0.52 | +13,516.6% | 358.65 | +12,946.5% | -0.34% | -37.7% |
| DYN PROD α=1.5 [0.55–1.30], 0.48/0.52 | +15,428.9% | 401.56 | +14,077.4% | +4.59% | -38.4% |

#### Лучшие результаты

**Лучший для FIXED:**
- `shock_z = 2.0`
- `boost = +0.01`
- Full Return: `+14,312.8%`
- Full Calmar: `379.78`
- 2025 Return: `+13,281.8%`  (`+335%` к baseline)
- 2026 Return: `+2.84%`      (`+3.18%` к baseline)

**Лучший для DYN PROD:**
- `shock_z = 2.5`
- `boost = +0.02`
- Full Return: `+17,151.8%`
- Full Calmar: `452.34`
- 2025 Return: `+15,522.5%`  (`+1,445%` к baseline)
- 2026 Return: `+5.44%`      (`+0.86%` к baseline)

**Вывод:**
- В отличие от cooldown, `shock threshold boost` оказался **полезным**.
- Для `FIXED` это уже заметное улучшение и по `2025`, и по `2026`.
- Для `DYN PROD` улучшение `2026` небольшое, но:
  - общий Return вырос очень сильно
  - Full Calmar тоже заметно вырос (`401.56 → 452.34`)
- На текущий момент `DYN PROD + shock threshold boost` выглядит **лучше простого DYN PROD**.

**Артефакты:**
- `shock_threshold_boost_results.csv`

---

### EXP-017 — Shock Asymmetric Threshold Boost (LONG vs SHORT separately)
**Дата:** 2026-04-12  
**Скрипт:** `shock_asym_threshold_grid.py`  
**Идея:** в shock-режиме повышать `LONG` и `SHORT` пороги не одинаково, а отдельно:
- `LTH = 0.48 + l_boost`
- `STH = 0.52 + s_boost`

Цель — понять, какая сторона рынка сильнее ломает стратегию в `2026`.

#### Бейзлайны

| Конфиг | Full Return | Full Calmar | 2025 Return | 2026 Return |
|--------|-------------|-------------|-------------|-------------|
| FIXED T=1.00, 0.48/0.52 | +13,516.6% | 358.65 | +12,946.5% | -0.34% |
| DYN PROD α=1.5 [0.55–1.30], 0.48/0.52 | +15,428.9% | 401.56 | +14,077.4% | +4.59% |

#### Лучшие находки

**Лучший FIXED:**
- `shock_z = 2.0`
- `l_boost = +0.02`
- `s_boost = +0.01`
- Full Return: `+14,550.0%`
- Full Calmar: `386.07`
- 2025 %`
- 2026 Return: `+2.06%`Return: `+13,606.5

**Лучший DYN PROD по 2026:**
- `shock_z = 2.0`
- `l_boost = +0.02`
- `s_boost = +0.00`
- Full Return: `+16,044.2%`
- Full Calmar: `417.58`
- 2025 Return: `+14,496.4%`
- 2026 Return: `+5.61%`

**Лучший DYN PROD по общему качеству остался симметричным (из EXP-016):**
- `shock_z = 2.5`
- `boost = +0.02 / +0.02`
- Full Return: `+17,151.8%`
- Full Calmar: `452.34`
- 2026 Return: `+5.44%`

**Вывод:**
- Асимметрия дала **полезный сигнал**, что для `DYN PROD` в shock-режиме выгоднее сильнее душить `LONG`, чем `SHORT`.
- По `2026 Return` вариант `LONG-only boost` оказался лучшим (`+5.61%` vs `+5.44%` у симметричного).
- Но по **общему Return и Calmar** асимметрия **не обошла** симметричный boost из `EXP-016`.
- Значит:
  - как диагностический вывод асимметрия полезна;
  - как финальный рабочий слой пока лучше выглядит **симметричный shock threshold boost**.

**Артефакты:**
- `shock_asym_threshold_results.csv`

---

### EXP-018 — 4H H3 FLIP Optuna с новым временным сплитом
**Дата:** 2026-06-17  
**Скрипт:** `transformer2602/optuna_4h_flip.py`  
**Статус:** стартуем baseline-прогон перед MTF-версией.

**Идея:** проверить 4H FLIP модель с таргетом `HORIZON = 3` свечи на более честной схеме годов:
- Train: `2018-01-01` → `2024-12-31`
- Validation / Optuna objective / threshold grid: `2025-01-01` → `2025-12-31`
- Final OOS holdout: `2026-01-01` → now / `2026-12-31`

**Почему так:**
- 2025 используется для выбора архитектуры, гиперпараметров и FLIP-порогов.
- 2026 не должен участвовать в подборе; его используем только после выбора параметров.
- Это даст честную точку сравнения перед экспериментом с расширенными MTF-фичами из `1h`.

**План запуска:**

```bash
python -m transformer2602.optuna_4h_flip --n_trials 30 --n_seeds 3 --device cuda --study prod4h_flip_h3_train18_24_val25
```

**Что фиксируем после завершения:**
- лучший trial и параметры;
- лучший FLIP return на 2025;
- количество trials/seeds;
- затем отдельный финальный прогон/оценка на 2026 без подбора.

#### Результат Optuna baseline

**Команда:**

```bash
python -m transformer2602.optuna_4h_flip --n_trials 30 --n_seeds 3 --device cuda --study prod4h_flip_h3_train18_24_val25
```

**Итог:**
- Trials: `30`
- Seeds/trial: `3`
- Features: `135`
- Train bars: `12031`
- Val/objective bars: `2185`
- Best trial: `#21`
- Best 2025 FLIP return: `+661.7%`

**Best params:**
- `d_model = 128`
- `nhead = 4`
- `num_layers = 2`
- `ff_mult = 4`
- `seq_len = 20`
- `dropout = 0.40`
- `lr = 0.00014996774557492245`
- `weight_decay = 0.0005092988415422602`
- `batch_size = 512`
- `epochs = 80`
- `patience = 8`

**Артефакты:**
- `transformer2602/runs/optuna_4h_flip/best_params.json`
- `transformer2602/runs/optuna_4h_flip/best_params_base.json`
- `transformer2602/runs/optuna_4h_flip/prod4h_flip_h3_train18_24_val25.db`

#### Финальный OOS 2026 по best params

**Команда:**

```bash
python -m transformer2602.champion_4h_flip --n_seeds 30 --device cuda --params "transformer2602/runs/optuna_4h_flip/best_params.json" --out_dir "transformer2602/runs/champion_4h_flip_train18_24_val25_test26"
```

**Итог по 30 seeds на 2026:**
- Max return: `+41.4%`
- Mean return: `+18.2%`
- Median return: `+19.6%`
- Min return: `-2.9%`
- Std return: `11.1%`
- Profitable seeds: `28/30` (`93%`)
- Max Calmar: `+15.67`
- Mean Calmar: `+3.48`
- Median Calmar: `+1.63`

**Champion by return:**
- Seed: `20`
- 2026 return: `+41.4%`
- Calmar: `+1.58`
- MDD: `-26.2%`
- Trades: `43`
- Win-rate: `56%`
- Thresholds: `LTH=0.60`, `STH=0.55`

**Champion by Calmar:**
- Seed: `26`
- 2026 return: `+7.4%`
- Calmar: `+15.67`
- MDD: `-0.5%`
- Win-rate: `75%`
- Thresholds: `LTH=0.70`, `STH=0.75`

**Вывод:**
- Новый split уже дает рабочий baseline: 2026 не использовался в Optuna, но 28/30 seed оказались прибыльными.
- Лучшие результаты идут не от большой модели, а от компактной конфигурации `d_model=128`, `num_layers=2`, `seq_len=20`.
- Следующий эксперимент имеет смысл делать как MTF-версию: target `4h H3`, но расширенные фичи строить из `1h` и аккуратно агрегировать к закрытому `4h` бару без утечки.

**Артефакты:**
- `transformer2602/runs/champion_4h_flip_train18_24_val25_test26/champion_return.pt`
- `transformer2602/runs/champion_4h_flip_train18_24_val25_test26/champion_calmar.pt`
- `transformer2602/runs/champion_4h_flip_train18_24_val25_test26/leaderboard.json`

---

### EXP-019 — 4H H3 FLIP Optuna с MTF 1H-derived фичами
**Дата:** 2026-06-17  
**Скрипты:**  
- `transformer2602/src/feature_builder_4h_mtf.py`
- `transformer2602/optuna_4h_flip.py --feature_set mtf`

**Идея:** оставить таргет `4h H3` и FLIP-стратегию, но расширить входы модели:
- baseline 4H feature set: `135` фич;
- completed 1H-derived snapshot на каждый закрытый 4H бар: `131` фича;
- итоговая MTF матрица: `266` фич;
- automatic drop all-zero/constant фичей по train-window до `2024-12-31`.

**Проверка данных перед запуском:**
- Last 4H bar: `2026-06-17 16:00`
- NaN: `0`
- Dead features dropped: `0`

**Команда запуска:**

```bash
python -m transformer2602.optuna_4h_flip --feature_set mtf --n_trials 30 --n_seeds 3 --device cuda --study prod4h_mtf_h3_train18_24_val25
```

#### Результат Optuna MTF

**Итог:**
- Trials: `30`
- Seeds/trial: `3`
- Features: `266`
- Best trial: `#5`
- Best 2025 FLIP return: `+777.5%`

**Best params:**
- `d_model = 128`
- `nhead = 8`
- `num_layers = 5`
- `ff_mult = 4`
- `seq_len = 30`
- `dropout = 0.15`
- `lr = 0.00001155233613914713`
- `weight_decay = 0.00006740513796374044`
- `batch_size = 128`
- `epochs = 50`
- `patience = 12`

**Top MTF trials:**
- `#5`: `+777.5%`
- `#14`: `+604.4%`
- `#22`: `+598.8%`
- `#15`: `+566.5%`
- `#19`: `+471.3%`

#### Финальный OOS 2026 по MTF best params

**Команда:**

```bash
python -m transformer2602.champion_4h_flip --n_seeds 30 --device cuda --params "transformer2602/runs/optuna_4h_flip/best_params.json" --out_dir "transformer2602/runs/champion_4h_mtf_train18_24_val25_test26"
```

**Итог по 30 seeds на 2026:**
- Max return: `+56.8%`
- Mean return: `+25.1%`
- Median return: `+21.3%`
- Min return: `-1.9%`
- Std return: `16.7%`
- Profitable seeds: `29/30` (`97%`)
- Max Calmar: `+29.83`
- Mean Calmar: `+3.68`
- Median Calmar: `+2.08`

**Champion by return:**
- Seed: `12`
- 2026 return: `+56.8%`
- Calmar: `+1.87`
- MDD: `-30.5%`
- Trades: `51`
- Win-rate: `65%`
- Thresholds: `LTH=0.50`, `STH=0.50`

**Champion by Calmar:**
- Seed: `3`
- 2026 return: `+5.7%`
- Calmar: `+29.83`
- MDD: `-0.2%`
- Win-rate: `67%`
- Thresholds: `LTH=0.65`, `STH=0.75`

**Сравнение с EXP-018 baseline 4H:**
- 2025 Optuna best: `+777.5%` vs `+661.7%`
- 2026 max return: `+56.8%` vs `+41.4%`
- 2026 mean return: `+25.1%` vs `+18.2%`
- Profitable seeds: `29/30` vs `28/30`
- Max Calmar: `+29.83` vs `+15.67`

**Вывод:**
- MTF-фичи из `1h`, агрегированные к закрытому `4h`, улучшили и validation 2025, и OOS 2026.
- Dead-feature drop ничего не удалил (`0`), то есть текущие нулевые OI/L/S не находятся в feature matrix.
- По 2026 MTF сильнее baseline, но return-champion имеет глубокую просадку (`-30.5%`), поэтому для боевого кандидата интереснее смотреть Calmar/threshold режимы или ограничивать риск.

**Артефакты:**
- `transformer2602/runs/optuna_4h_flip/best_params.json`
- `transformer2602/runs/optuna_4h_flip/best_params_mtf.json`
- `transformer2602/runs/champion_4h_mtf_train18_24_val25_test26/champion_return.pt`
- `transformer2602/runs/champion_4h_mtf_train18_24_val25_test26/champion_calmar.pt`
- `transformer2602/runs/champion_4h_mtf_train18_24_val25_test26/leaderboard.json`

---

### EXP-020 — 4H H4 FLIP Optuna с MTF 1H-derived фичами
**Дата:** 2026-06-17  
**Статус:** стартуем отдельный target `HORIZON = 4`.

**Идея:** проверить, даст ли более длинный таргет `4` свечи 4H (`16h` вперед) более устойчивый FLIP-сигнал.

**Важно по сохранению данных:**
- H3 артефакты не удаляем.
- H4 запускаем отдельным study и отдельными out_dir.
- Скрипт теперь сохраняет params с horizon-specific именем:
  - `best_params_<feature_set>_h<horizon>.json`

**Команда запуска:**

```bash
python -m transformer2602.optuna_4h_flip --feature_set mtf --horizon 4 --n_trials 30 --n_seeds 3 --device cuda --study prod4h_mtf_h4_train18_24_val25
```

#### Результат Optuna MTF H4

**Итог:**
- Trials: `30`
- Seeds/trial: `3`
- Features: `266`
- Best trial: `#15`
- Best 2025 FLIP return: `+623.4%`

**Best params:**
- `d_model = 512`
- `nhead = 4`
- `num_layers = 4`
- `ff_mult = 4`
- `seq_len = 60`
- `dropout = 0.25`
- `lr = 0.00002214053487091619`
- `weight_decay = 0.00000311281955471107`
- `batch_size = 512`
- `epochs = 50`
- `patience = 15`

**Top H4 trials:**
- `#15`: `+623.4%`
- `#11`: `+574.3%`
- `#12`: `+498.6%`
- `#21`: `+492.1%`
- `#8`: `+490.6%`

#### Финальный OOS 2026 по H4 best params

**Команда:**

```bash
python -m transformer2602.champion_4h_flip --n_seeds 30 --device cuda --params "transformer2602/runs/optuna_4h_flip/best_params_mtf_h4.json" --out_dir "transformer2602/runs/champion_4h_mtf_h4_train18_24_val25_test26"
```

**Итог по 30 seeds на 2026:**
- Max return: `+149.7%`
- Mean return: `+43.8%`
- Median return: `+35.7%`
- Min return: `+1.1%`
- Std return: `34.5%`
- Profitable seeds: `30/30` (`100%`)
- Max Calmar: `+13.85`
- Mean Calmar: `+4.36`
- Median Calmar: `+3.60`

**Champion by return:**
- Seed: `1`
- 2026 return: `+149.7%`
- Calmar: `+6.90`
- MDD: `-21.8%`
- Trades: `32`
- Win-rate: `78%`
- Thresholds: `LTH=0.55`, `STH=0.60`

**Champion by Calmar:**
- Seed: `4`
- 2026 return: `+23.4%`
- Calmar: `+13.85`
- MDD: `-1.7%`
- Win-rate: `100%`
- Thresholds: `LTH=0.75`, `STH=0.90`

**Сравнение с EXP-019 H3 MTF:**
- 2025 Optuna best: `+623.4%` vs `+777.5%` у H3
- 2026 max return: `+149.7%` vs `+56.8%` у H3
- 2026 mean return: `+43.8%` vs `+25.1%` у H3
- Profitable seeds: `30/30` vs `29/30` у H3
- Max Calmar: `+13.85` vs `+29.83` у H3

**Вывод:**
- H4 хуже H3 на validation 2025, но намного лучше на OOS 2026 по return и стабильности profitable seeds.
- H4 выглядит как более устойчивый боевой target для FLIP, особенно по return-champion.
- Для low-drawdown режима H3 Calmar still лучше, но H4 Calmar-champion тоже практичный: `+23.4%` при MDD `-1.7%`.

**Артефакты:**
- `transformer2602/runs/optuna_4h_flip/best_params_mtf_h4.json`
- `transformer2602/runs/optuna_4h_flip/best_params_mtf_h3.json`
- `transformer2602/runs/optuna_4h_flip/best_params_base_h3.json`
- `transformer2602/runs/optuna_4h_flip/prod4h_mtf_h4_train18_24_val25.db`
- `transformer2602/runs/champion_4h_mtf_h4_train18_24_val25_test26/champion_return.pt`
- `transformer2602/runs/champion_4h_mtf_h4_train18_24_val25_test26/champion_calmar.pt`
- `transformer2602/runs/champion_4h_mtf_h4_train18_24_val25_test26/leaderboard.json`

---

### EXP-021 — Top-return H3/H4 MTF модели: yearly threshold grids 2025/2026
**Дата:** 2026-06-17  
**Скрипт:** `eval_top_return_grids.py`  
**Статус:** готово.

**Идея:** взять две top-return модели:
- `H3 MTF champion_return.pt`
- `H4 MTF champion_return.pt`

И отдельно прогнать threshold grid на:
- `2025-01-01` → `2025-12-31`
- `2026-01-01` → current/latest data

**Данные на момент прогона:**
- `DOGEUSDT_4h`: `15235` bars
- `DOGEUSDT_1h`: `60901` bars
- `BTCUSDT_4h`: `18524` bars
- `BTCUSDT_1h`: `74035` bars
- `ETHUSDT_4h`: `18524` bars
- `ETHUSDT_1h`: `74035` bars
- MTF features: `266`
- Dead features dropped: `0`

**Команда:**

```bash
python eval_top_return_grids.py --device cuda --out_dir analysis_top_return_grids
```

#### Результаты grid by return

| Model | Period | Return | MDD | Trades | WR | LTH/STH |
|-------|--------|--------|-----|--------|----|---------|
| H3 MTF return | 2025 | `+58.5%` | `-18.5%` | `30` | `73.3%` | `0.60 / 0.70` |
| H3 MTF return | 2026 | `+54.4%` | `-30.5%` | `51` | `64.7%` | `0.50 / 0.50` |
| H4 MTF return | 2025 | `+58.8%` | `-75.4%` | `94` | `58.5%` | `0.50 / 0.50` |
| H4 MTF return | 2026 | `+149.7%` | `-21.8%` | `32` | `78.1%` | `0.55 / 0.60` |

#### Результаты grid by Calmar

| Model | Period | Return | Calmar | MDD | Trades | WR | LTH/STH |
|-------|--------|--------|--------|-----|--------|----|---------|
| H3 MTF return | 2025 | `+55.7%` | `+3.56` | `-15.7%` | `23` | `65.2%` | `0.60 / 0.75` |
| H3 MTF return | 2026 | `+41.7%` | `+9.41` | `-12.2%` | `21` | `71.4%` | `0.75 / 0.55` |
| H4 MTF return | 2025 | `+58.8%` | `+0.78` | `-75.4%` | `94` | `58.5%` | `0.50 / 0.50` |
| H4 MTF return | 2026 | `+104.9%` | `+32.28` | `-11.8%` | `30` | `73.3%` | `0.65 / 0.60` |

**Вывод:**
- H3 top-return модель заметно стабильнее на `2025`: примерно тот же return, но MDD намного ниже (`-18.5%` vs `-75.4%` у H4).
- H4 top-return модель резко сильнее на `2026`: `+149.7%` return by grid и `+104.9%` при Calmar-grid.
- H4 нельзя считать универсально лучше: на 2025 она ловит очень глубокую просадку.
- Для боевого кандидата логичнее смотреть:
  - H3 как более ровную модель по годам;
  - H4 как агрессивный 2026 regime-candidate;
  - ensemble/filter между H3 и H4, либо общий grid/режимный переключатель.

**Артефакты:**
- `analysis_top_return_grids/summary_top_return_grids_2025_2026.csv`
- `analysis_top_return_grids/summary_top_return_grids_2025_2026.json`

**Проверка после фикса симулятора:**
- Найден и исправлен риск в yearly eval/champion simulation: последняя открытая позиция должна закрываться на последнем баре выбранного периода, а не на последнем баре всего датасета.
- Перепрогон `eval_top_return_grids.py` в `analysis_top_return_grids_fixed/` подтвердил, что return/MDD по таблице практически не изменились; значит основная проблема не в этом баге, а в логике отбора моделей/seed/grid.
- Артефакты перепроверки:
  - `analysis_top_return_grids_fixed/summary_top_return_grids_2025_2026.csv`
  - `analysis_top_return_grids_fixed/summary_top_return_grids_2025_2026.json`

---

### EXP-022 — Honest selector: select seed/grid on 2025, fixed test on 2026
**Дата:** 2026-06-17  
**Скрипт:** `honest_selector_4h_mtf.py`  
**Статус:** готово.

**Идея:** проверить реальный production-like отбор:
- train: `2018-01-01` → `2024-12-31`
- selection/grid: `2025-01-01` → `2025-12-31`
- fixed test: `2026-01-01` → latest/current

Для каждой модели (`H3 MTF`, `H4 MTF`) обучаем `30 seeds`, на каждом seed подбираем threshold grid только по `2025`, затем эти же пороги применяем к `2026`.

**Selectors:**
- `max_return`: лучший return на 2025;
- `max_calmar`: лучший Calmar на 2025;
- `risk_adjusted`: `return + 5 * calmar + mdd`, с фильтрами `min_trades=10`, `max_mdd=35%`.

**Команда:**

```bash
python honest_selector_4h_mtf.py --n_seeds 30 --device cuda --out_dir analysis_honest_selector --min_trades 10 --max_mdd 35
```

#### Summary: fixed 2026 after 2025 selection

| Model | Selector | Best 2026 Return | Best 2026 MDD | Mean 2026 Return | Median 2026 Return | Min 2026 Return | Profitable Seeds |
|-------|----------|------------------|---------------|------------------|--------------------|-----------------|------------------|
| H3 MTF | max_return | `+28.9%` | `-15.9%` | `-0.5%` | `+0.6%` | `-30.0%` | `16/30` |
| H3 MTF | max_calmar | `+14.0%` | `-7.2%` | `+0.3%` | `+1.4%` | `-25.4%` | `17/30` |
| H3 MTF | risk_adjusted | `+28.9%` | `-15.9%` | `-0.3%` | `+1.4%` | `-30.0%` | `17/30` |
| H4 MTF | max_return | `+55.6%` | `-11.3%` | `+6.0%` | `+6.0%` | `-42.3%` | `18/30` |
| H4 MTF | max_calmar | `+47.1%` | `-29.6%` | `+6.2%` | `+7.7%` | `-42.3%` | `20/30` |
| H4 MTF | risk_adjusted | `+47.1%` | `-29.6%` | `+4.5%` | `+6.0%` | `-42.3%` | `18/30` |

**Вывод:**
- Красивые champion-цифры были результатом отбора по `2026` и/или переоптимизации; honest transfer `2025 -> 2026` намного слабее.
- `H3 MTF` почти не переносится: средний 2026 return около нуля, profitable seeds только `16-17/30`.
- `H4 MTF` переносится лучше, но всё равно нестабилен: средний 2026 return `+4.5%`–`+6.2%`, худшие seeds до `-42.3%`.
- Лучший отдельный H4 seed после честного отбора всё еще интересный (`+55.6%`, MDD `-11.3%`), но распределение слабое. Нельзя брать одиночный seed без ансамбля/дополнительного отбора.
- Следующий разумный шаг: не искать один seed, а проверить ансамбли top-N по 2025 или более строгий selector, который учитывает стабильность по месяцам 2025, MDD и минимум сделок.

**Артефакты:**
- `analysis_honest_selector/honest_selector_all_seeds.csv`
- `analysis_honest_selector/honest_selector_summary.csv`
- `analysis_honest_selector/honest_selector_all_seeds.json`
- `analysis_honest_selector/honest_selector_summary.json`

---

### EXP-023 - Robust Optuna 4H MTF H4 selector

**Дата:** 2026-06-18  
**Скрипт:** `transformer2602/optuna_4h_mtf_robust.py`  
**Статус:** запущен отдельный robust-прогон.

**Зачем:** предыдущий honest-selector показал, что одиночные модели нестабильны, а выбор по `max_return`/`max_calmar` на 2025 легко дает переоптимизм. Новый objective пытается отбирать не самый яркий seed, а параметры, которые дают устойчивое распределение по нескольким seeds.

**Схема:**
- train: `2018-01-01` -> `2024-12-31`
- objective/select: только `2025-01-01` -> `2025-12-31`
- holdout: `2026-01-01` -> `2026-12-31` не используется в Optuna
- target: `H4`
- feature set: `4h_mtf`
- стартовый прогон: `40 trials x 8 seeds`

**Objective:** для каждого seed ищем threshold-grid на 2025 не по чистому return, а по robust score:
- total return
- Calmar
- MDD penalty
- минимум сделок
- количество положительных месяцев
- worst-month penalty
- штраф за концентрацию прибыли в одном месяце
- neighbor-threshold median return как проверка стабильности вокруг выбранных thresholds

Итоговый trial-score агрегируется по seeds через median/mean/top-3, чтобы один удачный seed не доминировал.

#### Результат robust Optuna

Прогон завершился за `8h 2m`.

Финальный best trial:
- trial: `63`
- robust score: `371.7469`
- params: `d_model=128`, `nhead=8`, `num_layers=2`, `ff_mult=4`, `seq_len=100`, `dropout=0.15`, `lr=9.20e-05`, `weight_decay=9.64e-04`, `batch_size=128`, `epochs=50`, `patience=8`

В базе:
- `75` complete trials
- `1` stale/running trial от остановленного запуска
- фактически около `75 x 8 = 600` seed-обучений внутри robust objective

#### Honest 2026-check по robust params

**Команда:**
`python -u honest_selector_4h_mtf.py --param_file transformer2602/runs/optuna_4h_mtf_robust/best_params_mtf_h4_robust.json --model_name h4_mtf_robust --out_dir analysis_honest_selector_robust --n_seeds 30`

**Схема:** thresholds/selectors выбираются только на `2025`, затем фиксированно применяются на `2026`.

**Итоговая summary по 30 seeds:**

| selector | best 2026 seed | best test return | best MDD | best Calmar | mean test return | median test return | min test return | profitable seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max_calmar | 24 | `+45.42%` | `-19.44%` | `6.46` | `-8.96%` | `-5.43%` | `-63.98%` | `12/30` |
| max_return | 24 | `+45.42%` | `-19.44%` | `6.46` | `-8.07%` | `-7.10%` | `-63.98%` | `11/30` |
| risk_adjusted | 24 | `+45.42%` | `-19.44%` | `6.46` | `-8.26%` | `-7.10%` | `-63.98%` | `11/30` |

**Вывод:** robust Optuna нашла очень сильную 2025-зону, но честный перенос на 2026 всё ещё нестабилен. Лучший seed выглядит рабочим (`+45.4%`, MDD `-19.4%`), но распределение по seeds отрицательное: median и mean ниже нуля, profitable только `11-12/30`. Одиночную модель/seed по-прежнему нельзя считать надежной.

**Артефакты:**
- `transformer2602/optuna_4h_mtf_robust.py`
- `honest_selector_4h_mtf.py` теперь поддерживает `--param_file/--model_name` для честной проверки отдельного params-файла
- `transformer2602/runs/optuna_4h_mtf_robust/prod4h_mtf_h4_robust_v1.db`
- `transformer2602/runs/optuna_4h_mtf_robust/best_params_mtf_h4_robust.json`
- `analysis_honest_selector_robust/honest_selector_all_seeds.csv`
- `analysis_honest_selector_robust/honest_selector_summary.csv`
- `analysis_honest_selector_robust/honest_selector_all_seeds.json`
- `analysis_honest_selector_robust/honest_selector_summary.json`

---

### EXP-024 - LightGBM triple-barrier + purged walk-forward

**Дата:** 2026-06-18  
**Скрипт:** `exp024_lgbm_triple_barrier.py`  
**Статус:** первый quick-прогон завершён.

**Зачем:** проверить, есть ли trade-level edge в текущих 4H фичах без transformer и без выбора одного lucky seed. Target заменён с fixed horizon direction на triple-barrier событие.

**Схема:**
- features: базовый `build_features()` (`135` фичей, 4H index)
- labels: `0=short edge`, `1=neutral`, `2=long edge`
- entry: на `open` следующей 4H свечи после сигнала
- validation: quarterly walk-forward blocks `2024Q1 -> 2025Q4`
- purge: `max_hold + 2` баров между train и validation
- holdout: fixed `2026-01-01 -> 2026-12-31`
- model: `LightGBM multiclass`

**Quick grid:**
- `5` barrier configs
- `4` signal threshold configs
- `2` LightGBM configs
- всего `10` candidate model/barrier configs x `8` validation blocks

**Best CV config:**
- barrier: `TP 9.0%`, `SL 5.5%`, `max_hold=18` 4H bars
- signal: `entry_th=0.50`, `margin=0.00`
- LightGBM: `learning_rate=0.02`, `n_estimators=900`, `num_leaves=63`
- CV score: `344.37`
- median validation return: `+29.20%`
- mean validation return: `+30.52%`
- worst validation block: `-3.36%`
- profitable validation blocks: `7/8`
- median validation MDD: `-13.78%`

**Fixed holdout 2026 по выбранному CV config:**
- total return: `-5.44%`
- Calmar: `-0.58`
- MDD: `-19.76%`
- trades: `23`
- win rate: `39.1%`
- avg hold: `12.4` bars

**Вывод:** triple-barrier + LightGBM сильно лучше контролирует validation-блоки, чем одиночный transformer-seed, но первый fixed 2026 всё ещё отрицательный. Это не production-кандидат, но направление выглядит менее сломанным: MDD умеренный, 2026 около нуля, а не катастрофа. Следующий шаг - расширить grid/добавить более строгий objective с train только до 2024 для выбора параметров, либо сделать отдельный shock-filter/meta-label поверх этого сигнала.

**Артефакты:**
- `exp024_lgbm_triple_barrier.py`
- `analysis_exp024_lgbm_triple_barrier_quick/candidate_summary.csv`
- `analysis_exp024_lgbm_triple_barrier_quick/walkforward_blocks.csv`
- `analysis_exp024_lgbm_triple_barrier_quick/holdout_2026_summary.csv`
- `analysis_exp024_lgbm_triple_barrier_quick/holdout_2026_trades.csv`
- `analysis_exp024_lgbm_triple_barrier_quick/holdout_2026_equity.csv`
- `analysis_exp024_lgbm_triple_barrier_quick/result.json`

---

### EXP-025 - 1H Microtrend Model

**Дата:** 2026-06-18  
**Скрипт:** `exp025_microtrend_1h.py`  
**Статус:** quick-прогон завершён, гипотеза в текущем виде не прошла.

**Зачем:** проверить постановку, где модель предсказывает не fixed horizon, а микротренд:
- направление: `down / flat / up`
- качество/вероятность тренда: `good_trend`
- ожидаемую остаточную длительность: `duration_remaining`

**Target:** 1H ZigZag/ATR swing-labeling:
- adaptive threshold: `atr_mult * ATR_pct`, плюс `min_move`
- каждому бару внутри swing присваивается направление и сколько баров осталось до конца swing
- `good_trend` определяется через итоговый swing quality score и минимальную длительность

**Важно по утечке:** первый вариант использовал `ctx4h_*` из `feature_builder_1h_v2.py`, где 4H свеча forward-fill'илась на 1H index по open-time. Это дало нереалистичный результат (`+420%` на 2026, MDD `-3.3%`) и было признано утечкой. После этого EXP-025 был переделан:
- `ctx4h_*` удаляются из базового feature output
- 4H context пересобирается из завершённых 1H баров с `resample(..., label="right", closed="left")`
- daily context DOGE/BTC/ETH также пересобирается из 1H
- BTC dominance proxy пересобирается из 1H BTC/ETH
- запуск на RTX 5080 через LightGBM OpenCL: `gpu_platform_id=1`, `gpu_device_id=0`

**Quick grid:**
- `3` label configs
- `2` LightGBM configs
- `4` signal configs
- validation blocks: `2024Q1 -> 2025Q4`
- holdout: fixed `2026`

**Best CV config после исправления HTF-фичей:**
- label: `atr1p0_min0p012_q0p080_d4`
- signal: `dir_th=0.50`, `quality_th=0.55`, `min_duration=8`, `margin=0.06`
- LightGBM: `learning_rate=0.02`, `n_estimators=1000`, `num_leaves=63`, `min_child_samples=100`
- CV score: `23.64`
- median validation return: `+1.17%`
- mean validation return: `-0.81%`
- worst validation block: `-13.40%`
- profitable validation blocks: `4/8`
- median validation MDD: `-3.72%`

**Fixed holdout 2026:**
- total return: `-6.90%`
- Calmar: `-0.69`
- MDD: `-20.74%`
- trades: `318`
- win rate: `39.3%`
- avg hold: `2.59` bars

**Вывод:** сама идея microtrend-labeling реализована, но в текущем виде edge нет. Самое ценное из EXP-025 - поймали важную проблему выравнивания higher-timeframe фичей на 1H. Дальше любые 1H эксперименты должны строить 4H/1D context из 1H с timestamp доступности, а не forward-fill'ить готовые 4H свечи по open-time.

**Артефакты:**
- `exp025_microtrend_1h.py`
- `analysis_exp025_microtrend_1h_rtx5080_1hbuilt_quick/label_distribution.csv`
- `analysis_exp025_microtrend_1h_rtx5080_1hbuilt_quick/candidate_summary.csv`
- `analysis_exp025_microtrend_1h_rtx5080_1hbuilt_quick/walkforward_blocks.csv`
- `analysis_exp025_microtrend_1h_rtx5080_1hbuilt_quick/holdout_2026_summary.csv`
- `analysis_exp025_microtrend_1h_rtx5080_1hbuilt_quick/holdout_2026_trades.csv`
- `analysis_exp025_microtrend_1h_rtx5080_1hbuilt_quick/feature_importance.csv`
- `analysis_exp025_microtrend_1h_rtx5080_1hbuilt_quick/result.json`

---

### EXP-026 - PyTorch CUDA ZigZag phase / position target

**Дата:** 2026-06-18  
**Скрипт:** `exp026_torch_zigzag_phase.py`  
**Статус:** quick-прогон завершён.

**Зачем:** перенести идею ZigZag/phase/position target в PyTorch, чтобы использовать CUDA/RTX 5080, а не LightGBM OpenCL/CPU-side loops.

**Features:** те же no-leak 1H features, что в финальном EXP-025:
- базовый `build_features_1h_v2()`
- `ctx4h_*`, daily DOGE/BTC/ETH и BTC dominance proxy пересобираются из завершённых 1H баров с timestamp доступности
- experiment-local microtrend/trend features

**Target:**
- `direction`: `down / flat / up`
- `quality`: good microtrend flag
- `position`: плавная ideal exposure `[-1, +1]` по фазе ZigZag swing
- `phase`: `cos/sin` фазы swing
- `duration`: clipped remaining duration

**Model:**
- PyTorch Transformer encoder
- `seq_len=64`
- `d_model=128`
- `num_layers=2`
- `dropout=0.15`
- multi-head outputs: direction / quality / position / phase / duration
- device: `cuda`

**Quick grid:**
- labels: `2`
- torch configs: `1`
- signal configs: `3`
- validation: `2024Q1 -> 2025Q4`
- holdout: fixed `2026`

**Best CV config:**
- label: `atr1p0_min0p012_q0p080_d4`
- signal: `pos_th=0.15`, `quality_th=0.55`, `max_abs_pos=0.8`
- CV score: `-3.13`
- median validation return: `-0.78%`
- mean validation return: `-0.43%`
- worst validation block: `-3.12%`
- profitable validation blocks: `3/8`
- median validation MDD: `-3.87%`

**Fixed holdout 2026:**
- total return: `-3.38%`
- Calmar: `-1.81`
- MDD: `-3.99%`
- trades: `113`
- win rate: `35.4%`

**Вывод:** CUDA sequence model с position target пока не даёт положительного edge, но риск-профиль намного мягче, чем у EXP-025 LightGBM (`MDD -4%` против `-20.7%`). Это выглядит как более безопасная основа для следующих итераций: нужно улучшать target/position curve и selection objective, а не возвращаться к fixed horizon direction.

**Артефакты:**
- `exp026_torch_zigzag_phase.py`
- `analysis_exp026_torch_zigzag_phase_quick/candidate_summary.csv`
- `analysis_exp026_torch_zigzag_phase_quick/walkforward_blocks.csv`
- `analysis_exp026_torch_zigzag_phase_quick/holdout_2026_summary.csv`
- `analysis_exp026_torch_zigzag_phase_quick/holdout_2026_trades.csv`
- `analysis_exp026_torch_zigzag_phase_quick/holdout_2026_equity.csv`
- `analysis_exp026_torch_zigzag_phase_quick/final_model.pt`
- `analysis_exp026_torch_zigzag_phase_quick/result.json`

---

### EXP-027 - Optuna для EXP-026 Torch ZigZag phase

**Дата:** 2026-06-18  
**Скрипт:** `optuna_exp026_torch_zigzag_phase.py`  
**Статус:** запущен CUDA Optuna-прогон.

**Зачем:** EXP-026 quick показал низкий риск, но отрицательный return. Новый прогон оптимизирует:
- ZigZag label params
- Torch sequence params
- signal params для continuous position

**Важно:** `2026` не используется в Optuna. Objective считается только на quarterly walk-forward блоках `2024Q1 -> 2025Q4`.

**Стартовая команда:**
`python -u optuna_exp026_torch_zigzag_phase.py --n_trials 20 --device cuda --study exp026_torch_zigzag_phase_v1`

**Поисковое пространство:**
- label: `atr_mult`, `min_move`, `max_duration_clip`, `quality_ret`, `quality_min_duration`
- torch: `seq_len`, `d_model`, `nhead`, `num_layers`, `dropout`, `batch_size`, `lr`, `weight_decay`, `epochs`, `patience`
- signal: `pos_th`, `quality_th`, `max_abs_pos`

**Артефакты:**
- `optuna_exp026_torch_zigzag_phase.py`
- `transformer2602/runs/optuna_exp026_torch_zigzag_phase/exp026_torch_zigzag_phase_v1.db`
- `transformer2602/runs/optuna_exp026_torch_zigzag_phase/best_params_exp026_torch_zigzag_phase.json`
- `analysis_exp026_torch_zigzag_phase_optuna/optuna_trials.csv`

---

## EXP-028: Direct Position Torch (CUDA, quick)

**Дата:** 2026-06-18  
**Скрипт:** `exp028_torch_direct_position.py`  
**Статус:** quick-прогон завершён (~6 мин, CUDA).

**Гипотеза:** упростить задачу до одного target `optimal_position ∈ [-1, 1]`, построенного напрямую из ZigZag swing (remaining move / adverse risk), без phase/duration/quality heads. Модель учит ровно то, что потом торгуется.

**Метод:**
- no-leak 1h features (167) + microtrend features
- single-head Transformer regressor, Huber loss
- walk-forward CV: 8 quarterly blocks 2024–2025
- holdout: train ≤2025, fixed test 2026

**Лучший CV-кандидат:**
- label: `atr1.5 / min0.016 / q0.11 / d6`
- signal: `pos_th=0.08, max_abs_pos=0.7, smooth=0.2`
- CV score: **+36.07**
- median return/block: **+0.32%**, profitable blocks: **5/8**, median MDD: **-0.84%**

**Holdout 2026 (честный):**
| metric | EXP-028 | EXP-026 |
|---|---|---|
| total_return | **+0.21%** | -3.38% |
| MDD | **-0.18%** | -3.99% |
| n_trades | **1** | 113 |
| win_rate | 100% | 35% |

**Вывод:** постановка стала чище — CV в плюсе, 2026 не в минусе (в отличие от EXP-026). Но edge пока статистически нулевой: за 2026 YTD всего **1 сделка** (+0.22% за 9 часов long 4 января). Модель почти всегда сидит в нуле — консервативна, но не торгует.

**Следующий шаг:** Optuna по label/signal/torch params + проверить, не слишком ли агрессивны `pos_th=0.08` и late-entry penalty в target (модель может учиться «не торговать»).

**Артефакты:**
- `exp028_torch_direct_position.py`
- `analysis_exp028_torch_direct_position/`
- `analysis_exp028_torch_direct_position/result.json`

---

## EXP-029: Optuna EXP-028 Direct Position (CUDA)

**Дата:** 2026-06-18  
**Скрипт:** `optuna_exp028_torch_direct_position.py`  
**Статус:** остановлен (v1 Calmar-first, v2 return-first ~9/20 trials).

**Зачем:** EXP-028 quick показал хороший CV и нейтральный 2026 (+0.21%), но всего 1 сделка — модель слишком консервативна. Optuna искал params с большей активностью без потери стабильности.

**Objective (v2, return-first):** `3×median_ret + 1.5×mean_ret + 2.5×min_ret + 2×prof_blocks + 0.5×mdd + 1×calmar(capped)`. Calmar вторичен, доходность главная. Штраф за `median_trades < 3`.

**Objective v1 (отменён):** Calmar×20 доминировал, доходность почти не влияла.

**Итог:** лучший trial v2 — score +12.65, median +0.85%/квартал, min -0.99%. Edge нет. Forward-window target — отклонён. ZigZag hindsight — не переносится на inference.

**Поиск:** label (ZigZag), torch (seq/model), signal (`pos_th`, `max_abs_pos`, `smooth`).

**Команда:**
`python -u optuna_exp028_torch_direct_position.py --n_trials 20 --device cuda --study exp028_torch_direct_position_v2`

**Артефакты:**
- `optuna_exp028_torch_direct_position.py`
- `transformer2602/runs/optuna_exp028_torch_direct_position/exp028_torch_direct_position_v1.db`
- `transformer2602/runs/optuna_exp028_torch_direct_position/best_params_exp028_torch_direct_position.json`
- `analysis_exp028_torch_direct_position_optuna/optuna_trials.csv`

---

## EXP-030: Rule-based 3D microtrend (no ML)

**Дата:** 2026-06-18  
**Спека:** `MICROTREND_TARGET.md`  
**Код:** `microtrend_target.py`, `exp030_rule_microtrend.py`

**Формула:** `T=(dir,q,φ)`, always in market, `p = dir · q · g(φ)`, `g = size_min + (1-size_min)(1-φ)^γ`.

**Quick CV 2024-2025:** best score **-175**, median **-13.7%/block**, profitable **1/8**, ~152 trades/block.

**Holdout 2026:** **-10.9%**, MDD **-26.7%**, **271 trades**, WR 31%.

**Вывод:** rule-based always-in-market без edge = churn + fees. Baseline не проходит. Перед ML нужно: сильнее chop-filter (q→size_min только при q>floor), flip cooldown, или не carry dir в search-state.

**Артефакты:** `analysis_exp030_rule_microtrend/`

---

## EXP-031…033: Strategy ensemble / AI-trader / small torch (кратко)

**Дата:** 2026-06 → 2026-08  
**Статус:** отклонены как путь к edge.

| EXP | Идея | Честный итог |
|---|---|---|
| 031 | ML ensemble стратегий | 2025/2026 глубоко в минусе (тысячи сделок → churn) |
| 032 | AI-trader (supervised) | 2025 −7%, 2026 −18% |
| 033 | Small torch trader | 2025 −32%, 2026 −20% |

**Вывод:** фиксированный horizon / supervised direction без equity-симуляции не даёт устойчивого edge на DOGE.

**Артефакты:** `analysis_exp031_*`, `analysis_exp032_*`, `analysis_exp033_*`

---

## EXP-034: Policy GRU (continuous position, equity loss)

**Дата:** ~2026-08  
**Скрипт:** `exp034_policy_trader.py`  
**Статус:** отклонён (shortcut / buy-and-hold bias).

**Гипотеза:** учить continuous position `[-1,1]` напрямую по equity + MDD, без fixed-horizon labels.

**Split:** train ≤2025-06-30 / val H2’25 / test 2026.

**Итог:**
- val H2’25: **+3.4%**, MDD −30%, 77 trades
- test 2026: **−15.2%**, MDD −30%, 79 trades, WR 35%

**Вывод:** equity-loss на continuous позиции легко схлопывается в near-buy-and-hold / слабый edge; 2026 один как sole test плох (Trump/фундаментал). Нужен агент с явными входами/выходами (PPO) и финальный тест **H2’25+2026**.

**Артефакты:** `analysis_exp034_policy_trader/`

---

## EXP-035: Long/Short direction stack + meta

**Дата:** ~2026-08  
**Скрипт:** `exp035_stack_long_short_meta.py`  
**Статус:** отклонён (неверная метрика базы).

**Гипотеза:** отдельные long/short direction модели → meta по return.

**Критическая поправка:** base-модели оптимизировались/отбирались по **direction winrate**, а не по edge/return после комиссий. WR ~coin-flip на 2026 (long th0.45 WR~48%; short th0.55 WR~59% на редких сигналах). Meta IC≈0 / dir_acc≈50%.

**Вывод:** stack на coin-flip direction не спасает. Дальше — self-trading PPO по equity, не direction labels.

**Артефакты:** `analysis_exp035_stack_long_short/`

---

## EXP-036…039: PPO на DOGE-only 15m (диагностика shortcut’ов)

**Дата:** ~2026-08  
**Данные:** `datasets/doge_15m_self_trade/` (DOGE-only, ~120 фич, без BTC/ETH)  
**Скрипты:** `exp036_ppo_trader_15m.py`, `exp037_ppo_discrete_15m.py`, `exp038_ppo_exposure_budget.py`, `exp039_ppo_max_return.py`

### Что пробовали
| EXP | Постановка | Val | Test 2026 | Диагноз |
|---|---|---:|---:|---|
| 036 | continuous PPO | −16% | −24% | слабый сигнал vs noise |
| 037 | discrete {S,F,L} | −22% | −60% | almost-always-long (407L/11S) |
| 038 | exposure budget + MDD≤25% | +30%* | −33% | *val MDD всё равно −33%; exploit: short first ~40% окна |
| 039 | max return, no risk | −32% | −44% | короткий hold (~3 бара), WR↑ / expectancy↓ (хвосты+fees) |

### Найденные баги / ловушки PPO
1. **Absurd turnover_lambda** (EXP-036) → агент учится «не торговать».
2. **Train path drift** до Jun’25 сильно бычий → always-long shortcut, даже если bar-WR ~50%.
3. **Exposure budget reset** каждые N дней → «short в начале окна» exploit.
4. **Hysteresis вне sample** ломает PPO likelihood (sample ≠ execute).
5. **Trade PnL reporting** неполный vs equity+fees.
6. **MDD constraint** в objective не гарантирует MDD на val/test.

**Вывод линии 15m:** PPO на 15m без жёсткого контроля churn либо всегда-long, либо short-hold churn. Нужен исправленный engine + другой timescale.

**Артефакты:** `analysis_exp036_*` … `analysis_exp039_*`, `feature_builder_doge_15m.py`

---

## EXP-040: GRU PPO, решения раз в 1h на 15m барах

**Дата:** ~2026-08  
**Скрипты:** `exp040_gru_ppo_1h.py`, `optuna_exp040_gru_ppo_1h.py`  
**Статус:** структурно лучше 036–039, экономически не прошло; Optuna нестабильна по seeds.

**Постановка:** GRU, `decision_every=4` (1h), hysteresis, honest fees, regime windows bull/bear/chop.

**Итог (single run):**
- val: **+25.7%**, MDD −40%, 318 trades, WR 59%, PF 1.16
- test 2026: **−30.0%**, MDD −41%, 354 trades, WR 54%, PF 0.91 (buy&hold −40%)

**Optuna:** best trial score ~+19.7 на in-sample блоках; seeds внутри trial расходятся (один seed median +76%, другой −36%) → нестабильность, не брать Optuna пока engine/protocol не твердые.

**Вывод:** реже решать — лучше структура сделок, но 2026 всё ещё минус; hysteresis + selection protocol требуют пересмотра.

**Артефакты:** `analysis_exp040_gru_ppo_1h/`, `analysis_exp040_gru_ppo_1h_optuna/`

---

## EXP-041: PPO v2 HOLD/FLAT/LONG/SHORT (15m data)

**Дата:** ~2026-08  
**Скрипт:** `exp041_ppo_v2_hold.py`  
**Статус:** экономически fail; подозрение на engine → ушло в валидацию engine.

**Фиксы протокола:**
- явный HOLD; без hysteresis; action == executed
- entropy anneal; checkpoint каждый update
- select на **2025H1 внутри train**; final = **H2’25 + 2026**
- multi-seed ensemble

**Итог final (H2’25+2026):**
| model | return | MDD | trades | fee≈ |
|---|---:|---:|---:|---:|
| best_seed (76) | **−53.9%** | −77% | 261 | 13% |
| ensemble | **−35%** | −44% | 457 | 26% |

Select на 2025H1 у seeds 59/76 был **+70%…+72%** → сильный overfit select→final.

**Вывод:** протокол честнее, но результат плохой; нельзя хоронить PPO, пока не доказано, что accounting/masks/reward корректны.

**Артефакты:** `analysis_exp041_ppo_v2_hold/`

---

## PPO engine v2 — валидация (до DOGE)

**Дата:** 2026-08-09  
**Код:** `ppo_engine_v2.py`, `validate_ppo_engine_v2.py`  
**Статус:** **PASS** — engine OK.

| Проверка | Результат |
|---|---|
| Action masks (FLAT/LONG/SHORT states) | PASS |
| `sum(rewards) ≈ log(final_equity)` | PASS (~1e-12) |
| Invalid action rejected | PASS |
| Toy predictable market, 5/5 seeds | PASS (dir acc 96–100%) |

**Вывод:** проблема DOGE-smoke — **не** сломанный accounting. На реальном рынке сигнал слабее commission noise при высокой частоте решений.

**Артефакты:** `analysis_ppo_engine_v2_validation/`

---

## EXP-042: DOGE 15m smoke на validated engine

**Дата:** 2026-08-09  
**Скрипт:** `exp042_ppo_engine_v2_doge_smoke.py`  
**Статус:** экономический fail (ожидаемо после валидации).

**Протокол:** roll≤2024 / select 2025H1 / final H2’25+2026; 3 seeds; fee 4 bps/side.

| seed | select | final | turnover | fee-drag≈ |
|---:|---:|---:|---:|---:|
| 42 | −22% | **−98.6%** | 7116 | 94% |
| 59 | −29% | **−87.6%** | 4658 | 84% |
| 76 | −52% | **−94.4%** | 7426 | 95% |

Median final **−94%**, 0/3 profitable. Accounting error ~1e-12.

**Вывод:** на 15m churn убивает раньше edge. Не хоронить PPO — сменить timescale на **1h**.

**Артефакты:** `analysis_exp042_ppo_engine_v2_smoke/`

---

## EXP-043: DOGE 1h smoke на validated engine

**Дата:** 2026-08-09  
**Скрипты:** `feature_builder_doge_1h.py`, `exp043_ppo_1h_smoke.py`  
**Данные:** `datasets/doge_1h_self_trade/` — 61 865 баров, 120 DOGE-only фич, 2019-07 → 2026-08-09  
**Статус:** лучше 15m структурно; final всё ещё минус; select впервые стабильно в плюсе.

**Постановка:** тот же `ppo_engine_v2`; `decision_every=1` (каждый 1h бар); episode 720h; 20 updates × 3 seeds; fee 4 bps.

| seed | best select (2025H1) | final (H2’25+2026) | hold | turnover | fee-drag≈ |
|---:|---:|---:|---:|---:|---:|
| 42 | **+10.2%** (u07) | **−50.2%** | 0.86 | 1804 | 51% |
| 59 | **+20.1%** (u13) | **−73.3%** | 0.86 | 1940 | 54% |
| 76 | −3.6% (u14) | **−70.4%** | 0.75 | 2518 | 64% |

- Median final **−70%**, 0/3 profitable
- vs EXP-042: turnover ~4× ниже, fee-drag ~50–63% вместо ~85–95%, hold 75–86% вместо 33–55%
- Select→final gap огромный (overfit / нестойкость политики)

**Вывод:**
1. Переход 15m→1h **правильный**: меньше fee death, появляются положительные select-checkpoints.
2. Edge на final всё ещё нет: churn ~1.8–2.5k оборотов за ~9.7k часов всё ещё слишком частый.
3. PPO как идея жива; следующий рычаг — **давить частоту смен** (turnover penalty / cooldown / решать реже часа), не Optuna.
4. Live FLIP (редкие перевороты, длинные холды) — ориентир формы сделок, **не** вход PPO-фич.

**Артефакты:**
- `datasets/doge_1h_self_trade/`
- `feature_builder_doge_1h.py`
- `analysis_exp043_ppo_1h_smoke/`

---

## EXP-044: PPO 1h low-turnover (4h decisions + masked min-hold)

**Дата:** 2026-08-09  
**Скрипт:** `exp044_ppo_1h_low_turnover.py`  
**Engine:** `ppo_engine_v2.py` — добавлен `min_hold_bars`, ограничение реализовано внутри action mask (sample == execute)  
**Статус:** churn существенно снижен, но устойчивого edge нет.

**Постановка:**
- DOGE-only 1h, 120 фич
- решение раз в **4h**
- две конфигурации: min-hold **12h** и **24h**
- 20 updates, 5 seeds (`42,59,76,93,110`)
- checkpoint score: `median(block returns) + 0.5 × min(block returns)`
- блоки selection/stability: 2023H2, 2024H1, 2024H2, 2025H1
- final untouched: H2’25 + 2026; отдельно H2’25 и 2026
- baselines: FLAT, always-long, always-short, 24h momentum

**Важно о selection:** 2023H2–2024H2 входят в train roll и являются только диагностикой устойчивости checkpoint; 2025H1 — единственный полностью forward select-block. Это не заменяет полноценный walk-forward.

### Final H2’25 + 2026

| min-hold | median | mean | range | profitable | median turnover | fee-drag≈ |
|---|---:|---:|---:|---:|---:|---:|
| 12h | **−60.4%** | −52.9% | −80.0…−11.1% | 0/5 | 1088 | 35.3% |
| 24h | **−40.2%** | −17.4% | −67.2…+98.8% | 1/5 | 506 | 18.3% |

**24h seed details (final):**
- seed42: **+98.8%**, MDD −50.3%, turnover 684
- seed59: −67.2%
- seed76: −24.0%
- seed93: −40.2%
- seed110: −54.4%

Seed42 не является доказательством edge: 1/5 profitable, MDD −50%, а один selection block у выбранного checkpoint был −43%.

### Baselines (final H2’25 + 2026)

| baseline | return | MDD | turnover |
|---|---:|---:|---:|
| FLAT | 0.0% | 0.0% | 0 |
| always-long | −57.7% | −76.8% | 2 |
| always-short | **+26.7%** | −54.5% | 2 |
| momentum 24h + min-hold 12h | **+47.5%** | −54.6% | 662 |
| momentum 24h + min-hold 24h | −2.3% | −49.3% | 538 |

### Вывод

1. Masked cooldown работает корректно: accounting error ~1e-12, likelihood не ломается.
2. 4h decisions + 24h hold снизили median turnover **1800–2500 → 506** и fee drag **51–64% → 18%** против EXP-043.
3. Снижение churn улучшает результат, но **не создаёт устойчивый edge**: median −40%, 1/5 profitable.
4. Простые always-short и 12h momentum baselines лучше median PPO; часть результата объясняется медвежьим final-периодом.
5. Selection score не предсказывает final: seeds с красивыми train/select блоками всё равно часто проваливаются.
6. Следующий эксперимент должен отделять signal от execution: полноценный walk-forward, одинаковый low-turnover execution для PPO и momentum, ensemble/seed-consensus только после доказанного OOS edge.

**Артефакты:**
- `analysis_exp044_ppo_1h_low_turnover/summary.json`
- `analysis_exp044_ppo_1h_low_turnover/seed_results.csv`
- `analysis_exp044_ppo_1h_low_turnover/baselines.csv`
- `analysis_exp044_ppo_1h_low_turnover/training_history.csv`

---

## EXP-045: Honest walk-forward PPO 1h vs momentum

**Дата:** 2026-08-09  
**Скрипт:** `exp045_ppo_1h_walk_forward.py`  
**Статус:** текущая PPO-линия **не прошла** honest walk-forward.

**Постановка:**
- 5 последовательных half-year OOS folds: 2024H2, 2025H1, 2025H2, 2026H1, 2026H2 YTD
- для каждого fold свежая модель обучается только на данных до начала fold
- checkpoint выбирается по трём предыдущим полугодиям; evaluated fold не виден ни train, ни selection
- 3 фиксированных seed; 16 updates
- одинаковый execution для PPO и momentum: решения 4h, masked min-hold 24h, fee 4 bps/side

### PPO OOS по folds

| Fold | median return | range seeds | profitable | median MDD | median turnover |
|---|---:|---:|---:|---:|---:|
| 2024H2 | **−54.3%** | −69.4…−43.2% | 0/3 | −71.8% | 218 |
| 2025H1 | **+18.2%** | −24.5…+102.7% | 2/3 | −53.7% | 222 |
| 2025H2 | **−42.7%** | −50.4…+44.5% | 1/3 | −54.6% | 240 |
| 2026H1 | **−12.6%** | −55.4…+9.5% | 1/3 | −42.7% | 278 |
| 2026H2 YTD | **+2.9%** | −1.0…+17.2% | 2/3 | −7.9% | 58 |

Итого: прибыльны только **6/15** независимых fold×seed результатов.

### Stitched OOS

| model | all folds | final H2’25+2026 | worst fold MDD | turnover |
|---|---:|---:|---:|---:|
| PPO seed42 | −76.5% | −74.7% | −71.8% | 880 |
| PPO seed59 | −70.1% | −55.4% | −69.5% | 962 |
| PPO seed76 | −57.2% | +85.4% | −79.8% | 1108 |
| momentum 24h | **+150.3%** | −3.0% | −57.1% | 1024 |
| always-long | −44.6% | −57.8% | −68.1% | 10 |
| always-short | −68.1% | +26.9% | −88.6% | 10 |
| FLAT | 0% | 0% | 0% | 0 |

Seed76 нельзя выбирать post-hoc по final: до просмотра OOS нет основания предпочесть его seeds 42/59. Более того, его all-fold result −57% и worst-fold MDD −80%.

**Momentum по folds:** +34.5%, +91.8%, +29.2%, −28.2%, +4.6% — прибыльно 4/5 folds и значительно стабильнее PPO, но MDD всё ещё слишком большой.

### Вывод

1. После честного walk-forward исчезает иллюзия, что редкий сильный seed = edge.
2. Checkpoint selection переносится слабо: положительный selection score часто соседствует с большим OOS-минусом.
3. Low-turnover execution уже достигнут (обычно 200–300 turnover/fold), поэтому текущий провал нельзя списать только на комиссии.
4. **Текущие DOGE-only фичи + GRU PPO reward не дают устойчивого торгуемого edge.**
5. Optuna и дальнейший выбор seed по final запрещены: это только усилит overfit.
6. Momentum содержит более устойчивый простой signal, но пока не deployable из-за MDD; дальнейшая работа логичнее вокруг causal momentum/regime/risk layer, а не текущего PPO.

**Артефакты:**
- `analysis_exp045_ppo_1h_walk_forward/summary.json`
- `analysis_exp045_ppo_1h_walk_forward/fold_seed_results.csv`
- `analysis_exp045_ppo_1h_walk_forward/fold_baselines.csv`
- `analysis_exp045_ppo_1h_walk_forward/stitched_results.csv`
- `analysis_exp045_ppo_1h_walk_forward/training_history.csv`

---

## EXP-046: ML-Risk Filter для Causal Momentum (LightGBM)

**Дата:** 2026-08-10  
**Скрипт:** `exp046_csmom_ml_risk.py`  
**Статус:** диагностический NO-GO; первоначальный вывод об успехе был преждевременным.

**Зачем:** Предыдущие эксперименты показали, что Momentum имеет edge, но страдает от огромных просадок (MDD > 50%). Прямое предсказание направления (Direction ML) и PPO провалились. Новая гипотеза: использовать ML **только как фильтр риска/режима**, предсказывая вероятность сильного падения рынка (BTC), и модулировать размер позиции Momentum стратегии.

**Постановка:**
- **База:** `SlowCrossSectionalMomentumExpert` (168h, 336h, 720h) на `HISTORICAL_CORE`.
- **Таргет ML:** Бинарная классификация. `1`, если BTC упадет более чем на 5% в следующие 24 часа (`forward_min_ret_24h < -0.05`), иначе `0`.
- **Фичи ML:** Простые макро-фичи на 1h барах: BTC returns (1h, 4h, 24h, 168h), BTC vol (24h, 168h), ETH returns/vol, кросс-секционная дисперсия, средний фандинг.
- **Модель:** `LightGBMClassifier` (Train < 2025, Val 2025H1).
- **Интеграция:** `risk_multiplier = clip(1.0 - risk_prob_smoothed * 2, 0, 1)`. Если ML видит высокий риск (>50%), позиция Momentum обнуляется.
- **Бэктест:** один статический model fit и последующий прогон по `DEVELOPMENT_FOLDS`; это не полноценный fold-local walk-forward.

**Результаты (Walk-Forward на DEVELOPMENT_FOLDS):**

| Стратегия | Return | MDD |
|---|---:|---:|
| Baseline CSMOM | -1.29% | -34.50% |
| ML-Risk Modulated CSMOM | **+0.28%** | **-17.11%** |

**Топ-5 важных фичей (Feature Importance):**
1. `btc_vol_168h` (313) — 7-дневная волатильность BTC
2. `mean_funding` (264) — Средний фандинг по рынку
3. `btc_ret_168h` (209) — 7-дневная доходность BTC
4. `eth_vol_24h` (163) — 24-часовая волатильность ETH
5. `btc_vol_24h` (109) — 24-часовая волатильность BTC

**Исправленный вывод:**
1. Результат остаётся `NO-GO`: PF `1.001`, Sharpe `0.073`, CAGR около `0.13%`; три критерия протокола не выполнены.
2. `2024H2` входит в train, а `2025H1` использован для early stopping, поэтому общий stitched result нельзя считать полностью OOS.
3. Средняя gross exposure снизилась примерно `0.79 → 0.45`; без risk-matched baseline нельзя приписывать двукратное снижение MDD качеству ML-фильтра.
4. Feature importance не доказывает направление зависимости и не подтверждает тезис, что сочетание высокого funding и volatility предсказывает падение.
5. Гипотеза risk/meta-filter остаётся рабочей, но EXP-046 является только диагностикой и не подтверждает edge.

**Следующие шаги:**
- Расширить ML-фильтр: добавить больше макро-фичей (CVD, ликвидации).
- Применить этот же фильтр к `FundingCarryExpert` и посмотреть, спасет ли он Carry.
- Сформировать финальный портфель (Modulated CSMOM + Modulated Carry).

---

## EXP-047: Fold-local ML meta-label поверх DOGE momentum 24h

**Дата preregistration:** 2026-08-10  
**Статус:** завершён; `NO-GO`.

**Гипотеза:** momentum 24h из EXP-045 содержит OOS signal, но его MDD слишком велик. LightGBM должен не предсказывать направление рынка, а оценивать качество уже сформированной momentum-сделки и разрешать/масштабировать её.

**Замороженная постановка:**
- данные и causal execution полностью наследуются от EXP-045: DOGEUSDT 1h, решение каждые 4h, сигнал на `open[t]`, исполнение на `open[t+1]`, minimum hold 24h, fee `4 bps/side`;
- baseline direction: `sign(open[t] / open[t-24] - 1)`;
- label return: net side-adjusted return от `open[t+1]` до `open[t+25]` после round-trip fee;
- tail label: adverse side-adjusted excursion хуже `-5%` внутри следующих 24h;
- отдельные LightGBM return/tail модели для LONG и SHORT;
- features: только доступные в `t` DOGE-only признаки плюс momentum `12/24/72/168h`, их согласованность и side-adjusted значения;
- purge: label horizon полностью заканчивается до train cutoff;
- для каждого OOS fold обучается новая модель только на прошлом;
- один из трёх заранее заданных sizing-профилей (`loose`, `balanced`, `strict`) выбирается по трём предыдущим cross-fitted half-year блокам; evaluated fold не участвует в выборе;
- обязательные сравнения: full-size momentum и constant-size momentum с той же средней OOS exposure, что у meta-filter.

**Критерий полезности фильтра:**
- положительный stitched OOS return после costs;
- не менее `4/5` положительных folds;
- улучшение против exposure-matched baseline, а не только против full-size baseline;
- MDD не хуже `-25%`;
- отсутствие fold хуже `-20%`.

### Результаты

| Strategy | Stitched return | Stitched MDD | Positive folds | Worst fold | Mean exposure | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| Momentum 24h full-size | `+150.34%` | `-57.08%` | `4/5` | `-28.20%` | `1.00` | `1024` |
| ML meta-momentum | `+31.22%` | `-43.58%` | `3/5` | `-8.71%` | `0.42` | `531` |
| Constant-size risk match | `+86.90%` | `-31.97%` | `4/5` | `-10.18%` | `0.42` | `428.7` |

**ML по OOS folds:**
- `2024H2`: `+50.01%`, MDD `-17.9%`;
- `2025H1`: `+0.67%`, MDD `-36.4%`;
- `2025H2`: `-6.57%`, MDD `-24.0%`;
- `2026H1`: `-8.71%`, MDD `-33.2%`;
- `2026H2 YTD`: `+1.87%`, MDD `-5.2%`.

**Вывод:**
1. Meta-label уменьшил худший fold-loss и turnover, но пропустил слишком большую часть положительного momentum edge.
2. При одинаковой средней exposure обычное постоянное уменьшение позиции существенно лучше ML: `+86.9%` против `+31.2%`, MDD `-32.0%` против `-43.6%`, `4/5` против `3/5` положительных folds.
3. Фильтр не прошёл три из пяти preregistered критериев; прошли только положительный stitched return и отсутствие fold хуже `-20%`.
4. Текущий trade-quality target (`24h net return + 5% adverse tail`) не переносится достаточно хорошо между режимами.
5. Практический результат EXP-047 — не ML, а подтверждение, что простой risk scaling momentum сохраняет больше edge. Следующий кандидат должен начинаться с portfolio-level volatility targeting / trend diversification, а не с нового classifier grid.

**Артефакты:**
- `exp047_momentum_meta_label.py`
- `tests/test_exp047_momentum_meta_label.py`
- `analysis_exp047_momentum_meta_label/summary.json`
- `analysis_exp047_momentum_meta_label/fold_results.csv`
- `analysis_exp047_momentum_meta_label/selection_results.csv`

---

## EXP-048: Multi-horizon momentum + causal volatility targeting

**Дата preregistration:** 2026-08-10  
**Статус:** завершён; alpha подтверждён, deployment `NO-GO` из-за MDD.

**Гипотеза:** часть просадки momentum 24h вызвана зависимостью от одного горизонта и постоянным размером позиции. Согласование нескольких горизонтов и обратное масштабирование по известной в момент решения волатильности должны улучшить return/MDD относительно простого constant-size risk scaling.

**Замороженная постановка:**
- DOGEUSDT 1h и execution из EXP-045/047: решение каждые 4h, исполнение на `open[t+1]`, minimum hold 24h, fee `4 bps/side`;
- causal momentum строится только по `open[t]` и прошлым значениям;
- signal families: `24h`, среднее знаков `24/72h`, `24/72/168h`, `12/24/72/168h`;
- causal realized volatility: rolling `168h` standard deviation часовых open-to-open returns, annualization `sqrt(8760)`;
- volatility targets: `35%` и `50%` annualized; position cap `1.0`;
- candidate target = средний знак momentum horizons × `min(1, target_vol / trailing_vol)`;
- один из восьми заранее заданных signal×vol candidates выбирается отдельно для каждого OOS fold только по трём предыдущим half-year блокам;
- frozen selection score: `median(block returns) + 0.5 × min(block returns)`;
- evaluated fold не участвует в выборе;
- обязательные baselines: full-size momentum 24h и constant-size momentum 24h с той же средней exposure, что выбранный dynamic candidate.

**Критерии GO:**
- положительный stitched OOS return;
- минимум `4/5` положительных folds;
- return выше exposure-matched baseline;
- stitched MDD не хуже `-25%`;
- отсутствие fold хуже `-20%`.

### Результаты

| Strategy | Stitched return | Stitched MDD | Positive folds | Worst fold | Mean exposure | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| Momentum 24h full-size | `+150.34%` | `-57.08%` | `4/5` | `-28.20%` | `1.00` | `1024` |
| Multi-horizon + vol target | **`+366.67%`** | `-41.53%` | **`5/5`** | **`+11.53%`** | `0.68` | `515.4` |
| Constant-size risk match | `+84.38%` | `-40.63%` | `4/5` | `-21.45%` | `0.68` | `637.4` |

**Выбранные без просмотра evaluated fold конфигурации и OOS:**
- `2024H2`: `mom24_72_vol50` → `+121.20%`, MDD `-29.8%`;
- `2025H1`: `mom24_72_vol50` → `+31.66%`, MDD `-19.3%`;
- `2025H2`: `mom24_vol50` → `+17.76%`, MDD `-24.2%`;
- `2026H1`: `mom24_vol50` → `+22.00%`, MDD `-41.5%`;
- `2026H2 YTD`: `mom24_vol50` → `+11.53%`, MDD `-7.7%`.

**Решение:**
1. Пройдены `4/5` preregistered критериев: положительный stitched return, `5/5` положительных folds, превосходство над exposure-matched baseline и отсутствие отрицательного fold.
2. Не пройден единственный, но обязательный критерий: stitched MDD `-41.5%` хуже лимита `-25%`.
3. Улучшение нельзя объяснить простым снижением exposure: при одинаковой средней exposure dynamic strategy дала `+366.7%` против `+84.4%`.
4. Multi-horizon consensus полезен в ранних folds, после чего causal selection переключился на чистый `24h`; volatility targeting остаётся общим рабочим компонентом.
5. Alpha-ветка получает research promotion, но текущий размер позиции не deployable. Следующий слой — заранее определённый causal portfolio risk budget / drawdown scaling без изменения direction signal.

**Артефакты:**
- `exp048_momentum_vol_target.py`
- `tests/test_exp048_momentum_vol_target.py`
- `analysis_exp048_momentum_vol_target/summary.json`
- `analysis_exp048_momentum_vol_target/fold_results.csv`
- `analysis_exp048_momentum_vol_target/selection_results.csv`

---

## EXP-049: HISTORICAL_CORE time-series momentum + ML allocator

**Дата preregistration:** 2026-08-10  
**Статус:** завершён; ML allocator `NO-GO`, equal-risk и per-symbol diagnostics информативны.

**Гипотеза:** causal 24h time-series momentum из EXP-048 переносится на другие ликвидные фьючерсы. Общая cross-sectional модель может ранжировать уже сформированные momentum-сигналы лучше простого equal-risk распределения.

**Universe:** фиксированный `HISTORICAL_CORE`: BTC, ETH, SOL, XRP, BNB, DOGE, LINK, ADA. Результаты отдельных символов являются диагностикой и не используются для post-hoc исключения монет.

**Execution и costs:**
- 15m Binance USDT-M execution через `v2_backtest`;
- hourly OHLC строится только из завершённых 15m свечей и маркируется правой границей;
- решения ежедневно в `00:00 UTC`, исполнение на первом 15m open строго позже решения;
- fee `4 bps/side`, slippage `2 bps/side`, фактический funding;
- обязательная ликвидация в конце каждого fold;
- portfolio config: annual vol target `50%`, max symbol `30%`, gross `100%`, abs net `100%`, существующий causal drawdown scaling/terminal kill;
- individual-symbol diagnostic: тот же vol target `50%`, max symbol/gross/net `100%`.

**Signal:**
- для каждого символа независимо: `direction = sign(close[t] / close[t-24h] - 1)`;
- inverse-vol sizing по causal rolling `168h` annualized volatility;
- equal-risk baseline использует все доступные символы и нормирует gross raw target к `1.0`.

**ML allocator:**
- одна общая `LightGBMRegressor` для всех symbol×timestamp observations;
- target: side-adjusted net return momentum-сделки на следующих `24h`, рассчитанный по 15m execution opens, включая round-trip fee/slippage и funding;
- features: momentum `12/24/72/168h`, volatility `24/72/168h`, volume z-score, last-known/rolling funding, cross-sectional ranks, BTC market momentum/volatility, signal direction и signed momentum;
- каждый OOS fold получает свежую модель, обученную только на прошлом;
- purge `24h`: label целиком должен завершиться до train cutoff;
- фиксированная модель/гиперпараметры, без Optuna и threshold grid;
- в каждый decision timestamp выбираются top-4 доступных сигналов по predicted net return; внутри top-4 — inverse-vol weights с gross raw target `1.0`;
- evaluated fold не участвует в обучении, выборе параметров или состава universe.

**Обязательные сравнения:**
1. каждый из 8 символов отдельно при одинаковом vol target;
2. equal-risk portfolio всех доступных символов;
3. ML top-4 allocator при том же portfolio config.

**Критерии GO для ML allocator:**
- положительный stitched OOS return после всех costs;
- минимум `4/5` положительных folds;
- return выше equal-risk baseline;
- MDD не хуже `-25%`;
- PF не ниже `1.20`;
- отсутствие terminal kill.

### Portfolio results

| Portfolio | Return | MDD | Positive folds | PF | Terminal kills |
|---|---:|---:|---:|---:|---:|
| Equal-risk all 8 | **`+38.99%`** | **`-26.65%`** | **`5/5`** | `1.103` | `0` |
| ML top-4 allocator | `+26.61%` | `-32.28%` | `3/5` | `1.054` | `2` |

**Fold returns:**
- equal-risk: `+10.11%, +5.39%, +2.05%, +13.06%, +3.80%`;
- ML top-4: `+31.69%, -9.32%, +2.18%, -3.54%, +7.57%`.

ML прошёл только критерий положительного stitched return. Он не прошёл `5/6` условий: folds, excess return, MDD, PF и terminal survival. Feature importance стабильно концентрируется вокруг BTC regime features, но importance не доказывает направление зависимости; фактическое ранжирование ухудшило диверсификацию и повысило turnover/costs.

### Individual-symbol diagnostic

| Symbol | Return | MDD | Positive folds | PF |
|---|---:|---:|---:|---:|
| DOGEUSDT | **`+322.60%`** | `-27.42%` | **`5/5`** | **`1.394`** |
| XRPUSDT | **`+157.90%`** | `-29.33%` | **`5/5`** | **`1.378`** |
| ADAUSDT | `+82.62%` | `-44.33%` | `3/5` | `1.164` |
| BNBUSDT | `+50.89%` | `-36.87%` | `4/5` | `1.171` |
| SOLUSDT | `+30.16%` | `-37.88%` | `3/5` | `1.075` |
| ETHUSDT | `+10.15%` | `-41.41%` | `3/5` | `1.032` |
| BTCUSDT | `-26.21%` | `-45.74%` | `2/5` | `0.849` |
| LINKUSDT | `-34.88%` | `-50.16%` | `1/5` | `0.849` |

**Вывод:**
1. Общая модель не умеет переносимо выбирать лучшие текущие пары; простой equal-risk portfolio лучше по return, MDD, стабильности folds и survival.
2. Time-series momentum не универсален по universe: edge сильно концентрируется в DOGE/XRP, тогда как BTC/LINK отрицательны.
3. DOGE-результат EXP-048 воспроизвёлся в более строгом futures accounting: `+322.6%`, PF `1.394`, `5/5` folds после fee, slippage и funding.
4. XRP дал независимый сильный diagnostic (`+157.9%`, PF `1.378`, `5/5`), но DOGE/XRP нельзя задним числом объявлять историческим портфелем: выбор сделан после просмотра OOS.
5. Допустимый следующий шаг — заморозить DOGE/XRP как forward-only sleeve с даты после EXP-049 либо исследовать заранее заданный non-ML breadth/risk overlay без выбора символов по этим OOS.

**Артефакты:**
- `crypto_portfolio_v1/exp049_multiasset_ml_allocator.py`
- `crypto_portfolio_v1/tests/test_exp049_multiasset_ml_allocator.py`
- `crypto_portfolio_v1/reports/exp049_multiasset_ml_allocator/decision_summary.json`
- `crypto_portfolio_v1/reports/exp049_multiasset_ml_allocator/individual_symbol_comparison.csv`
- `crypto_portfolio_v1/reports/exp049_multiasset_ml_allocator/feature_importance_by_fold.csv`

---

## EXP-050: Frozen top-50 volatility × momentum diagnostic

**Дата preregistration:** 2026-08-10  
**Статус:** завершён; гипотеза не подтверждена.

**Гипотеза:** преимущество DOGE/XRP связано с тем, что 24h momentum лучше монетизируется на более волатильных инструментах. Высокая raw volatility сама по себе не должна давать преимущество размером позиции, поскольку все individual arms приводятся к одинаковому `50%` ex-ante vol target.

**Universe и окно:**
- неизменяемые `37` eligible symbols из frozen CMC top-50 manifest;
- фиксированное diagnostic window `2025-08-10 → 2026-08-09`;
- поздние листинги не исключаются после просмотра результата; manifest уже требует полное локальное покрытие этого окна;
- это development diagnostic, не новый pristine holdout.

**Signal/execution:**
- тот же causal 24h time-series momentum, daily `00:00 UTC` decisions;
- completed 15m → right-labeled hourly bars;
- first strictly-later 15m execution open;
- fee `4 bps/side`, slippage `2 bps/side`, funding, end liquidation;
- causal rolling `168h` volatility;
- individual arms: одинаковый annual vol target `50%`, max symbol/gross/net `100%`.

**Проверки:**
1. все 37 symbols отдельно, без удаления проигравших;
2. Spearman correlation между median causal volatility и individual net return;
3. causal daily high-vol portfolio: top quartile по известной в `t` volatility;
4. causal low-vol portfolio: bottom quartile;
5. all-symbol equal-risk baseline;
6. portfolio arms используют одинаковый config и inverse-vol sizing.

**Критерии поддержки гипотезы:**
- high-vol portfolio return выше all-symbol и low-vol;
- high-vol portfolio не хуже по MDD, чем low-vol, более чем на `5 п.п.`;
- Spearman volatility/return не ниже `+0.25`;
- median individual return верхнего unconditional volatility quintile выше нижнего.

### Результаты

| Portfolio | Return | MDD | PF |
|---|---:|---:|---:|
| High-vol causal quartile | `-23.37%` | `-25.07%` | `0.554` |
| All 37 equal-risk | `-23.72%` | `-25.04%` | `0.469` |
| Middle-vol | `-24.27%` | `-26.09%` | `0.345` |
| Low-vol causal quartile | **`-20.44%`** | **`-24.67%`** | **`0.774`** |

- Spearman correlation `median volatility ↔ net return`: только `+0.075`, существенно ниже preregistered `+0.25`.
- Median return верхнего volatility quintile: `-21.31%`.
- Median return нижнего volatility quintile: `-20.06%`.
- Верхний quintile (`PENGU, ENA, HYPE, BONK, TAO, ONDO, NEAR, PEPE`) не содержит прибыльной пары.
- Лучшие individual результаты на этом окне: XRP `+18.63%`, BNB `+16.40%`, UNI `+4.33%`; DOGE за последний год `-11.97%`.

**Вывод:**
1. Пройдены только `2/5` проверок: high-vol незначительно лучше all-symbol и его MDD находится в пределах `5 п.п.` от low-vol.
2. High-vol проиграл low-vol по return, correlation почти нулевая, верхний volatility quintile хуже нижнего.
3. Raw volatility не объясняет исторический успех DOGE/XRP. После одинакового risk normalization важнее качество/устойчивость самого trend edge и режим рынка.
4. EXP-049 показывал DOGE/XRP сильными на stitched `2024H2 → 2026H2`, но EXP-050 показывает существенную regime dependence: на последнем фиксированном году DOGE уже отрицателен, XRP остаётся положительным.
5. Расширять universe по критерию «самые волатильные» не следует.

**Артефакты:**
- `crypto_portfolio_v1/exp050_top50_volatility_momentum.py`
- `crypto_portfolio_v1/tests/test_exp050_top50_volatility_momentum.py`
- `crypto_portfolio_v1/reports/exp050_top50_volatility_momentum/summary.json`
- `crypto_portfolio_v1/reports/exp050_top50_volatility_momentum/individual_symbol_results.csv`

---

## EXP-052: BTC-neutral residual-shock reversal

**Дата preregistration:** 2026-08-11  
**Статус:** завершён, **NO-GO; primary stage gate провален**.

**Гипотеза:** после редкого idiosyncratic shock относительно BTC часть движения вызвана краткосрочной forced-flow/overreaction и откатывается в следующие `12h`. Это отличается от EXP-049–051: стратегия не держит постоянный momentum и торгует только события.

**Primary universe/evaluation:** фиксированный `HISTORICAL_CORE` без выбора победителей (BTC, ETH, SOL, XRP, BNB, DOGE, LINK, ADA), существующие пять OOS folds и тот же causal 15m futures accounting.

**Frozen signal:**
- hourly log-return каждого alt и BTC;
- rolling beta к BTC: `168h`, minimum `168h`, оценка сдвинута на один час;
- residual return: `alt_return - beta[t-1] * BTC_return`;
- shock score: сумма residual за `12h`, делённая на prior rolling `168h` residual std × `sqrt(12)`;
- volume confirmation: current log-volume выше prior `168h` mean на `1.0` prior std;
- решения каждые `4h`, исполнение на первом 15m open строго позже решения;
- событие: `abs(shock) >= 2.5`; одновременно берутся максимум четыре strongest shocks;
- фиксированный hold `12h`, без повторного входа до закрытия;
- primary side: `-sign(shock)` (reversal);
- falsification arm: `+sign(shock)` (continuation) на тех же timestamps/symbols;
- inverse-vol alt sizing, после чего BTC notional hedge нейтрализует estimated aggregate beta; общий raw gross нормируется к `1.0`;
- annual portfolio vol target `50%`, max symbol `30%`, gross `100%`, fee `4 bps/side`, slippage `2 bps/side`, actual funding, fold-end liquidation.

**Primary GO criteria:**
- reversal stitched return после costs положительный;
- минимум `3/5` положительных folds;
- PF не ниже `1.20`;
- MDD не хуже `-25%` и ни одного terminal kill;
- reversal return минимум на `10 п.п.` выше continuation при идентичных events.

**Stage gate:** frozen top-50 не запускается, если primary HISTORICAL_CORE не проходит все критерии. Пороги после primary run не изменяются.

**Primary results (`2024H2 → 2026H2_YTD`, 5 folds):**
- residual reversal: return `-24.75%`, MDD `-25.87%`, PF `0.773`, positive folds `1/5`, terminal kills `0`, turnover `197.86x`;
- reversal fold returns: `-17.42%, -5.34%, -0.69%, -3.66%, +0.62%`;
- residual continuation на тех же events: return `+0.84%`, MDD `-19.42%`, PF `1.006`, positive folds `1/5`, turnover `268.05x`;
- continuation fold returns: `+14.41%, -2.17%, -5.21%, -2.91%, -2.09%`;
- найдено `737` events; active decisions `16.6%`, среднее число active alts `0.23`, максимум `4`.

**Решение:** reversal провалил `5/6` checks: return, folds, PF, MDD и превосходство над continuation; прошёл только отсутствие terminal kill. Знак forced-flow reversal не подтверждён. Continuation формально положителен, но PF около `1.0`, только `1/5` положительных folds и весь результат создан `2024H2`; это не edge и не основание переворачивать гипотезу post-hoc. Согласно stage gate top-50 run не выполнялся, пороги не подбирались.

**Артефакты:**
- `crypto_portfolio_v1/exp052_residual_shock_reversal.py`
- `crypto_portfolio_v1/tests/test_exp052_residual_shock_reversal.py`
- `crypto_portfolio_v1/reports/exp052_residual_shock_reversal/summary.json`
- `crypto_portfolio_v1/reports/exp052_residual_shock_reversal/event_positions.csv`

---

## EXP-051: Frozen top-50 causal trend-quality diagnostic

**Дата preregistration:** 2026-08-10  
**Статус:** завершён, **NO-GO — trend-quality hypothesis не подтверждена**.

**Гипотеза:** 24h momentum работает лучше не на самых волатильных парах, а когда предшествующее движение было направленным. Causal Kaufman-style efficiency должен различать trend и noise лучше raw volatility.

**Universe/window/execution:** те же frozen 37 symbols и diagnostic window `2025-08-10 → 2026-08-09`, что в EXP-050; daily decisions, completed 15m bars, strictly-later 15m execution, fee/slippage/funding, одинаковый portfolio risk config.

**Trend-quality score:**
- для `24h`, `72h`, `168h`: `abs(log_close[t] - log_close[t-h]) / sum(abs(hourly_log_return), h)`;
- каждый компонент causal и ограничен `[0, 1]`;
- итоговый score — равное среднее трёх efficiency horizons;
- quality влияет только на выбор symbols; direction остаётся frozen `24h momentum`;
- sizing внутри выбранной группы остаётся inverse-vol.

**Проверки:**
1. high-quality portfolio: top causal quartile каждый день;
2. low-quality portfolio: bottom causal quartile;
3. all-symbol equal-risk baseline;
4. exact next-24h net labels для проверки cross-sectional rank IC после costs;
5. correlation между median symbol quality и уже сохранённым individual return EXP-050.

**Критерии поддержки гипотезы:**
- high-quality portfolio return выше all-symbol и low-quality;
- high-quality PF выше `1.0`;
- high-quality MDD не хуже low-quality более чем на `5 п.п.`;
- mean daily cross-sectional Spearman quality/net-return выше `+0.05`;
- median symbol-quality / individual-return Spearman выше `+0.20`.

**Результаты frozen diagnostic (`2025-08-10 → 2026-08-09`):**

| Arm | Return | MDD | PF | Turnover |
|---|---:|---:|---:|---:|
| all equal-risk | `-23.72%` | `-25.04%` | `0.469` | `54.22x` |
| high trend-quality | `-24.00%` | `-25.15%` | `0.456` | `51.44x` |
| low trend-quality | `-21.77%` | `-25.04%` | `0.854` | `208.33x` |

- high-quality не победил ни all-symbol, ни low-quality;
- high-quality достиг terminal drawdown kill в октябре 2025; поэтому нулевые месяцы после октября не являются стабильностью;
- exact net-label daily rank IC: mean `+0.011`, median `+0.003`, positive share `50.4%` — практически случайный ranking;
- Spearman между median symbol trend-quality и individual return: `+0.109`, ниже preregistered `+0.20`;
- выполнен только MDD-tolerance check; остальные preregistered checks провалены.

**Вывод:** Kaufman-style направленность предыдущего движения не даёт полезного causal ranking для следующего 24h momentum return. Вместе EXP-050/051 показывают, что успех отдельных DOGE/XRP периодов нельзя объяснить простой постоянной характеристикой пары — ни raw volatility, ни средней trend efficiency. Дальнейший подбор quality thresholds на этом же frozen окне запрещён как post-hoc overfit.

**Артефакты:**
- `crypto_portfolio_v1/exp051_top50_trend_quality.py`
- `crypto_portfolio_v1/tests/test_exp051_top50_trend_quality.py`
- `crypto_portfolio_v1/reports/exp051_top50_trend_quality/summary.json`
- `crypto_portfolio_v1/reports/exp051_top50_trend_quality/daily_rank_ic.csv`
- `crypto_portfolio_v1/reports/exp051_top50_trend_quality/symbol_quality_vs_return.csv`

---

## ПРОТОКОЛ ОЦЕНКИ (актуальный, 2026-08-09)

| Кусок | Значение |
|---|---|
| Train roll | ≤ 2024-12-31 (или ≤ 2025-06-30 для более старых EXP) |
| Internal select | 2025-01-01 … 2025-06-30 (внутри train, без подглядывания в final) |
| Final test | **2025-07-01 → latest** (H2’25 + 2026), части можно репортить отдельно |
| Objective | max net return после реальных fees |
| Actions (engine v2) | HOLD / FLAT / LONG / SHORT + masks |
| Нельзя | FLIP-бот сигналы в PPO; broken hysteresis; Optuna до стабильного engine/политики churn |

---

## СВОДКА ВЫВОДОВ (PPO / self-trade, авг 2026)

1. **Direction ML / ZigZag / rule microtrend** — edge на честном OOS нет или статистически ноль.
2. **Winrate ≠ edge** — base models по WR бесполезны для return-stack.
3. **PPO engine после фикса корректен** (masks, reward=log equity, toy market).
4. **15m timescale** — commission noise доминирует (fee-drag 84–95%).
5. **1h timescale + 24h masked hold** — turnover и fees заметно ниже; median final улучшился до −40%, но 1/5 profitable.
6. **Churn был важной, но не единственной проблемой** — после его снижения остаются отсутствие устойчивого signal edge, seed variance и select→final overfit.
7. **2026 alone** — плохой sole test; использовать H2’25+2026.
8. EXP-045 закрыл это условие: при low-turnover и honest walk-forward текущий PPO edge не подтверждён.

---

## ОТКРЫТЫЕ ВОПРОСЫ / СЛЕДУЮЩИЕ ШАГИ

- [x] Grid поиск трешхолдов и температуры (EXP-010)
- [x] Динамическая температура — концепция и тест (EXP-012)
- [x] Честный walk-forward тест (EXP-013)
- [x] Тепловая карта (EXP-014)
- [x] Валидация PPO engine v2 (masks / accounting / toy)
- [x] DOGE 15m smoke на fixed engine (EXP-042) — fail fees
- [x] Переход на DOGE 1h + smoke (EXP-043) — лучше, но final минус
- [x] PPO 1h: 4h decisions + masked 12/24h hold (EXP-044) — churn↓, edge неустойчив
- [x] Полноценный walk-forward low-turnover PPO vs momentum (EXP-045) — PPO fail, momentum сильнее
- [x] Causal momentum + regime/risk layer для снижения MDD (EXP-046)
- [x] Fold-local ML meta-label поверх momentum 24h (EXP-047) — fail против risk-matched baseline
- [x] Multi-horizon momentum + causal volatility target (EXP-048) — alpha PASS, MDD fail
- [x] HISTORICAL_CORE momentum + ML allocator (EXP-049) — ML fail; DOGE/XRP diagnostic strong
- [x] Frozen top-50 volatility × momentum (EXP-050) — raw volatility hypothesis rejected
- [ ] Seed-consensus/ensemble PPO — отложить: базовый walk-forward edge не доказан
- [ ] Длиннее train + multi-seed stability до Optuna
- [ ] Реализовать динамическую температуру в боте (`bot_binance_flip`)
- [ ] Накопить реальные данные торговли на сервере
- [ ] Разобраться с системной проблемой 2026-01-29 (LONG перед Trump dump)
- [ ] Рассмотреть "Trump/Musk index" — внешний сигнал для фильтрации политических событий

---

## ЗАМЕЧАНИЯ

- **Январь–Февраль 2025** — бычий рынок, любая модель выглядит хорошо
- **Январь–Февраль 2026** — Trump/геополитика, непредсказуемые движения, главный риск
- **2026-01-29** — системная проблема: все модели ошибаются одинаково → внешний шок
- Зануление фичей ≠ обучение без них: модель продолжает их "ожидать", веса не пересчитаны
- **2024 vs 2025** — несравнимые режимы: 2024 = трамп-ралли, 2025 = смешанный рынок
- **Динамическая T** — 100% комбинаций бьют PROD → механизм логически обоснован
- **Walk-forward** всегда хуже PROD → рынок не авторегрессивен на уровне параметров
- **MDD -27%** достижимо с динамической T (vs -37–39% у фиксированной)
- **Shock cooldown filter** — слишком тупой: слегка помогает только на `FIXED`, но не чинит `DYN PROD`
- **Shock threshold boost** — лучше cooldown: небольшой anti-shock слой без паузы может быть полезен
- **Асимметрия в shock-режиме** указывает на проблему LONG-входов, но по total-metrics пока не лучше симметричного boost
- **PPO on 15m** умирает от turnover/fees даже при корректном engine
- **PPO on 1h** — первый режим, где select стабильно бывает в плюсе; final всё ещё ломает fee + overfit
- **EXP-044:** 24h min-hold снижает median fee drag до 18%, но median final −40%; значит fees не были единственной причиной
- **Простой momentum 12h** (+47.5%, MDD −54.6%) обошёл median PPO; обязательный baseline для следующего walk-forward
- **EXP-045:** honest WF PPO прибыльный только в 6/15 fold×seed; stitched all-folds все seeds −57…−77%
- **Momentum 24h** прибыльный 4/5 folds и stitched +150%, но MDD −57%; signal интереснее PPO, execution/risk ещё не годится
- **Live FLIP** (редкие холды) = референс формы сделки, не supervised target и не PPO feature
- **Select на 2025H1 ≠ final H2’25+2026** — большой gap; один «красивый» checkpoint не доказательство
- **Copy-trading «лотерея»:** разовый срез лидерборда + бэктест по выжившим = survivorship; early-hot хуже cold для 5×; нужен point-in-time архив топа (см. EXP-082)
- **Не цену, а вероятности событий:** точка `close[t+h]` почти непредсказуема; считаемые частоты `P(событие | контекст)` и EV после издержек — рабочий объект (см. NOTE-083); мерилка `prob_meter` (NOTE-084/085): BTC 5m+1h depth/flow после 8 bps — карман не найден, микро-направление закрыто → дальше regime/maker

---

### EXP-053 addendum — extended Optuna seed check

После обнаружения, что победивший в первом Optuna stage seed `29` не
вошёл в финальный диапазон `0…19`, seeds `29/47/71/89` дополнительно
проверены на top-3 архитектурах.

- новый победитель `2025H2`: trial `42`, seed `47`;
- selection: `+38.65%`, MDD `-24.02%`, PF `1.230`;
- diagnostic 2026: `-27.17%`, MDD `-45.59%`, PF `0.769`;
- решение: **NO-GO**.

Seed `29` не перенёс победу с `2025H1` на `2025H2`: результаты по трём
архитектурам `-6.83%`, `-12.92%`, `-23.97%`. Выбор отдельного лучшего
seed остаётся нестабильным между последовательными режимами.

---

### EXP-053 addendum 2 — fixed architecture, 50 seeds

Frozen Optuna trial `51` обучен с seeds `0…49` только на данных до конца
2024. Seed и единый correction grid выбирались на полном 2025 при
обязательной положительной доходности отдельно в H1 и H2.

- eligible: `26/50` seeds;
- champion: seed `29`;
- 2025: `+806.95%`, MDD `-41.01%`, PF `1.575`;
- 2025H1 / H2: `+652.40%` / `+20.24%`;
- frozen 2026 diagnostic: `-54.41%`, MDD `-61.86%`, PF `0.619`;
- решение: **NO-GO**.

Даже фильтр по двум различным режимам 2025 не обеспечил переносимость
лучшего historical seed на 2026.

---

### EXP-053 addendum 3 — rolling monthly adaptation

Проверена причинная схема `36 месяцев train → 1 месяц validation → 1 месяц
test` со сдвигом на месяц. Test охватывает `2025-01 → 2026-08`; frozen
архитектура trial `51`, 10 seeds на каждом из 20 folds, 200 обучений.

- stitched return: `-68.47%`;
- MDD: `-74.89%`;
- PF: `0.827`;
- profitable months: `7/20`;
- 2025 / 2026 YTD: `-62.20%` / `-16.61%`;
- validation-to-next-test return correlation: `+0.053`;
- решение: **NO-GO**.

Ежемесячная адаптация не решила regime shift: лучший seed/grid предыдущего
месяца почти не содержит информации о следующем. Увеличивать pilot с 10 до
50 seeds нецелесообразно, поскольку это усилит selection overfit.

---

### EXP-054 — simple UP/DOWN/FLAT 6h edge pilot

До любой Optuna проверена простая постановка: три независимые binary-модели
UP/DOWN/FLAT, horizon `6h`, sequence `60`, seed `0`, FLAT band `±0.20%`.
Train до конца 2024, validation 2025, frozen diagnostic 2026; объединение
простым argmax без threshold/temperature grid.

- mean directional AUC 2025 / 2026: `0.514 / 0.514`;
- FLAT AUC 2025 / 2026: `0.552 / 0.507`;
- 2025: net `-45.74%`, gross `-35.50%`, PF `0.784`;
- 2026: net `+19.55%`, gross `+34.68%`, PF `1.181`;
- fixed SHORT: `+61.60%` в 2025 и `+20.05%` в 2026;
- решение: **NO EDGE**.

Положительный 2026 replay не подтверждает ML edge: directional AUC около
случайного, а fixed SHORT дал немного больше. Optuna для этой постановки
запускать нельзя — она будет подбирать шум.

#### EXP-054 follow-up — FLAT band ±2%

Без перебора промежуточных значений FLAT band расширен до `±2%`.

- directional AUC 2025 / 2026: `0.537 / 0.522`;
- trades 2025 / 2026: `96 / 32` против `154 / 107` при `±0.2%`;
- 2025: net `-9.96%`, gross `+0.24%`, PF `0.915`;
- 2026: net `-1.46%`, gross `+2.13%`, PF `0.966`;
- решение: **NO EDGE**.

Широкий FLAT band заметно снизил churn и убыток, но gross edge остался около
нуля и не покрывает costs. Высокая combined accuracy 2026 (`74.9%`) в основном
объясняется FLAT share `83%`.

#### EXP-054 follow-up — core-30 features

При неизменных horizon `6h`, sequence `60`, seed `0` и FLAT `±2%` feature
set сокращён со 129 до 30 заранее зафиксированных market-признаков.

- directional AUC 2025 / 2026: `0.563 / 0.621`;
- 2025: net `+78.11%`, gross `+111.91%`, MDD `-39.20%`, PF `1.365`;
- 2026: net `+0.93%`, gross `+4.95%`, MDD `-35.46%`, PF `1.020`;
- fixed SHORT 2025 / 2026: `+61.60% / +20.05%`;
- статус: **PREDICTIVE EDGE, EXECUTION NO-GO**.

Сокращение признаков дало первый явно переносимый ranking signal, но простой
argmax трёх независимо class-weighted моделей плохо конвертирует его в PnL.
Full Optuna по frozen критериям пока запрещена; сначала требуется исправить
только calibration/fusion, не меняя feature set и model weights.

#### EXP-054 follow-up — frozen-score multinomial fusion

Три frozen core-30 score откалиброваны unweighted multinomial logistic
regression на 2025 и без изменений применены к 2026.

- class accuracy 2025: `56.1% → 71.5%`;
- class accuracy 2026: `76.1% → 83.0%`;
- calibrated trades 2025 / 2026: `1 / 0`;
- calibrated return 2025 / 2026: `+26.55% / 0.00%`;
- решение: **FUSION FAILED**.

Калибровка закономерно выбрала доминирующий FLAT почти всегда. Рост accuracy
не конвертируется в торговлю. Для использования ranking edge требуется
иерархическая схема MOVE gate + conditional UP/DOWN direction, а не
трёхклассовый posterior argmax.

#### EXP-054 follow-up — ignore FLAT

На frozen core-30 scores FLAT полностью исключён: каждый час выбирался только
максимум UP/DOWN.

- 2025: net `+30.82%`, gross `+65.37%`, PF `1.108`, trades `210`;
- 2026: net `-7.76%`, gross `+7.98%`, PF `0.952`, trades `141`;
- 2026 cost drag: `15.74 п.п.`;
- решение: **FAILED**.

Без FLAT gross signal немного усилился, но число FLIP выросло в четыре раза
и costs уничтожили edge. FLAT нужен как gate; прямой always-in-market
UP/DOWN controller непригоден.

#### EXP-054 follow-up — hierarchical MOVE + direction

Обучены две unweighted core-30 модели: MOVE/FLAT на всех samples и UP/DOWN
только на MOVE samples. Controller использовал threshold `0.5` и подтверждение
FLAT/FLIP двумя часами.

- MOVE AUC 2025 / 2026: `0.599 / 0.645`;
- conditional direction AUC: `0.546 / 0.492`;
- 2025: `+55.35%`, MDD `-23.16%`, PF `2.456`, 21 сделка;
- 2026: `-1.36%`, MDD `-5.31%`, PF `0.385`, 2 сделки;
- решение: **NO-GO**.

MOVE gate дал сильный переносимый ranking signal, но conditional direction
на малой MOVE-выборке не перенёсся. Gate перспективен как фильтр поверх
исходных core-30 UP/DOWN scores; отдельная conditional direction model — нет.

---

### EXP-055 — six-step future-close trajectory

Одна core-30 Transformer предсказывала cumulative returns для следующих
`1…6h`. Max epochs `300`, patience `25`, ReduceLROnPlateau; seed `0`.

- обучение остановлено на epoch `28`, лучший checkpoint — epoch `3`;
- mean Spearman 2025 / 2026: `0.026 / 0.008`;
- mean sign accuracy: `50.96% / 49.96%`;
- MAE хуже zero-return baseline на 5/6 horizons в 2025 и 6/6 в 2026;
- 2025: `-22.87%`, MDD `-41.41%`, PF `0.851`;
- 2026: `+14.47%`, MDD `-27.67%`, PF `1.310`;
- решение: **NO-GO**.

Прибыльный 2026 replay не подтверждён regression diagnostics и не перенёсся
с 2025. Большее число эпох не помогло: переобучение началось сразу после
третьей эпохи.

---

### EXP-056 — causal Fibonacci reversal event study

На DOGEUSDC проверены реакции после causal-confirmed swing impulses без
нейросети. Pivot подтверждался через четыре правых часа; Fibonacci
`38.2/50/61.8%` сравнивались с линейными `33.3/50/66.7%` и тремя
детерминированными random retracements на тех же impulses. Вход — следующий
open, результаты net of fees, slippage и funding.

- primary 12h, 2025 Fibonacci: mean `-0.149%`, PF `0.884`, `1,109` events;
- 2025 linear / random: `-0.120% / -0.100%`;
- primary 12h, 2026 Fibonacci: mean `-0.067%`, PF `0.910`, `745` events;
- 2026 linear / random: `-0.093% / -0.099%`;
- единственный positive cell: Fibonacci 24h в 2026 `+0.048%`, PF `1.048`,
  но bootstrap CI `[-0.154%; +0.223%]` включает zero, а 2025 дал `-0.228%`;
- решение: **NO-GO**.

Fibonacci ratios не дали стабильного преимущества перед обычными и random
retracement depths. Возможный regime-specific pullback effect не подтверждает
ценность точных Fibonacci levels как самостоятельного reversal signal.

---

### EXP-057 — DOGEUSDC whale-tape pilot

Из 55.1 млн Binance Futures `aggTrades` causal-агрегацией построено 875,520
минут DOGEUSDC. Whale threshold рассчитывался только по предшествующим 30
завершённым дням; shock threshold — по предшествующим семи дням. Проверены
следование за крупным агрессивным flow и простой absorption-разворот.

- найдено `1,709` shocks, классифицировано `1,521`;
- continuation 5m, 2025 / 2026: `-0.145% / -0.140%`, PF `0.363 / 0.321`;
- continuation 60m: `-0.124% / -0.132%`, PF `0.741 / 0.649`;
- absorption 5m: `-0.235% / -0.064%`, PF `0.257 / 0.556`;
- absorption 60m: `-0.301% / -0.066%`, PF `0.504 / 0.787`;
- ни один 2026 bootstrap CI не подтвердил positive mean;
- решение: **NO-GO as standalone signal**.

Публичная лента крупных сделок несёт информацию о фактическом агрессивном
потоке, но вход после видимого пятиминутного impulse запаздывает, а absorption
без контекста уровня недостаточен. Whale tape имеет смысл проверять как
условный признак внутри заранее существующей crowd-attention zone.

---

### HYP-058 — hindsight labels, causal precursors

**Гипотеза:** преимущество legacy `bot_binance_flip` могло возникнуть не только
из прямой утечки будущего во входы. Ретроспективный ZigZag/turning-point
pipeline мог создавать качественные, очищенные от шума labels с использованием
будущих свечей, а модель параллельно обучилась распознавать их предвестники по
той части признаков, которая реально доступна в момент решения.

Использование будущего для определения training target само по себе допустимо:
будущее запрещено во входном tensor при inference, но может использоваться для
последующей разметки исторического состояния. Это объясняет, почему старый
checkpoint сохраняет прибыльность при deterministic causal replay, хотя
переобученные полностью causal постановки edge не воспроизвели.

Альтернативные объяснения, которые необходимо отделить:

- прибыль создана выбором champion среди 100 seeds и подбором thresholds;
- causal replay использует feature distribution, отличную от старого live
  mutable pipeline;
- модель опирается на leaked columns, но сохраняет остаточный trend bias;
- результат держится на одном рыночном режиме или высокой compounding exposure.

**Проверка гипотезы:**

1. Заморозить legacy checkpoint, thresholds и dynamic temperature.
2. Разделить исходные признаки на строго causal и подозрительные/noncausal.
3. На одном фиксированном replay по очереди lag/zero/permutation-ablate каждый
   подозрительный признак и группы признаков.
4. Сравнить изменение probabilities, signal timestamps, net PnL, PF и MDD.
5. Если causal-only tensor сохраняет существенную часть edge, восстановить
   legacy hindsight labels и обучить новый student только на causal inputs.
6. Окончательное подтверждение — только frozen checkpoint на новом untouched
   forward period с сохранением каждого live feature snapshot.

**Фальсификация:** гипотеза отвергается, если удаление noncausal inputs разрушает
сигналы и PnL либо causal-only student не переносит результат на untouched
forward data.

---

### EXP-059 — честный grid-retest порогов flip на 2026

Production thresholds (`long=0.48 / short=0.52`) были выбраны 529-cell grid
search (`long, short ∈ [0.48, 0.70]`, шаг `0.01`) по периоду до конца февраля
2026 (`model_card.json`), т.е. этот период не был honest OOS для победителя.
Тот же grid, тот же frozen `prod_doge_1h_v1/model.pt` (seed 72), тот же
signal/execution convention пересчитан заново, но оценка вынесена на
`2026-03-01` и позже — первый день, не тронутый исходным подбором. Исполнение
DOGEUSDC — реальные 1-минутные бары (не часовые), совпадает со стандартом
`replay_binance_flip_causal_1m.py`; causal 1h/4h/1d фичи по DOGEUSDT/BTC/ETH
строятся тем же minute-aggregated count-filtered способом, что и в
`run_flip_causal_2025_2026.py`.

- production cell на untouched-окне (март 2026+): `+38.4%`, MDD `-33.1%`,
  134 сделки — выше p75 всей сетки (`+31.8%`) и выше медианы (`+15.0%`);
- ранг production среди 529 cells по untouched-return: **97/529** (top 18%);
- лучшая cell на untouched-окне: `(0.51, 0.48)` → `+63.7%`, MDD `-28.4%`;
  худшая reliable cell: `(0.65, 0.62)` → `-21.0%`;
- 400/529 (76%) cells дали positive untouched return; 129 — negative;
- 3-cell окрестность вокруг production: median `+42.4%` (стабильный локальный
  регион, не изолированный пик);
- **исправление после первого прогона:** grid изначально симулировался на
  часовых, а не минутных барах (`evaluate_cell` был подключён к hourly-кадру,
  который нужен только для калибровки temperature, а не к `execution_1m`),
  из-за чего MDD был занижен — production 2025/full MDD показывал `-31.6%`
  вместо правильных `-56.5%` (совпадает с ранее проверенным
  `analysis_flip_causal_2025_2026`). Доходности и число сделок не изменились
  (цена входа/выхода = open первой минуты часа в обоих случаях). Баг
  исправлен, grid пересчитан на настоящих 1-минутных барах;
- Spearman ρ между 2025-selection return и 2026-untouched return по всей
  сетке: `0.68` — сетка несёт реальный forward-signal, но недостаточно точный,
  чтобы конкретная top-2025 cell осталась top-tier out of sample;
- на touched Jan–Feb 2026 (уже видел исходный grid) production почти flat
  (`-0.6%`), тогда как соседние cells уже показывали двузначный плюс;
- решение: **production не debunked, но и не подтверждён как оптимум** —
  пороги остаются рабочими, ревизия не требуется, но будущая честная
  re-optimization (fit только до 2026-03-01, eval только после) может дать
  лучший результат, например в районе `(0.51, 0.48)…(0.54, 0.49)`;
- **проверка границы грида (follow-up):** best cell по untouched-окну упирался
  в нижнюю границу `short_threshold=0.48/0.49` для 22 из 23 значений
  `long_threshold` (у `long_threshold` оптимум всегда внутренний, `0.51`, ни
  разу не на границе). Досчитал 69 cells с `short ∈ {0.45, 0.46, 0.47}`
  (переиспользуя закэшированные calibrated probabilities и 1m execution —
  без передокачки/переинференса). Результат: расширение **не побило** старый
  максимум — для `long=0.51` доходность выходит на плато `63.4%` (было
  `63.74%`, те же 122 сделки), для большинства других `long` при `short<0.49`
  доходность равна или ниже, чем при `0.49`. Т.е. это было плато, а не
  монотонно растущий тренд, отрезанный границей — `(0.51, 0.48)/(0.51, 0.49)`
  остаётся лучшей найденной cell. **Важно:** это решение расширить сетку было
  принято на основе того, что мы уже видели на untouched-окне (март 2026+),
  поэтому формально это окно больше не honest OOS для будущих threshold-решений
  — нужен новый forward-период (от `2026-09-10` и позже), не тронутый ни
  исходным, ни расширенным поиском;
- **regime-check (объекция: "2025 шортовый, 2026 флет"):** цена DOGEUSDC
  падала во **всех** окнах (2025: `-62.8%`, touched Jan-Feb: `-19.9%`,
  untouched Mar+: `-11.3%`) — за весь период (янв 2025 – сен 2026) не было ни
  одного bull-режима, т.е. "робастность по окнам" = "робастность по двум
  степеням одного и того же снижения", а не по разным режимам. Но гипотеза
  "весь эдж = шорт-бета на тренд" не подтвердилась: корреляция месячного
  price_pct и strategy return_pct лучшей cell `(0.51, 0.48)` = `+0.15`
  (почти ноль, у чистой шорт-беты была бы сильно отрицательная); в UP-месяцы
  (n=9) стратегия зарабатывала в среднем `+46.4%` (win rate 89%), в
  DOWN-месяцы (n=12) — `+21.0%` (win rate 75%), т.е. лучше именно в
  растущие месяцы. Открытый вопрос на будущее: как поведёт себя стратегия
  в устойчивом bull-режиме — такого периода в истории пока не было;
- **попытка снизить просадку (best cell 0.51/0.48):** просадку создают не
  массовые мелкие whipsaw-убытки, а единичные многодневные сделки против
  тренда — все 10 худших сделок держались 57-307 часов, худшая серия
  (`-27.15%`) — это всего 2 сделки подряд. "Уверенность" входа (запас
  вероятности над порогом) убыток не предсказывает (корр. `0.10`). Проверил
  3 вида риск-оверлея на настоящих 1-минутных ценах: (1) стоп-лосс с
  reset-и-переоткрытием — не улучшает MDD ни при одной конфигурации, только
  режет доходность на 15-30%+ (лишние комиссии на каждый reset); (2)
  стоп-лосс с уходом в кэш до следующего flip — сильно хуже по обоим
  параметрам (доходность падает в разы, MDD местами растёт до `-60.9%`),
  потому что именно долгие сделки (>48ч) дают наибольший средний P&L
  (`+2.50%`/сделка) — обрезка убирает больше хороших сделок, чем плохих;
  (3) throttle размера позиции по текущей просадке equity — лучшая
  3-tier конфигурация даёт MDD `-22.19%` (return `+6296%`), но это
  статистически не отличается от простого снижения базового размера позиции
  до 85% без всякой условной логики (MDD `-22.02%`, return `+6027%`).
  **Итог:** ни один из протестированных "умных" риск-оверлеев не бьёт
  тривиальное снижение плеча/размера позиции — просадка похожа на
  структурное свойство выбранного уровня экспозиции на волатильном
  инструменте, а не на исправимый баг в механике выхода. Непроверенные
  идеи на будущее: сайзинг по реализованной волатильности (проактивно, до
  входа, а не реактивно после просадки) и диверсификация по нескольким
  малокоррелированным инструментам/сигналам;
- **4-й вариант — хедж тем же инструментом с гистерезисом** (открыть
  противоположную сделку при просадке `-5%`, снять при восстановлении до
  `-2%` или смене сигнала): экономически при equal-and-opposite хедже
  цена/фандинг обнуляются, пока хедж активен, поэтому это = "заморозить"
  P&L на момент открытия хеджа минус комиссия за хедж-ногу. Прогнал 7
  конфигураций (5/2%, 5/3%, 5/0%, 8/3%, 8/5%, 10/5%, 10/3) — **везде MDD
  стал хуже baseline** (`-37%…-46%` против `-25.56%`), доходность упала в
  4+ раза. Причина проверена напрямую: на 11 по-настоящему катастрофических
  сделках (baseline `<-8%`) механизм даёт `+56.1` п.п. суммарно — работает
  как задумано. Но по всем 342 сделкам 84 стали хуже baseline против только
  66 лучше, и на сделках, которые в baseline были прибыльными, но задели
  хедж-логику по пути, суммарная просадка `-123.2` п.п. — более чем вдвое
  перекрывает выигрыш на хвостовых рисках. Механизм при развороте цены
  внутри полосы `-5%/-2%` фактически продаёт на локальном дне и откупает
  выше — на волатильном/choppy DOGE это происходит часто и обходится
  дороже, чем защита от хвостовых потерь.   Тот же фундаментальный дефект,
  что и у "stop-loss с reset" (только смягчённый гистерезисом);
- **5-й вариант — проактивный vol-scaled sizing** (размер позиции ДО входа
  по trailing realized vol относительно causal expanding-median target):
  проверка на 10 худших исторических сделках показала, что realized vol
  перед входом во ВСЕ 10 была на уровне медианы или ниже (худшая сделка,
  `-18.33%`, вошла при vol `0.0092` против медианы `0.0093` — ничем не
  примечательно; сделка `2026-06-01` вошла при vol `0.0046`, вдвое ниже
  медианы). Т.е. катастрофы не предвещаются всплеском волатильности.
  Конфиги с cap≤1.0 почти не двигают MDD (`-25.34%` vs baseline `-25.56%`,
  в пределах шума) при потере ~40% доходности; конфиги с cap>1.0 делают
  MDD хуже (низкая вола чаще высокой, поэтому средний leverage растёт
  выше baseline).
- **итог по риск-оверлеям:** проверено 5 разных механизмов (stop-reset,
  stop-and-wait, equity-throttle, hedge-гистерезис, vol-sizing) — ни один
  не бьёт тривиальное снижение размера позиции по risk/return. Хвостовые
  сделки не предвещаются ни уверенностью сигнала, ни реализованной
  волатильностью. Два рычага остаются НЕ фальсифицированными (структурно
  отличаются от всего протестированного): (1) плоское снижение
  плеча/размера — работает, но это прямой трейдофф, не "умное" решение;
  (2) диверсификация другим инструментом/сигналом (не тем же DOGEUSDC) —
  не проверялась. Более крупный, непроверенный вариант — искать фичи,
  которые реально предсказывают хвостовые сделки (funding rate экстремумы,
  дисбаланс ликвидности, корреляционные пробои с BTC) — это отдельный
  research-проект, а не быстрый фикс.

Артефакты: `causal_flip/FLIP_THRESHOLD_GRID_2026_PROTOCOL.md`,
`causal_flip/flip_threshold_grid_2026.py`,
`causal_flip/reports/flip_threshold_grid_2026/{REPORT.md,summary.json,grid.csv,grid_extended_short_down.csv}`.

### EXP-060 — Cross-asset transfer: тот же frozen-модель на HYPE и SOL

Проверка: работает ли `prod_doge_1h_v1` (те же веса, те же production
пороги `(0.49, 0.55)` из чекпоинта, без переобучения) на других волатильных
парах — подставили HYPEUSDT/SOLUSDT в DOGE-слот 129-фичевой схемы (BTC/ETH
остаются настоящими BTC/ETH, macro/sentiment фичи универсальны), прогнали
через тот же frozen inference + честный 1-минутный `simulate_account`.

- **По full-period цифрам выглядит неоднозначно**: SOL `+397.4%` (MDD
  `-55.2%`, 487 сделок, win rate 57.3%), HYPE `-17.5%` (MDD `-67.0%`, 350
  сделок, win rate 59.1%);
- **но разбивка по тем же честным окнам показывает, что SOL-результат — это
  артефакт fit-периода**: SOL 2025 (тот самый период, на котором тюнилась
  DOGE-модель) `+508.0%`, а вот 2026 touched `-5.0%` и 2026 **untouched
  `-13.9%`** — обе честные OOS-цифры отрицательные. HYPE аналогично: 2025
  `+102.8%`, touched `-14.6%`, untouched `-52.4%` — тоже отрицательно, и
  сильнее;
- **итог: перенос НЕ работает zero-shot ни на одну из двух пар** на честной
  OOS-проверке, хотя и не выглядит как чистый шум (модель дала связную
  крупную прибыль именно в тот период, на котором она обучалась для DOGE,
  что вряд ли было бы совпадением при бессмысленных весах) — то есть модель
  не 100% DOGE-специфичный оверфит, но и не показывает переносимый edge без
  собственной калибровки/обучения на новом активе;
- SOL остаётся наиболее теоретически интересным кандидатом для диверсификации
  по чистой корреляции (residual corr `0.484`, самый низкий из проверенных в
  EXP-059 pair screen), но это исследование показывает, что "просто взять
  готовую модель и торговать SOL" не работает — если делать SOL стратегию,
  нужно отдельное обучение/калибровка с тем же честным walk-forward
  протоколом, а не zero-shot перенос.

Артефакты: `cross_asset_transfer.py`,
`causal_flip/reports/cross_asset_transfer/{REPORT.md,summary.csv,HYPEUSDT_trades.csv,SOLUSDT_trades.csv}`.

**Follow-up (та же серия, тот же день): это провал модели или провал DOGE-порогов?**
Резонное замечание: модель обучалась на таргете generic-фичами, а FLIP-стратегия
(пороги `0.49/0.55` + dynamic temperature) — отдельный слой, калиброванный
конкретно под распределение вероятностей DOGE. Проверили раздельно, используя
уже сохранённые `{HYPEUSDT,SOLUSDT}_signals.csv` (без повторного inference/download):

1. **IC-проверка (независимо от порогов)**: Spearman-корреляция
   `prob_up - prob_down` с forward return. У SOL edge сильный на fit-периоде
   (+0.22 на 72ч) и разваливается OOS (touched `-0.15`, untouched `+0.05`,
   слабо). У HYPE слабее, но остаётся **положительным и значимым во всех
   окнах включая untouched** (`+0.045`, p=0.0026) — то есть сырой сигнал
   модели не чистый шум даже OOS на HYPE.
2. **Персональный grid порогов на каждый актив** (21×21=441 ячеек, 0.30-0.70
   шаг 0.02, честный протокол — выбор на fit-периоде, оценка на untouched,
   плюс oracle-потолок для сравнения):
   - SOL: честный выбор `-42.2%` на untouched (хуже, чем просто взять
     DOGE-пороги: `-13.9%`); oracle-максимум (с читерским подглядыванием в
     untouched) всего `+8.8%`, и только 9/441 ячеек положительны;
   - HYPE: честный выбор `-53.2%` (≈ то же самое, что DOGE-пороги `-52.4%`);
     **oracle-максимум тоже отрицательный: `-31.2%`, 0/441 ячеек положительны**;
     ранговая корреляция fit↔untouched по гриду **отрицательная** (`-0.58`) —
     то, что хорошо на fit-периоде, в среднем хуже на untouched (признак
     нестационарности режима, а не просто смещённой калибровки).
3. **Вывод: гипотеза "дело только в порогах" не подтвердилась.** Она была
   разумной и стоила проверки — распределение вероятностей действительно
   сдвинуто (mean `prob_down≈0.55-0.60` на SOL/HYPE против условно
   центрированного на DOGE), и на HYPE сырой сигнал реально не является
   чистым шумом OOS. Но даже с неограниченным hindsight-подбором порогов
   под конкретный актив рабочей, переживающей издержки стратегии не находится
   ни на одном из двух активов на честном untouched-окне. Слабый edge (там,
   где он есть — на HYPE) недостаточно силён относительно транзакционных
   издержек/whipsaw, чтобы его можно было извлечь любым статическим
   пороговым правилом.

Артефакты доп.: `causal_flip/reports/cross_asset_transfer/{SOLUSDT,HYPEUSDT}_threshold_grid.csv`.

### EXP-061 — Нативное обучение с нуля на SOL и XRP (H6-таргет + отдельный FLIP-слой)

Уточнение пользователя: DOGE-модель изначально обучалась на **H6
fixed-horizon** таргете (предсказать направление через 6 баров, держать
ровно 6 баров, без флипа) — см. `backtest_h6_fixed` в
`prod_doge_1h_v1/model_card.json`. **FLIP** (держать до смены сигнала +
dynamic temperature + свой grid порогов) — это ОТДЕЛЬНЫЙ слой, найденный
и откалиброванный уже ПОСЛЕ обучения модели, поверх её сырых вероятностей.
EXP-060 тестировал zero-shot перенос и весов, и DOGE-порогов сразу — то
есть смешивал два разных вопроса. Здесь исправлено: обучили модель **с
нуля** на SOL и XRP (собственные веса, та же фиксированная архитектура
DOGE — без per-asset Optuna-поиска, ради скорости), затем применили FLIP
как отдельный шаг, с полной честной walk-forward дисциплиной.

- **Нашли и исправили критичный баг до трат compute**: код по умолчанию
  делит train/val по ДОЛЕ выборки (последние 15%). У DOGE ~8 лет истории —
  это безопасно попадает в 2024H2. У SOL/XRP история короче, и тот же
  сплит по доле утёк прямо в Oct 2025 – Sep 2026, то есть внутрь
  fit-периода 2025 И честных 2026-окон одновременно! Исправлено на явные
  календарные даты (train `< 2024-07-01`, val `2024-07-01..2024-12-31`,
  fit `2025`, honest eval `2026+` — как у DOGE);
- 15 сидов на актив (не 100 как у DOGE — компромисс по скорости), чемпион
  выбирается по H6-fixed-horizon backtest return, отдельно best-by-return
  и best-by-calmar;
- FLIP-слой: та же dynamic temperature calibration (актив-агностична по
  построению) + grid порогов на fit-периоде (2025, быстрый hourly-proxy
  симулятор), финальная честная оценка — **реальная 1-минутная
  симуляция** на touched (Jan-Feb 2026) и untouched (Mar+ 2026);

**Итог по всем 8 комбинациям (2 актива × 2 чемпиона × 2 способа выбора порога):**

| Актив | Комбо | Пороги | Fit 2025 | Touched | Untouched |
|---|---|---|---|---|---|
| SOL | return×by_return | 0.55/0.50 | +79.6% | -24.2% | +1.8% |
| SOL | return×by_calmar | 0.50/0.50 | +85.4% | -19.9% | **+13.3%** |
| SOL | calmar×by_return | 0.80/0.95 | +12.8% | -12.8% | -7.8% |
| SOL | calmar×by_calmar | 0.80/0.95 | +12.8% | -12.8% | -7.8% |
| XRP | все 4 (тот же seed=0) | 0.60/0.65 | +83.8% | -10.5% | **-53.7%** |

- **XRP: чистый, однозначный провал** — все 4 комбинации сходятся к одному
  и тому же seed=0, `-53.7%` untouched — та же магнитуда провала, что и
  zero-shot перенос на HYPE в EXP-060 (`-52.4%`). Переобучение с нуля не
  помогло вообще;
- **SOL: надёжного edge нет, несмотря на один красивый результат.**
  Единственный положительный untouched (`+13.3%`) принадлежит той же
  модели/порогам, которые проиграли `-19.9%` в непосредственно
  предшествующем touched-окне — настоящий устойчивый edge не должен так
  резко менять знак между соседними многомесячными окнами на фиксированной
  стратегии; больше похоже на шум. `champion_calmar` при этом равномерно
  отрицателен на обоих 2026-окнах;
- **Вывод: "переобучить нативно" не оказалось лёгким фиксом.** И
  zero-shot перенос (EXP-060), и нативное переобучение (эта серия, с
  архитектурой DOGE) не дают надёжного OOS edge на двух лучших по
  корреляции кандидатах для диверсификации (SOL, XRP). Это не доказывает,
  что такой модели не может существовать в принципе — полноценный
  per-asset Optuna-поиск (здесь сознательно пропущен) или другой набор
  фичеймог бы сработать иначе — но короткий путь "взять архитектуру DOGE,
  просто переобучить веса" не сработал;
- диверсификация-по-корреляции и диверсификация-по-рабочей-стратегии
  остаются двумя разными, всё ещё нерешёнными вопросами.

Артефакты: `train_new_asset_h6.py`, `flip_layer_new_assets.py`,
`causal_flip/reports/cross_asset_native_training/{REPORT.md,SOLUSDT/,XRPUSDT/}`,
`causal_flip/reports/cross_asset_native_flip/{SOLUSDT,XRPUSDT}/summary.json`.

---

## ЗАКРЫТИЕ ФАЗЫ ИССЛЕДОВАНИЯ (2026-09-11)

По решению пользователя: активный поиск нового edge/улучшений
приостановлен. Итоговый протокол (что реально задеплоено, что подтверждено,
что закрыто, план мониторинга) — в `FINAL_PROTOCOL.md` (корень проекта).

**Про два набора порогов** (уточнено): на сервере через `.env` задано
`0.48/0.52` — именно это и использовал EXP-059 как "продакшн", то есть отчёт
верен. В чекпоинте `prod_doge_1h_v1/model.pt` лежит `0.49/0.55`, который
`runtime.py` берёт только если env-переменные не заданы. Для справки, ячейка
`0.49/0.55` из того же грида: untouched `+34.9%`, MDD `-30.4%`, ранг 159/598.
Обе ячейки в одной области, выводы не меняются. Расхождение между
`model_card.json` и деплоем стоит устранить.

---

## EXP-062 — живой бот торгует не ту стратегию (2026-09-11)

**Статус:** production-баг найден и локализован.

Реконструкция реального P&L живого бота из серверного лога ордеров
(11 апр – 20 авг 2026, 16 закрытых сделок): **+1.57%**, win rate `43.8%`,
MDD по закрытиям `-19.5%`. Бэктест за то же окно обещает `+21.4%` при win rate
`64.5%`. Живой бот при этом сделал 17 флипов там, где пайплайн делает 121.

**Диагностика по серверным почасовым тикам** (3438 записей с залогированным
`prob_up`, который реально видела живая модель):
- корреляция серверного и пайплайнового `prob_up` — `0.23`, средняя абсолютная
  разница `0.24`; средний `prob_up` `0.267` против `0.492`; доля LONG-часов
  `8.4%` против `53.9%`;
- пересчёт сигнала из серверных же вероятностей даёт 100% совпадение с
  залогированным сигналом — правило входа и пороги исправны, ломаются признаки;
- гипотеза «`DATA_START=2024` вместо 2018» **отвергнута**: из 129 признаков
  только один расходится больше `0.01`, вероятности идентичны (`0.493`/`0.494`);
- гипотеза «частичные 4h/1d свечи» **отвергнута**: разрыв постоянен во все 24
  часа (`-0.215…-0.238`), корреляция с часом суток `-0.03`;
- **ablation внешних источников попал точно**: без макро и Fear&Greed средний
  `prob_up` после T=0.8 равен `0.270` при доле LONG `6.2%` — против фактических
  серверных `0.267` и `7.6%`. Контрольная ablation (убрать ETH) смещения не даёт.

**Вывод:** на сервере недоступны yfinance-макро (SPY/QQQ/VIX) и Fear&Greed,
`load_all` молча возвращает пустые фреймы, 13 из 129 признаков залипают на
дефолтах, и модель уходит в постоянный шорт. Локальные кэши этих источников
полные и свежие, так что проблема сетевая/серверная.

**Следствия:** живой результат `+1.57%` не является свидетельством ни за, ни
против исследованной стратегии — последние 4.5 месяца торговалась сломанная
почти-всегда-короткая версия, и весь «forward OOS с марта 2026» к модели не
относится. Это не значит, что после починки будет `+21%`: все прежние оговорки
к бэктесту остаются в силе.

**До возобновления мониторинга:** починить фиды; заставить `load_all` падать
при пустом/устаревшем внешнем источнике вместо подстановки дефолтов; включить
почасовое логирование live feature snapshots; отсчитывать форвард-статистику
заново с момента починки.

**Подтверждено сквозным воспроизведением (не только ablation).** Точный
механизм: `download_fear_greed`/`download_macro_yfinance` при устаревшем кэше
и упавшем сетевом запросе возвращают пустой `DataFrame` вместо отката на
последний известный кэш — самоподдерживающийся отказ без предела по времени.
Воспроизведено через настоящий `load_all()`→`build_features_1h()`→инференс с
намеренно упавшей сетью для этих двух источников: средний `prob_up` `0.295`,
доля LONG `6.2%` — почти точно как у сервера (`0.267`/`7.6%`), радикально
отличается от здорового пайплайна (`0.49`/`46-54%`).

**Проверка реального счёта на бирже (read-only, `/fapi/v2/positionRisk`,
`/fapi/v1/userTrades`, `/fapi/v1/income`).** Актуальная позиция на момент
проверки: шорт `17,370 DOGEUSDC`, вход `0.09113` (открыт 2026-09-08),
нереализовано `+$119.7`, плечо `5x` (в локальном `.env` стоит `LEVERAGE=1` —
расхождение с реальной настройкой на бирже, требует сверки). Полная история
с 1 апреля 2026 (`/fapi/v1/income`, 818 записей): `REALIZED_PNL +207.16`,
`FUNDING_FEE +17.69`, `COMMISSION -22.28`, итого **+202.56 USDC** реализовано
(+322 USDC с учётом текущей открытой позиции) на балансе `~1700.76 USDC`.
Это ощутимо больше, чем оценка `+1.57%`, названная ранее (та была построена
на устаревшем и неполном логе ордеров, обрубленном на 20 августа). Помесячно:
апрель `-77`, май `-205`, июнь `+258`, июль `+91`, август `+150`, сентябрь
пока `-9`. Учитывая подтверждённую поломку признаков (шорт-смещение с
апреля), это выглядит как везение сломанной почти-всегда-шорт модели на
преимущественно падающем DOGE, а не как валидированный edge — то есть
итоговый плюс не является контраргументом против диагноза EXP-062.

**Артефакты:**
- `causal_flip/reports/live_vs_pipeline/REPORT.md`
- `causal_flip/diagnose_live_vs_pipeline.py`
- `causal_flip/diagnose_feature_gap.py`
- `causal_flip/diagnose_feature_ablation.py`
- `causal_flip/reproduce_live_bug.py`
- `causal_flip/reports/live_vs_pipeline/{hourly_comparison,prob_by_cache,ablations,feature_scaled_diff_2018_vs_2024}.csv`
- `analysis_binance_flip_causal_1m/server/income_history_live.csv`

---

## EXP-063 — настоящая причина шорт-перекоса: баг кэширования свечей (2026-09-12)

**Вердикт EXP-062 отменён.** По инструкции пользователя подключился к серверу
по SSH (доступ найден в удалённом ранее `SERVER_COMMANDS.md`), проверил
доступность данных напрямую и нашёл: macro/Fear&Greed фиды **здоровы**
(678/984 дня истории без единого сбоя с 30 мая, хэши кода/модели совпадают
с локальными). Прямое воспроизведение сигнала на РЕАЛЬНОМ серверном кэше
(скачан по SFTP) дало корреляцию `0.93` с тем, что сервер реально
залогировал в `stats_binance_flip.sqlite` (4468 тиков, 20 запусков, но один
непрерывный run с 9 марта) — то есть сервер считает именно то, что должен
на здоровых входных данных. Диагноз EXP-062 был ложным срабатыванием
(случайное совпадение агрегатной статистики при `correlation=0.37` в
исходном тесте с намеренно упавшей сетью — этот низкий коэффициент нужно
было счесть red flag ещё тогда).

**Настоящая причина:** `bot_binance_flip/src/data_loader.py::download_spot_klines`.
`_is_fresh()` (порог свежести = 2 интервала) проверяет только возраст
последнего бара в кэше, не проверяя, закрыт ли он. Binance всегда отдаёт ещё
формирующуюся последнюю свечу при живом запросе. После сохранения в кэш
следующий часовой цикл видит её как «свежую» и не перезапрашивает —
инкрементальная дозагрузка стартует с `last_ts + step`, то есть уже
сохранённая последняя строка никогда не перезаписывается. Итог: каждая
часовая/4h/1d свеча навсегда замораживается в частичном состоянии (обычно
первые несколько минут часа).

**Подтверждено на 216 часах (1-10 июня 2026):** истинный `(high-low)/open`
(из 1-минутных баров) в среднем **в 13 раз больше** (`ratio` 0.03-0.34,
среднее 0.077), чем то, что было в кэше бота. Обнаружено случайно при
попытке понять, почему `causal_flip`-исследовательский пайплайн (агрегация
из 1m) даёт сбалансированные ~49% LONG, а сервер — экстремальный шорт-перекос,
хотя оба используют один и тот же `build_features_1h`/`load_all` и одну и ту
же модель (хэши совпадают). Обнаружился 1-часовой сдвиг при выравнивании
двух серий, который при разборе оказался не сдвигом лейбла, а систематическим
обрезанием диапазона в нативном REST-кэше бота.

**Прямое сравнение (2026-09-11 18:00/19:00 UTC), сервер (на баге) vs
корректный расчёт (1m-агрегация):**

| Час | Сервер (баг) | Корректно |
|---|---|---|
| 18:00 | `prob_up=0.359` → SHORT | `prob_up=0.799` → **LONG** |
| 19:00 | `prob_up=0.391` → SHORT | `prob_up=0.757` → **LONG** |

Противоположные направления с высокой уверенностью в обе стороны. Sanity-check:
DOGE дал импульс +5.5% (12:00-14:00 UTC 11 сентября), после чего держится
выше уровней начала дня — правдоподобная причина разворота в LONG у модели,
видящей полный диапазон.

**Фикс (согласован с пользователем, вариант «полная перекачка»):**
1. В `download_spot_klines` убран ранний выход по `_is_fresh()`; инкрементальная
   дозагрузка теперь стартует с `last_ts` (не `last_ts + step`), то есть всегда
   перезапрашивает и перезаписывает последний сохранённый бар.
2. Локально: `force_refresh=True` пересборка кэша 1h/4h/1d для DOGE/BTC/ETH.
   Проверено: `(high-low)/open` восстановлен до `1.334%` (было `0.092%`);
   сигнал на 17:00/18:00/19:00 UTC 11 сентября — LONG (0.846/0.801/0.759),
   практически совпадает с расчётом через 1m-агрегацию.
3. Задеплоено на сервер (<server>) по SFTP: исправленный `data_loader.py`
   + пересобранный кэш, с бэкапом старых файлов
   (`data_loader.py.bak_2026-09-12`, `data_cache_flip_bak_2026-09-12/`).
   Бот перезапущен (`systemctl restart bot_binance_flip`).
4. Прямой прогон инференса на сервере (тот же venv/код/модель, `force_refresh=False`
   на уже пересобранном кэше) подтвердил: `range_check=1.334%`, сигнал на
   19:00/20:00/21:00 UTC — LONG (`prob_up` 0.7594/0.7040/0.7700).

**Три независимых метода (1m-агрегация, локальная полная перекачка,
прогон на сервере) сходятся на одном выводе: LONG.**

**Следствие для позиции:** встроенная FLIP-логика бота (`trading_bot.py`,
строки ~861-866: `close(reason="flip")` → `open(sig=LONG)` при смене знака
сигнала) должна сама развернуть текущий шорт (17,370 DOGEUSDC от 2026-09-08,
вход 0.09113, плечо 5x) в лонг на следующем часовом цикле — без ручного
вызова API. Решение по позиции принято по правилу, заданному пользователем:
«если предикт другой — меняем направление».

**Открытые вопросы / TODO:**
- Аналогичная логика (`_is_fresh` + resume-only-forward) есть в
  `download_oi_history`/`download_funding_rate`/`download_ls_ratio`, но эти
  функции делают полный re-fetch при устаревании (без инкрементального
  resume), поэтому риск навечного замораживания там ниже — не тронуты
  в этом фиксе, но стоит пересмотреть отдельно.
- `download_fear_greed`/`download_macro_yfinance`/`download_btc_dominance`
  используют дневное разрешение с порогом 2-3 дня — тот же класс риска
  (частичное значение "сегодня", если запрос идёт внутри дня) теоретически
  возможен, но эффект намного меньше (дневные бары, а не часовые OHLC) —
  не тронуты в этом фиксе.
- Нужно понаблюдать за ботом после фактического флипа (не позднее следующего
  часового цикла) и подтвердить, что позиция реально перевернулась на бирже.
- Весь накопленный live-трек до 2026-09-12 (весь `stats_binance_flip.sqlite`)
  собран на сломанном пайплайне — как источник статистики по эффективности
  стратегии он непригоден; отсчёт live-трека нужно начинать с момента фикса.

**Артефакты:** `bot_binance_flip/src/data_loader.py` (пофикшен),
`causal_flip/verify_against_real_server_cache.py`,
`causal_flip/correct_current_signal.py`,
`causal_flip/reports/live_vs_pipeline/repro_vs_real_server_cache.csv`,
`analysis_binance_flip_causal_1m/server/ticks_full_sqlite.csv`,
`causal_flip/reports/live_vs_pipeline/REPORT.md` (обновлён с отменой EXP-062
и разделом «ПЕРЕСМОТР»), `FINAL_PROTOCOL.md` (обновлён стоп-флаг).

---

## EXP-064 — измерение оверфита, бычий режим и односторонний хвостовой риск (2026-09-12)

Серия диагностик, запущенная вопросом «куда двигаться дальше» и вопросом
пользователя «может, мы предсказываем не то». Все расчёты на уже имеющихся
данных, без новых загрузок.

### 064a — размер оверфита, измеренный per-trade

Разбиение того же бэктеста (`analysis_flip_causal_2025_2026/trades.csv`, 420
сделок) по тому, что реально видел поиск порогов:

| Период | Сделок | Мес. | Итого | %/мес | MDD | Средний net на сделку |
|---|---:|---:|---:|---:|---:|---:|
| Весь бэктест | 420 | 19.6 | +8131% | 25.17% | −32.4% | 1.184% |
| 2025 (**fit**, здесь выбирались пороги) | 263 | 12.0 | +6170% | 41.35% | −25.9% | **1.745%** |
| 2026 Jan-Feb (частично засвечен) | 28 | 1.9 | +0.1% | 0.06% | −23.6% | 0.099% |
| 2026 Mar+ (**untouched**) | 129 | 5.7 | +31.1% | **4.83%** | −28.5% | **0.274%** |

Вклад периодов в итоговый множитель `x82.3`: 2025 → `x62.70`, Jan-Feb → `x1.00`,
Mar+ → `x1.31`. **Весь headline происходит из fit-периода.**

Правильная метрика здесь — средний net на сделку, потому что она не зависит ни
от компаундинга, ни от размера счёта: **просадка edge с 1.745% до 0.274%, то
есть оверфит 6.4x**. Этой разбивки в EXP-059 не было (там были untouched-цифры,
но без сопоставления с fit в терминах per-trade edge).

### 064b — бычий режим невозможно протестировать

`prod_doge_1h_v1/model_card.json`: `train 2018-01-01..2024-06-30`,
`val 2024-07-01..2024-12-31`, `test 2025-01-01..2025-12-31`. Бычьих лет у DOGE
ровно два (2021: +2898%, 2024: +243%), и **оба целиком внутри train/val**. Всё
после 2025-01 — медвежий режим (2025: −63.9%, 2026: −34.1%).

**Ограничение №1 протокола («не протестировано в устойчивом бычьем рынке») этой
моделью на этом активе структурно незакрываемо** — непросмотренных бычьих
данных не существует. Корректный путь потребовал бы переобучения с окном без
2021 года (leave-the-bull-out), т.е. повторения всей процедуры отбора чемпиона.

Контекст распределения (buy & hold DOGE, дневки 2019-07..2026-09, 86 месяцев):
средний месяц `+16.94%`, **медианный `−2.92%`**, лучший `+692%` (янв 2021),
MDD `−92.3%`, геометрически `+3.63%/мес`. 29.1% месяцев дают >+10%, 17.4% >+30%.
**21.7% скользящих 12-месячных окон превышают 10%/мес, 13.5% — 30%/мес.** То есть
цель «10-30% в месяц» в этом активе достижима, но доход сосредоточен в правом
хвосте из горстки месяцев, а плата за право в них находиться — просадка 92%.

### 064c — трендследование на полном цикле (единственный честный OOS-тест)

Правило `sign(close[t]/close[t-N]-1)` имеет **ноль обученных параметров**,
поэтому его прогон на 2019-2026 включая 2021/2024 законен как OOS.
Дневные решения, исполнение на следующем часовом открытии, `4bps fee + 2bps slip`
на сторону, funding где доступен (с 2020-07). Публикуются все длины окна.

| Правило | Итого | %/мес | MDD | Разорение |
|---|---:|---:|---:|---|
| 12h long/short | −100% | — | −100% | **2021-01-28** |
| 24h long/short (окно EXP-049) | −100% | — | −100% | **2021-01-28** |
| 48h long/short | −100% | — | −100% | **2021-01-28** |
| 72h long/short | −100% | — | −100% | **2021-01-28** |
| 168h long/short | −100% | — | −100% | **2021-01-28** |
| 12h long-only | +189.9% | +1.24% | −93.2% | — |
| 24h long-only | +338.6% | +1.73% | −85.4% | — |
| 48h long-only | +1363.3% | +3.17% | −77.9% | — |
| 72h long-only | +2810.7% | +4.00% | −70.6% | — |
| 168h long-only | +911.8% | +2.73% | −86.7% | — |
| Buy & hold | +2029.2% | +3.61% | −93.2% | — |

**Все симметричные long/short варианты погибают 2021-01-28** (DOGE вырос ~4.7x
за сутки; шорт 1x потерял 470% за день). Это не просадка, а обнуление.
Long-only выживают все; 72h бьёт buy & hold по обеим осям. Оговорка: n=5 на
одном активе — слабая статистика, хотя форма зависимости (рост до 72h, спад на
168h) выглядит гладкой. Замороженное окно 24h из EXP-049 оказалось одним из
худших long-only вариантов — вопрос к робастности самого EXP-049.

Накопленный funding: −44%…−52% для long-only (лонг платит в те 78% периодов,
когда ставка положительна; средний funding DOGE `0.00685%`/8ч ≈ 7.5% годовых,
но с распадом: 2024 +14.0%, 2025 +4.3%, 2026 ≈ +2.7% годовых).

### 064d — где на самом деле находится хвостовой риск FLIP

Прогон продакшн-модели (пороги `.env` 0.48/0.52, dynamic temp) вокруг сквиза.
**Оговорка: 2021 в train-окне, поэтому позиционирование смещено в пользу
модели; это оценка риска, не оценка мастерства.**

Модель была **LONG все 24 часа 2021-01-28** (+471.2% за день), перевернувшись в
лонг 27-го в 15:00 — за 9 часов до сквиза, против падающей цены. С высокой
вероятностью это **запоминание** уникального события, а не распознавание
паттерна (18 МБ весов, обучение на этих данных с таргетом H6).

Опасность пришла на следующий день: **2021-01-29 02:00 модель в SHORT, цена за
час 0.04899 → 0.08329 (+70.0%) → −70.0% на счёте за 60 минут.** Разорения при
1x не случилось (MDD окна −70.7%), но **порог фатальности этого часа — 1.43x
ноционала**.

Продакшн работает на ~1.0x (`POSITION_SIZE_PCT=95`, ноционал 1706 при балансе
1700; биржевая настройка `leverage 5` задаёт лишь максимум и дистанцию
ликвидации, а не размер — ложная тревога из EXP-063 закрыта). **Запас 43%.**

**Следствие для целей по доходности:** чтобы получить 10-30%/мес из этого edge
(untouched 4.83%/мес), нужен множитель размера 1.8-5.3x. Любой множитель выше
1.43 превращает исторически переживаемое событие в фатальное. Путь к 10-30%/мес
через плечо тождественен пути к обнулению.

**Артефакты:** `causal_flip/trend_full_cycle_2019_2026.py`,
`causal_flip/flip_on_2021_squeeze.py`,
`causal_flip/reports/trend_full_cycle/` (`yearly_frozen_24h.csv`,
`robustness_lookbacks.csv`, `segments_frozen_24h.csv`,
`flip_signals_2021_squeeze.csv`, `flip_signals_full_history.csv`).

**Проверка целостности данных:** исследовательский кэш `*_2018-01-01.csv`
сверен с пересобранным после фикса `*_2024-01-01.csv` на 23611 барах
перекрытия — соотношение диапазонов `1.0000` почти во всех месяцах, расхождение
только в 8 барах. Баг EXP-063 задел лишь инкрементальный live-кэш, массовые
исследовательские загрузки чистые.

---

## EXP-065 — асимметричное снижение шортов: плоское vs волатильностное (2026-09-12)

**Гипотеза (от пользователя):** раз хвостовой риск односторонний, снижать объём
именно на шортах, а не плечо целиком. Исходная формулировка — «при просадке».

**Метрика — ruin-риск, не MDD.** MDD слеп к разнице между «−70% и выжил» и
«−100% и умер», а протокол ранее отклонял все 5 risk-overlay механизмов (включая
equity-drawdown throttle и vol-scaled sizing) именно по MDD. Правильная
величина — **худшая накопленная просадка внутри одной позиции**, потому что
ликвидация срабатывает на ней, и производный от неё `safe_notional_x`
= множитель ноционала, при котором это событие становится обнулением.

### Условие «при просадке» эмпирически не работает

Худший час (2021-01-29 02:00, −70%) случился, когда счёт стоял на **историческом
максимуме**: просадка перед ним `+13.58%` к предыдущему пику (эквити x622.68 при
пике x622.68). Из 6 опасных часов throttle поймал бы 4 — но только мелкие
(−25.9%, −25.1%, −15.0%); смертельный пропускает. Причина структурная:
катастрофа приходит **сразу после крупного выигрыша**, когда волатильность
взорвалась, а просадка по определению запаздывает.

### Асимметрия обоснована: шорты не являются основным заработком

| Период | Вклад шортов | Вклад лонгов |
|---|---:|---:|
| Все 420 сделок | x13.01 (winrate 67.1%) | x6.33 (61.9%) |
| 2025 (fit) | x10.27 | x6.10 |
| **2026 Mar+ (untouched)** | **x1.16** | **x1.13** |

На fit-периоде шорты вдвое сильнее, но в честном окне стороны **практически
равны** — перекос в пользу шортов есть артефакт периода настройки. Значит
урезание шортов стоит пропорциональную, а не катастрофическую долю edge.

### Результат по периодам

| Период | flat 1.0x (база) | vol>p95 → шорт 0.3x |
|---|---|---|
| TRAIN 2019-07..2024-06 | запас **1.42x** | запас **3.06x** |
| VAL 2024-07..12 | 3.81x | 3.81x |
| TEST 2025 | 4.74x | 4.74x |
| **UNTOUCHED 2026-03+** | **4.62x**, +41.66% | **4.62x**, +39.65% |

**Весь выигрыш волатильностного фильтра происходит из одного события — сквиза
января 2021.** На трёх периодах после обучения он не улучшает запас вообще
(худшая внутрипозиционная просадка везде идентична), зато стоит доходности.
Результат с n=1.

**Переформулировка задачи:** запас в нормальных условиях `3.8-4.7x`, а цифра
`1.42x` — целиком артефакт 2021 года. То есть риск не «хронически тонкий запас»,
а «комфортный запас плюс одно событие, которое почти убило». Это **решение про
страховку от хвоста, а не задача оптимизации**, и статистически оно не
закрываемо по построению: для валидации защиты от события, случившегося раз в
7 лет, нужно много таких событий.

Цена страховки на честном окне (6 месяцев): `vol>p95 → 0.3x` стоит −2 пп
(41.66% → 39.65%) при запасе 3.06x в train; `flat 0.6x` стоит −9 пп
(41.66% → 32.58%) при запасе 2.38x. Волатильностная версия эффективнее вчетверо,
но у неё два параметра, откалиброванных на n=1.

**Вывод/рекомендация:** брать **плоскую асимметричную** (шорты меньше лонгов,
безусловно). Разорение — состояние невосстановимое, поэтому страховка разумна;
но выбирать хитрую версию, чья эффективность обоснована одним наблюдением,
значит повторять ошибку EXP-062 (`correlation=0.37` принята за причинность) и
ошибку fit-периода (разрыв 6.4x). У плоской версии цена **известна**, а не
оценена по одному случаю. Ориентиры: `0.8x` ≈ −2 пп / запас 1.79x,
`0.6x` ≈ −9 пп / запас 2.38x. Конкретная величина — вопрос аппетита к риску,
не вопрос данных.

**Не внедрено в боевого бота.** Требует отдельного решения и деплоя.

**Артефакты:** `causal_flip/asymmetric_short_sizing.py`,
`causal_flip/reports/asym_short_sizing/ruin_risk_comparison.csv`.

## EXP-075 — Transformer 3-класса (long/short/flat) FLIP на тик-точных dollar bars BTC (2026-09-13)

**Задача пользователя (буквально):** настоящий трансформер, все фичи, потиковое
обучение, long/short/flat, предикт на 20 баров вперёд, flip при смене предсказания,
комиссии учтены, критерий — Калмар и доходность, momentum обязательно в фичах.

**Датасет:** `btc_bars_with_features.parquet`, 411,911 tick-precision dollar bars
BTCUSDT (2022-11-06 .. 2026-09-11). Фичи: depth imbalance (±1..5%), positioning
divergence, OI change, taker vol ratio, flow imbalance, realized vol, trade
intensity, + возвращённый momentum (`mom_5/20/60/120`). Лейбл: fwd_ret через 20
баров, 3 класса по fee-aware порогу (`> +8bps` → long, `< -8bps` → short, иначе
flat). Баланс классов: 45.4% / 43.9% / 10.7% — не вырожденный.

**Модель:** causal-masked TransformerEncoder (2 слоя, d_model=64, nhead=4,
seq_len=64 баров), обучен на GPU (RTX 5080) отдельно на каждом из 5 walk-forward
фолдов (2024H2..2026H2_YTD), train строго до начала фолда, скейлинг фич
(mean/std) — тоже только по train. Инференс — предсказание класса на каждом баре
эвал-окна.

### Первый прогон (сырой argmax каждый бар): катастрофа, но не из-за точности

Все 5 фолдов: **-91% .. -99.9%** итог, Calmar ≈ -1.0 везде. Причина —
**не ошибка направления**, а частота: 6400-8500 флипов на ~52k баров/фолд
(смена предсказанного класса почти каждый бар — 3-way softmax шумит около
1/3 базовой линии). `0.9992^8000 ≈ e^-6.4 ≈ -99.8%` — именно комиссии на
рефлекторных флипах съедают счёт вчистую, независимо от того, права модель
или нет.

### Фикс: hysteresis (confidence threshold + N-бар персистентность)

Без переобучения — по кэшированным вероятностям (`probs_*.npz`) прогнан грид
`conf_thresh ∈ [0.34..0.65] × confirm_bars ∈ [1..40]` (флип фиксируется только
если argmax держится ≥ confirm_bars баров подряд при вероятности ≥ conf_thresh;
иначе позиция не меняется). Результат — **широкая, не точечная** зона
положительного Calmar при `confirm_bars≈20`: даже без порога уверенности
(`conf_thresh=0.34`) один персистентный фильтр разворачивает -99% в
Calmar +0.25; лучшая точка **`conf_thresh=0.60, confirm_bars=20`**:

| Фолд | Return | Calmar | Monthly (implied) |
|---|---:|---:|---:|
| 2024H2 | +138.0% | — | +15.6%/мес |
| 2025H1 | +48.8% | — | +7.0%/мес |
| 2025H2 | +15.3% | — | +2.4%/мес |
| 2026H1 | +47.0% | — | +6.8%/мес |
| 2026H2_YTD | 0.0% (без сделок) | — | 0.0%/мес |
| **Итог (chained, 2.18 года)** | **+500.2%** | **4.27** | — |

MDD (chained) = **-29.9%** (впритык к лимиту 30%), winrate по сделкам 82% (9/11),
прибыльно в 4 из 5 фолдов, в пятом — не убыток, а просто нулевая активность
(порог не пробит).

### Честная оговорка (tail-check, как для momentum ранее)

Всего **11 сделок** за 2.18 года. Топ-3 сделки дают **65% суммарного
лог-return** — концентрация есть, но распределена по **трём разным годам**
(2024, 2025, 2026), а не по одному эпизоду (в отличие от DOGE-2021 у momentum).
Худшие 2 сделки — маленькие (-3.5%, -4.8%), не катастрофы. Время удержания
позиции — от 350 баров (~1 день) до 40,000 баров (**несколько месяцев**): чтобы
выжить после комиссий, персистентность обязана быть большой, из-за чего система
**перестаёт быть «flip на 20 барах»** и становится низкочастотным
трендовым/режимным фильтром — по характеру близким к уже известному
momentum/FLIP-подходу, только сигнал теперь генерирует модель, а не готовое
правило.

**Статус: не готово к бою.** 11 сделок — слишком тонкая выборка для уверенности,
среднемесячная доходность (2.4-15.6%/мес по фолдам) не всегда попадает в
целевые 10-30%, и нужна проверка за пределами конкретной точки грида (0.60/20)
на других символах/периодах прежде чем считать edge подтверждённым.

**Артефакты:** `causal_flip/ml_dataset/train_transformer_flip.py`,
`causal_flip/ml_dataset/flip_threshold_grid_transformer.py`,
`causal_flip/ml_dataset/inspect_transformer_trades.py`,
`causal_flip/reports/ml_transformer_flip/{transformer_flip_results.csv,
hysteresis_grid.csv, winning_config_trades.csv, probs_*.npz}`.

## EXP-075d — проверка на замаскированный buy&hold (2026-09-13)

**Вопрос пользователя:** нет ли штрафа за долгое пребывание в одной сделке —
может, "модель" просто держит одну ставку всю дорогу?

Проверка: для каждого фолда посчитан **наивный one-shot** (взять первое
направление модели и держать без единого дальнейшего флипа) и сравнён с
реальной multi-flip стратегией (EXP-075, `conf_thresh=0.60, confirm_bars=20`).

| Фолд | Реальная | Наивная (1 ставка) | Buy&hold |
|---|---:|---:|---:|
| 2024H2 | +138.0% | -46.3% | +49.2% |
| 2025H1 | +48.8% | -23.0% | +14.5% |
| 2025H2 | +15.3% | **+15.3% (идентично)** | -18.2% |
| 2026H1 | +47.0% | **+48.2% (наивная лучше)** | -33.2% |

Подозрение подтвердилось частично: в 2024H2/2025H1 реальная стратегия сильно
обгоняет наивную (флипы добавляют настоящую ценность — модель поймала
разворот). В 2025H2/2026H1 результат **неотличим от одной направленной
ставки** — это и есть замаскированный buy&hold. Дополнительно: сырой (до
hysteresis) argmax внутри одного "уверенного" удержания флипает 1000-4500 раз
на 8-35 тыс. баров — под фильтром персистентности прячется по сути монетка,
а не стабильный режим; confirm_bars=20 работает как медленный low-pass
(trend-following), а не как детектор дискретных состояний.

**Причина:** per-bar cross-entropy loss не штрафует и не поощряет
длительность удержания вообще — вся "устойчивость" исходит из
hysteresis-фильтра на инференсе, а не из целевой функции обучения.

## EXP-076 — decay/reconfirmation фикс + честный objective (2026-09-13)

**Задача пользователя (буквально):** штрафовать за длительные
однонаправленные удержания / вознаграждать за реальные флипы, чтобы модель
училась таймингу, а не одной ставке.

**Механизм (без переобучения, на тех же кэшированных probs):** позиция
остаётся открытой только пока скользящее среднее вероятности удерживаемого
класса за последние `decay_window` баров держится ≥ `decay_thresh`; если
уверенность "протухла" — выход в flat даже без противоположного сигнала.
Явно убивает поведение "подтвердил один раз — катайся месяцами на стухшем
сигнале" без жёсткого лимита времени.

**Objective для отбора параметров сменён:** вместо raw Calmar/return —
`flip_added_value = реальный_return − наивный_one_shot_return` на фолд,
отбор по **worst-case (min) по фолдам**, а не по средней/итоговой цифре —
конфиг не может выиграть, спрятав одну удачную одностороннюю ставку.

Грид 180 комбо (`conf_thresh × confirm_bars × decay_window × decay_thresh`)
по кэшу. Победитель по критерию min_flip_added:
**`conf_thresh=0.55, confirm_bars=10, decay_window=400, decay_thresh=0.42`**:

| Фолд | Return | MDD | Calmar | Наивная | flip_added | Сделок | Max span | Winrate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024H2 | +68.6% | -9.0% | 20.49 | -46.3% | **+114.9%** | 21 | 1.7% | 86% |
| 2025H1 | +18.2% | -4.4% | 9.31 | -22.8% | **+41.0%** | 9 | 0.9% | 78% |
| 2025H2 | -3.6% | -12.5% | -0.57 | -18.1% | **+14.5%** | 8 | 2.7% | 38% |
| 2026H1 | +14.1% | -11.5% | 2.67 | -4.0% | **+18.1%** | 6 | 4.1% | 67% |
| 2026H2_YTD | 0.0% | 0.0% | — | 0.0% | 0.0% | 0 | 0.0% | — |
| **Итог (chained, 2.18г)** | **+119.2%** | **-13.0%** | **3.34** | — | — | **44** | — | **73%** |

Разница с EXP-075 (raw Calmar-selection): **flip_added положителен во ВСЕХ
4 активных фолдах** (было 2 из 4), максимальная доля одной направленной
сделки в фолде упала с 65-76% до **0.9-4.1%** — больше не замаскированный
buy&hold. Сделок стало 44 (было 11) — выборка менее тонкая. Топ-3 сделки
по вкладу: 10.4%/7.5%/7.4% (было 65% на топ-3) — концентрация хвоста
устранена. MDD упал вдвое (-13% против -29.9%). Итоговый Calmar ниже
(3.34 против 4.27), но честнее.

Проверка соседних ячеек грида (heatmap по `conf_thresh × decay_thresh` и
`confirm_bars × decay_window`) показала **широкую положительную зону**
(Calmar 1-6 почти везде в регионе `conf∈[0.55,0.65], decay_thresh∈[0.38,0.50]`),
не единичный пик — механизм устойчив к выбору точных чисел. Найдена ячейка
с Calmar 6.58 (`confirm=20, decay_window=200`), но у неё `flip_added`
**отрицателен в 2 из 4 фолдов** — отбракована тем же критерием, которым
её и нашли бы при наивном отборе по Calmar (напоминание, почему raw-return
selection доверять нельзя).

**Помесячная доходность (implied):** 2024H2 +9.1%/мес, 2025H1 +2.9%/мес,
2025H2 -0.6%/мес (убыточный), 2026H1 +2.3%/мес, 2026H2_YTD 0%. **Ниже
целевых 10-30%/мес в большинстве периодов**, один фолд убыточен. MDD
(-13%) далеко от лимита 30% — запас есть, но за счёт консервативности,
не за счёт достижения целевой доходности.

**Статус: честнее, но не готово к бою.** Реальная timing-ценность теперь
подтверждена во всех активных фолдах (не совпадение), MDD и tail-риск
радикально ниже — но headline-доходность тоже ниже, и до целевых 10-30%/мес
не дотягивает. Следующий шаг — обсуждается с пользователем (Optuna с этим
же flip_added-objective, либо расширение выборки на другие монеты/историю).

**Артефакты:** `causal_flip/ml_dataset/check_buyhold_disguise.py`,
`causal_flip/ml_dataset/flip_quality_grid.py`,
`causal_flip/reports/ml_transformer_flip_btc/flip_quality_grid.csv`.

## EXP-077 — тот же пайплайн на DOGEUSDT: не переносится (2026-09-13)

**Задача:** проверить, воспроизводится ли EXP-076 (transformer + decay/
reconfirmation) на DOGE тем же кодом/фичами/архитектурой, без изменений.

Докачаны `bookDepth` (1348 дней, 2023-01-01+) и `metrics` (1747 дней,
2021-12-01+) для DOGEUSDT (`aggTrades` уже были). Построено 421,176 тик-точных
dollar bars (2022-11-06..2026-09-11, сопоставимо с BTC 411,911). Качество
данных аналогично BTC (те же ~3% NaN на depth-фичах, 0-1.6% на positioning).
Скрипты параметризованы (`--symbol`), BTC-кэш переименован в
`ml_transformer_flip_btc/` для консистентности.

Обучение (те же 5 walk-forward фолдов, та же архитектура) на сырых predictions
дало тот же catastrophic collapse, что и BTC до фикса (-78%..-100% на 4 из 5
фолдов, 7000-10000 флипов/фолд) — механизм комиссионного разорения
воспроизводится идентично.

**После decay/reconfirmation грида (180 комбо, тот же протокол что EXP-076):
ни один конфиг не проходит одновременно все три честных критерия**
(`min_flip_added > 0` И `overall_MDD > -30%` И `total_flips >= 10`) — **0 из
180**. Два типа "хороших на вид" конфигов, и оба нечестны по разным причинам:

| Конфиг | Overall Calmar | Overall MDD | flip_added по фолдам | Проблема |
|---|---:|---:|---|---|
| `conf=0.65, confirm=40, decay_win=100, decay_th=0.42` | 4.13 | -12.0% | 0/0/+91.6/0/+9.5 (3 из 5 фолдов — вообще без сделок) | Тонкая выборка: 2 сделки за 2.18 года, activity почти нулевая |
| `conf=0.55, confirm=10, decay_win=200, decay_th=0.46` | 0.12 | **-64.4%** | +49.4/+1.2/+30.1/+21.9/+9.7 (положителен везде!) | Реальный timing-скилл есть во всех фолдах, но 2025H2 даёт внутрифолдовую просадку **-63.5%** — вдвое хуже лимита |

Второй конфиг — самый интересный и самый честный результат: модель
действительно "понимает" DOGE лучше naive-ставки в каждом протестированном
периоде (в отличие от BTC EXP-075, где 2 из 4 фолдов были чистым
buy&hold-in-disguise, тут все 5 показывают добавленную ценность) — но абсолютный
риск профиля не совместим с задачей (MDD -64% против лимита 30%). Это не
"стратегия сломана", а "DOGE волатильнее/более склонен к резким разворотам,
которые эта конкретная связка hysteresis-параметров не успевает поймать
достаточно быстро".

**Вывод:** архитектура/фичи не универсальны между активами "из коробки" —
то же самое ограничение, что уже задокументировано в `FINAL_PROTOCOL.md`
для zero-shot (EXP-060) и native retrain (EXP-061) на SOL/XRP с DOGE-моделью.
Для DOGE нужен **отдельный подбор** hysteresis-параметров (шире грид decay_window/
decay_thresh специально под DOGE-волатильность) или asset-specific
risk-sizing (уменьшенный notional при повышенной реализованной vol) — не
приоритет пока BTC-версия не прошла Optuna-этап.

**Артефакты:** `causal_flip/reports/ml_dataset/{doge_dollar_bars.parquet,
doge_bars_with_features.parquet}`, `causal_flip/reports/ml_transformer_flip_doge/
{probs_*.npz, transformer_flip_results.csv, flip_quality_grid.csv}`.

## EXP-079 — Direct-PnL Transformer (control, no Optuna) (2026-09-13)

**Задача:** перестать учить 20-бар класс и торговать отдельным FLIP.
Модель каждый бар выдаёт непрерывную позицию `[-1, +1]`. Loss =
`-log-return после комиссий` + штраф за MDD>20% + штраф, если результат
хуже лучшей односторонней ставки (long-only или short-only) на том же
чанке. Комиссия 4 bps на единицу `|dPos|` (полный флип = 8 bps).

**Почему так:** EXP-034 (equity-loss GRU) схлопнулся в buy&hold; EXP-075d
подтвердил тот же shortcut. Штраф vs one-shot — явный анти-BH член.
Optuna не запускался: сначала control.

**Walk-forward BTC dollar bars, 7 фолдов:**

| Фолд | Strat | MDD | %/мес | \|pos\| | Long-BH | vs best one-shot |
|---|---:|---:|---:|---:|---:|---:|
| 2023H2 | +25.6% | -20.9% | +3.9 | 0.97 | +38.7% | **-13.1%** |
| 2024H1 | +7.8% | -2.8% | +1.3 | 0.12 | +48.2% | **-40.4%** |
| 2024H2 | +3.1% | -1.9% | +0.5 | 0.02 | +49.2% | **-46.1%** |
| 2025H1 | +5.6% | -12.1% | +0.9 | 0.35 | +14.5% | **-8.8%** |
| 2025H2 | -3.7% | -9.7% | -0.6 | 0.26 | -18.2% | **-15.2%** |
| 2026H1 | -2.5% | -2.9% | -0.4 | 0.04 | -33.2% | **-35.3%** |
| 2026H2_YTD | +0.4% | -0.2% | +0.2 | 0.02 | +31.5% | **-31.1%** |

**Диагноз:** модель почти нигде не торгует (2024H2 `|pos|=0.02`, 2026H1
0 оборотов). Единственный активный фолд — 2023H2, и там это почти
полный лонг, который всё равно проиграл buy&hold. Ни один фолд не
обыграл лучшую одностороннюю ставку. В бычьем 2024 (+48–49% за полгода
у long-only) стратегия взяла +3…+8%. До 10–30%/мес не дотягивает ни
один период.

**Вывод / статус: control провален, Optuna не запускать.** Прямой PnL-loss
на этих фичах не дал тайминга: либо сидит в нуле, либо копирует тренд
хуже простого лонга. Повторяет суть EXP-034 на новых данных. Данные
(тики + полосы стакана 30с + metrics) сами по себе не превращаются в
edge, если цель — обыграть тренд 2024 года внутри одного BTC.

**Артефакты:** `causal_flip/ml_dataset/train_direct_pnl.py`,
`causal_flip/reports/ml_direct_pnl/direct_pnl_results.csv`.

## EXP-080 — тот же Direct-PnL Transformer на обычных 5m свечах BTC (2026-09-13)

**Зачем:** dollar bars убирают календарное время (сессии, фандинг, новости).
Пользователь попросил уйти на обычные свечи. 5m выбран потому что metrics
уже 5-минутные, bookDepth 30с джойнится без дыр, плотность ~288 баров/день.

**Данные:** 406,360 пятиминуток BTCUSDT 2022-11-01..2026-09-11 из aggTrades
+ as-of bookDepth/metrics. Те же 27 фичей, та же модель, тот же PnL-loss,
walk-forward 7 фолдов, без Optuna.

| Фолд | Strat | MDD | %/мес | \|pos\| | Long-BH | vs one-shot |
|---|---:|---:|---:|---:|---:|---:|
| 2023H2 | −38.5% | −39.8% | −7.8 | 0.81 | +38.8% | −77% |
| 2024H1 | +34.6% | −20.6% | +5.1 | 0.81 | +47.8% | −13% |
| 2024H2 | +25.0% | −17.7% | +3.8 | 0.56 | +49.1% | −24% |
| 2025H1 | +5.7% | −14.0% | +0.9 | 0.40 | +14.3% | −9% |
| 2025H2 | −8.4% | −16.1% | −1.4 | 0.40 | −18.2% | −20% |
| 2026H1 | −14.2% | −18.1% | −2.6 | 0.41 | −33.2% | −47% |
| 2026H2_YTD | 0.0% | −0.1% | 0.0 | 0.01 | +31.5% | −31% |

**Сравнение с dollar bars (EXP-079):** на 5m модель **активнее** в бычьем
2024 (держит 0.56–0.81, не сидит в нуле) и берёт больше тренда (+35%/+25%
против +8%/+3%). Но это всё ещё хуже buy&hold, 2023H2 разваливается
комиссиями (8995 оборотов, −39%), 2025–2026 в минусе. Целевые 10–30%/мес
не достигнуты ни в одном фолде.

**Вывод:** смена нарезки (dollar → 5m) не создала edge. Модель лучше
видит бычий 2024, но не умеет ни вовремя выходить, ни шортить медвежьи
окна. Optuna по этой постановке не запускать.

**Артефакты:** `causal_flip/ml_dataset/build_time_bars.py`,
`causal_flip/reports/ml_dataset/btc_5m_bars_with_features.parquet`,
`causal_flip/reports/ml_direct_pnl_5m/direct_pnl_results.csv`.

## EXP-081 — сырые тики без свечей: горизонт 100/500/2000/10000 aggTrades (2026-09-13)

**Вопрос пользователя:** убрать время и свечи, окно 100 тиков → куда цена
через следующие 100 тиков.

**Протокол:** каждый 10-й день 2023-01..2026-09 (135 дней, все режимы),
BTCUSDT aggTrades. Решение на границе непересекающихся окон. Комиссия
8 bps. Сигнал-контроль: flow imbalance за прошлые H тиков (signed
notional taker buy/sell).

| H тиков | Время (мед.) | \|ход\| мед. | доля > 8bps | Flow IC | Naive flow после комиссий |
|---|---:|---:|---:|---:|---:|
| 100 | **4.7 сек** | **1.3 bps** | **0.2%** | +0.19 | **−7.7 bps** |
| 500 | 27 сек | 2.9 bps | 6.5% | +0.03 | −7.9 bps |
| 2000 | 1.8 мин | 5.9 bps | 35% | −0.01 | −8.1 bps |
| 10000 | 9.5 мин | 13.2 bps | 67% | −0.03 | −8.0 bps |

**Вывод:** на 100 тиках сигнал ленты **есть** (IC 0.19 — короткое
продолжение потока), но ход **в 6 раз меньше комиссии**. Winrate
naive-flow 0.1%: почти каждое окно меньше 8 bps, после комиссии
минус почти ровно размер комиссии. На всех горизонтах naive net ≈ −8 bps
— классика «корреляция есть, edge после издержек нет».

Трансформер на «100 тиков → следующие 100» учить нечего: таргет почти
всегда внутри спреда/комиссии. Чтобы ход стал сопоставим с 8 bps, нужно
порядка 10k+ тиков (~10 мин) — и там flow IC уже около нуля. Это снова
возвращает к минутным/5m свечам, только нарезка по числу сделок.

**Артефакты:** `causal_flip/ml_dataset/tick_horizon_probe.py`,
`causal_flip/reports/ml_tick_horizon/horizon_summary.csv`.

## EXP-082 — Binance Copy Trading: «лотерея» vs честный архив лидерборда (2026-09-15)

**Мысль:** копировать топ лидеров Binance Futures Copy — это не edge-модель, а
селекция по публичной таблице. Любой разовый scrape сегодняшних топ-N и прогон
их 180D-кривых назад раздувает return: в выборке только те, кто **уже** выжил
и попал в рейтинг (survivorship). Красивый walk-forward на таких данных —
гипотеза, не доказательство.

**Что посчитали на текущем срезе (~1000 лидеров, ROI+PNL, ~180D chartItems):**
- WF `lb7/h7/top10`: hit ~82%, compound выглядит огромным, MDD ~33% — **но**
  все лиды взяты из сегодняшнего топа → bias.
- «Ранний выстрел» (peak ≥5×): base p≈5.5%; среди **early-hot** (верхний
  квартиль e30) p_shot≈2.6%, среди **cold** ≈11.7%; EV_end cold ≫ hot.
  Интуиция «взять того, кто уже разгорелся» для 5× — хуже случайного холодного.

**Инфра (чтобы мысль можно было проверить честно):** daily Docker-snapshot
топ-500 ROI + топ-500 PNL + 180D charts на LAN `<lan-server>`
(`/mnt/data/copy_leaderboard`, контейнер `copy-leaderboard`, раз в 24h).
Первый снимок `2026-09-15` (~11 MB) уже записан. Через месяцы — point-in-time
панель без ретроактивного отбора выживших; только тогда имеет смысл снова
мерить copy/lottery / early-shot.

**Артефакты:** `market_watch/_copy_lottery.py`, `_copy_early_math.py`,
`copy_leaderboard/` (+ `SERVER_COMMANDS.md` секция).

## NOTE-083 — Предсказывать не цену, а вероятности (2026-09-15)

**Мысль:** много прогонов целились в «куда пойдёт цена» (регрессия return /
траектория / continuous position). Саму : ход часто меньше спреда+комисбудущую цену угадать нельзя в смысле
полезного point-forecastсии (EXP-081), а
красивый IC без EV после fees — пустышка.

Другой объект: **вероятности дискретных событий**, которые в истории просто
считаются частотой — `P(touch +X% раньше −Y%)`, `P(peak≥5× | early state)`,
`P(up за H | regime)`, калиброванный `p` → ставка только если
`p·win − (1−p)·loss > costs`. Это не «магия вероятностей вместо цены»:
таргет должен быть **событием с чётким определением**, иначе снова
переобучимся на шум. FLIP/`p_up` уже ближе к этой рамке, чем регрессия close;
слабое место было не форма выхода модели, а (а) плохое событие/горизонт,
(б) калибровка и EV после fees, (в) selection bias (как в EXP-082).

**Следствие для следующих опытов:** сначала зафиксировать событие и его
base-rate в данных, потом модель как поправку к base-rate; критерий —
калибровка + net EV, не RMSE цены и не raw return без издержек.

## NOTE-084 — Probability meter (инфра без ML) + smoke BTC 5m (2026-09-15)

**Что сделано:** общая мерилка `causal_flip/ml_dataset/prob_meter.py` —
событие + causal quantile-страты признака + fee → base-rate / P(win) /
mean net EV / reliability. GPU не используется.

**Smoke (локальная панель, сервер не трогали):**
`btc_5m_bars_with_features.parquet` (2022-11 → 2026-09), fee 8 bps RT,
фичи `depth_imb_total` / `flow_imbalance`, горизонты 1/3/6 баров (5–30 мин),
5 causal bins (roll 2000).

**Результат smoke:** стабильно прибыльных страт **0**. Лучшие net EV
≈ **−6.5…−8 bps** — почти ровно минус комиссия (gross ≈ 0 при winrate
~51–52%). То есть частоты считаются, «кармана» над costs на этих
стратах/горизонтах нет — согласуется с EXP-073/081.

**Артефакты:** `causal_flip/reports/prob_meter/{strata_by_fold,strata_summary,reliability,base_rates}.csv`,
`summary.json`. Дальше: новые события/фичи подставлять в тот же скрипт;
ML не начинать, пока какая-то страта не даст устойчивый net EV > 0.

## NOTE-085 — 1h + extreme bins: микро-направление depth/flow закрыто (2026-09-15)

**План по очереди:** (1) мерилка на 1h, (2) событие `|ход|>16/32 bps` +
только extreme bins, (3) вердикт.

**Данные:** собрана панель `btc_1h_bars_with_features.parquet` (33,864 бара,
2022-11→2026-09, as-of bookDepth/metrics). Сервер не трогали.

**(1) 1h, fee 8 bps, depth/flow, H=1/3/6h:** стабильных плюсовых страт **0**.

**(2) Extreme-only (bins 0 & 4), move 16/32 bps:**
- 5m: net ≈ −7.4…−8.5 bps, folds_pos 0/5.
- 1h depth extreme H=6: mean net ≈ **−3.5 bps**, folds_pos **2/5**, max +4.3 /
  min −8.1 — не стабильно.
- 1h flow extreme: net ≈ **−8…−9 bps**, winrate **<50%** (хуже random).

**Вердикт:** направленный edge «depth/flow imbalance → сторона» на taker 8 bps
**закрыт** на 5m и 1h (включая extreme). Gross ≈ 0 или слабый; после fees
минус. GPU/ML на ту же постановку не запускать.

**Следующее (по плану):** не микро-направление, а (а) **regime gate**
(пила/тренд/шок → торговать или нет), (б) **maker / adverse-selection**
вероятности при более низких fees.

## EXP-086 — та же 1h-обучалка на других монетах: кривой разрез, честный FLIP, NEAR и BCH (2026-09-26/27)

**Вопрос:** последняя хорошая обучалка (та, что дала живой DOGE FLIP, сид 72) ещё находит край на другой ликвидной монете, если 2026 не участвует в выборе архитектуры, сида и порогов.

**Что считается последней хорошей обучалкой.** Прогон 2026-02-28 / 2026-03-01: `python -m transformer2602.optuna_1h`, study `prod1h_h6_maxret`, 60 триалов, победитель триал 22, затем `champion_1h` 100 сидов, сид 72 → `prod_doge_1h_v1/model.pt`. `optuna_1h_v2` к этой модели отношения не имеет. Архив рецепта (без v2 и без секретов): `archive_1h_flip_trainer.zip`. Оригинальные `optuna_1h.py` / `champion_1h.py` не переписывались. Живой бот не останавливался.

**Комиссия и исполнение FLIP в этих прогонах.** Fee 4 bps на сторону. Сигнал на закрытии бара t, вход/выход по следующему open, держим пока сторона не сменилась, round-trip = 2×4 bps. Сетка порогов FLIP: 0.45…0.95 шаг 0.05. Фиксированный H6 (`simulate_1h`) — отдельный, более ранний протокол: держит ровно горизонт метки.

**Почему первый NEAR-поиск был недействителен.** В оригинале ранняя остановка — последние 15% баров, а очки триала/сида — календарный 2025 (`TEST_START=2025-01-01`, `TEST_END=2025-12-31`). На DOGE (история с 2018, прогон в феврале 2026) разрез 15% попадал около конца 2024, поэтому 2025 почти не входил в обновление весов. На NEAR история с 2020-10-14, и те же 15% отрезаются около 2025-11-03: январь–начало ноября 2025 одновременно в обучении и в счёте. 2026 при этом сидит только в loss ранней остановки, не в выборе.

**Честный разрез, на котором дальше всё считалось.** Обучение: бары до 2025-02-01, метка горизонта H не заходит в февраль. Валидация: 2025-02-01…2025-12-31 (ранняя остановка, сетка порогов, очки триала и сида). Тест: с 2026-01-01, в выбор не входит. Scaler (mean/std) только по барам до валидации. Свечи целевой пары подставляются в слоты `DOGEUSDT_*`, которые ждёт `build_features_1h`. Funding в этих слотах по-прежнему пустой — как в оригинале.

Колонка `cal` в логе чемпиона — Кальмар на порогах максимальной доходности. Файл `champion_calmar.pt` выбирается по другому порогу (`best_calmar`) и может быть другим сидом.

### 1. NEAR, кривой разрез, фиксированный H6

Study `near1h_h6_maxret`, `transformer2602/runs/optuna_1h_near/`. Триалов 60. Победитель **триал 58, +481177%** на полном 2025. d=256, nhead=4, 6 слоёв, seq=40, dropout 0.1, lr ~2.04e-4, batch 1024, epochs 50, patience 15, H=6. Чемпион на этих параметрах остановлен до сохранения весов; в убитом логе сид 4 печатал порядка +315104%. **Эти числа не OOS.** Study не возобновлять.

### 2. NEAR, честный разрез, фиксированный H6

Study `near1h_h6_val2025`, `runs/optuna_1h_near_wf/`. Победитель **триал 25, +562.8%** на фев–дек 2025. d=64, nhead=8 в записи триала (в модели при d=64 и nhead=8 головы принудительно становятся 4), ff_mult=4, 6 слоёв, seq=80, dropout 0.3, lr ~8.47e-5, batch 1024, epochs 50, patience 15, H=6. Более ранний study `near1h_h6_wf` (валидация захвачена до конца 2026) убит, не использовать.

Чемпион 100 сидов, `runs/champion_1h_near_wf/`, 1 ч 49 мин. Счёт — тот же H6, не FLIP.

| что | сид | окно | ret | MDD | пороги |
|---|---:|---|---:|---:|---|
| return-чемпион | 57 | фев–дек 2025 | +232.8% | −37.4% | 0.60 / 0.50 |
| calmar-файл | 52 | фев–дек 2025 | +17.1% на паре Кальмара | −1.98% | 0.70 / 0.85, best_calmar 8.68 |
| сид 0 в логе | 0 | фев–дек 2025 | +224.7% | −28.0% | 0.55 / 0.55, cal=+8.05 |

У сида 52 в строке лога `cal=+1.14`: это Кальмар при порогах максимальной доходности, не `best_calmar`. У сида 0 напечатанный Кальмар совпадает с `best_calmar`, поэтому по колонке лога он впереди, а файл `champion_calmar.pt` — сид 52.

Сид 0, пороги заморожены с валидации, H6 на 2026: **−71.0%**, MDD **−73.2%**, 422 сделки. Помесячно 2026: янв −7.7, фев +5.8, мар −19.2, апр −11.4, май +31.4, июн −44.3, июл −7.1, авг −5.4, сен −35.5.

Тот же сид 0, FLIP, пороги подобраны только на фев–дек 2025: выбранная пара **0.51 / 0.45**, валидация **+1470.6%**, MDD −51.1%, 115 сделок. Те же пороги на 2026: **−78.5%**, MDD **−81.4%**, 115 сделок. Верх сетки на 2026 красный. Пара 0.51 / 0.55: валидация +1175%, 2026 −56.4% / MDD −73.2%.

Отдельный сид 4 (не кривой чемпион): обучение до 2024-12-31, валидация весь 2025, пороги с 2025, затем 2026. Валидация +35.4%, MDD −22.6%, пороги 0.85 / 0.65. Тест 2026: **−29.7%**, MDD **−40.0%**, сентябрь −24.7%.

### 3. NEAR, честный разрез, FLIP внутри Optuna

Горизонт метки ищется от 3 до 6. Батч 64 / 128 / 256 / 512. Очки — максимальный FLIP-return на валидации, лучший из 5 сидов. Убитые черновики, не возобновлять: `near1h_flip_h3to6` (батч зафиксирован 512), `near1h_flip_h3to6_b512` (только 256/512).

Рабочий study `near1h_flip_h3to6_b64`, `runs/optuna_1h_near_flip/`. Победитель **триал 56, +798.4%** на фев–дек 2025. d=256, nhead=8, 4 слоя, ff_mult=2, seq=168, dropout 0.05, lr ~1.91e-5, weight decay ~3.86e-5, batch 128, epochs 50, patience 12, **H=4**.

Чемпион 100 сидов, `runs/champion_1h_near_flip/`, 3 ч 53 мин. И доходность, и Кальмар — **сид 22**: валидация **+965.9%**, MDD **−36.3%**, Кальмар 26.6, пороги **0.45 / 0.45**, 78 сделок. Оба файла `champion_return.pt` и `champion_calmar.pt` — этот сид.

Те же замороженные пороги:

| окно | ret | MDD | сделки |
|---|---:|---:|---:|
| фев–дек 2025 (окно выбора) | +965.9% | −36.3% | 78 |
| 2026-01-01…2026-09-26 | **−93.0%** | **−96.5%** | 39 |

2026 помесячно: янв −23.8, фев −2.8, мар +2.7, апр +9.1, май −55.7, июн −23.2, июл +9.9, авг −16.8, сен −72.8.

### 4. Bitcoin Cash, тот же FLIP-протокол

Пара BCHUSDT, история 1h с 2019-11-28. Study `bch1h_flip_h3to6_b64`, `runs/optuna_1h_bch_flip/`, 12 ч 44 мин. Победитель **триал 52, +694.4%** на фев–дек 2025. d=256, nhead=4, 6 слоёв, ff_mult=2, seq=120, dropout 0.2, lr ~1.30e-5, batch 128, epochs 80, patience 12, **H=3**. Рядом триалы 53 / 36 / 33 / 42: +647%, +571%, +554%, +553%, та же форма сети (d=256, 4 головы, 6 слоёв, seq=120, H=3).

Чемпион 100 сидов запущен и **остановлен по просьбе** около сида 73. На диск из этого прогона чемпионские `.pt` не записаны: скрипт пишет только лидеров в конце. Отдельно переобучен **сид 1** (он совпал со счётом оптуны). Валидация снова +694.4%, MDD −31.1%, пороги **0.45 / 0.45**. Веса: `runs/champion_1h_bch_flip/seed1.pt`.

2026, пороги не перебирались, данные по 2026-09-27: **+26.4%**, MDD **−66.8%**, 81 сделка. Помесячно: янв −2.4, фев +46.0, мар −4.0, апр −5.5, май −35.4, июн −30.3, июл +10.8, авг +13.8, сен +72.4. Плюс на конце года есть, просадка для рабочей модели слишком глубокая — поэтому чемпион дальше не считался.

### 5. Что из этого следует

На честном разрезе валидация 2025 у NEAR и BCH легко даёт сотни процентов: сид и пороги выбираются по этому же окну. На нетронутом 2026 ни один дожатый NEAR не удержался (H6 сид 0 −71%, FLIP сид 0 −78%, FLIP-чемпион сид 22 −93%, ранний сид 4 −30%). BCH сид 1 на 2026 чуть в плюсе и всё равно с просадкой −67%. Край живого DOGE-чекапоинта этим рецептом на новую монету не перенёсся.

XMRUSDT на Binance spot в статусе BREAK, Монеро этой загрузкой не скачать. Второй GPU-процесс поверх уже идущего обучения не ставить.

### 6. Quant FLIP

2026-09-27, study `qnt1h_flip_h3to6_b64`, 7 ч 54 мин. Победитель **триал 3, +744.8%** на фев–дек 2025. d=64, в записи nhead=8 (в модели при d=64 головы становятся 4), 5 слоёв, seq=120, dropout 0.35, batch 64, epochs 50, patience 8, **H=6**.

Чемпион остановлен около сида 58. Отдельно переобучены сиды 2 и 22, пороги с валидации **0.50 / 0.45**, на 2026 не двигались. Валидация совпала с логом: сид 2 **+744.8%** / MDD −50.0%, сид 22 **+566.9%** / MDD −26.0%. Оба на 2026: **−100%**, MDD **−100%**. Сентябрь обнулил счёт: Quant с ~61 вырос до 195, шорт забрал больше 100%. Веса `runs/champion_1h_qnt_flip/seed2.pt`, `seed22.pt`.

**Артефакты:** `transformer2602/optuna_1h_near.py`, `optuna_1h_bch.py`, `optuna_1h_qnt.py`, `champion_1h_near.py`, `champion_1h_bch.py`, `champion_flip_near_years.py`, `bch_seed1_2026.py`, `seed0_near_2026.py`, `seed4_oos_2026.py`, `seed4_flip_grid.py`; каталоги `runs/optuna_1h_near/`, `optuna_1h_near_wf/`, `optuna_1h_near_flip/`, `champion_1h_near_wf/`, `champion_1h_near_flip/`, `optuna_1h_bch_flip/`, `champion_1h_bch_flip/`, `optuna_1h_qnt_flip/`; архив `archive_1h_flip_trainer.zip`.

## EXP-087 — таргет вместо FLIP в обучении: QNT и BCH (2026-09-28)

**Решение:** и оптуна, и чемпион снова считают фиксированное удержание на горизонт метки (`simulate_1h`), пороги 0.50…0.95 шаг 0.05. Горизонт по-прежнему ищется от 3 до 6, батч до 512. Разрез тот же: обучение до 2025-02-01, валидация фев–дек 2025, 2026 в выбор не входит. FLIP, если применяется, только после обучения: пороги FLIP с 0.45…0.95 шаг 0.05 подбираются по валидации, потом замораживаются.

Строка лога чемпиона — пороги максимальной доходности. Колонка `cal` — Кальмар на этой паре. Файл `champion_calmar.pt` — другая пара, максимум Кальмара.

### 1. QNT, таргет

Study `qnt1h_target_h3to6`, `runs/optuna_1h_qnt_target/`, 6 ч 45 мин. Победитель снова **триал 3, +433.0%** (против +744.8% у FLIP-счёта той же формы). d=64, nhead в файле 8 / в сети 4, 5 слоёв, seq=120, dropout 0.35, batch 64, H=6.

Чемпион на этих параметрах остановлен. Сид 2 совпал с логом: валидация **+432.1%**, MDD −43.2%, пороги **0.50 / 0.50**.

| режим | окно | пороги | ret | MDD | сделки |
|---|---|---|---:|---:|---:|
| таргет | 2026 | 0.50 / 0.50 | −41.7% | −73.5% | 811 |
| FLIP | фев–дек 2025 | 0.50 / 0.45 | +768.1% | −50.0% | — |
| FLIP | 2026 | те же 0.50 / 0.45 | −100% | −100% | 59 |

2026 таргет помесячно: янв +7.5, фев −3.9, мар −8.9, апр −4.0, май +1.0, июн −13.5, июл −13.4, авг +11.5, сен −23.5. FLIP на 2026 снова умер в сентябре на том же росте цены. Веса `runs/champion_1h_qnt_target/seed2.pt`.

### 2. BCH, таргет, батч 1024 не лучше

Study `bch1h_target_h3to6` остановлен на 58 завершённых триалах. Лучший на этот момент **триал 44, +183.9%**. d=128, nhead=4, 2 слоя, ff_mult=2, seq=80, dropout 0.15, lr 1.266e-05, weight decay 5.15e-06, batch 512, epochs 50, patience 8, H=6. Триалы 43 и 44 почти соседи: батч у обоих 512, отличаются только lr и weight decay (1.283e-05 / 5.27e-06 против 1.266e-05 / 5.15e-06), счета +173.5% и +183.9%.

Та же сеть триала 44, батч **1024**, пять сидов: +28.9, +146.1, +40.7, +126.9, +116.3. Максимум **+146.1%**, среднее **+91.8%**. У батча 512 максимум из пяти сидов **+183.9%**, среднее **+92.6%**. Среднее то же, лучший сид на 1024 слабее. В поиск батч 1024 не возвращали.

### 3. BCH чемпион, сид 4

100 сидов, 2 ч 40 мин, `runs/champion_1h_bch_target/`. По доходности **сид 4**: валидация **+183.9%**, MDD −18.2%, Кальмар на этой паре +10.15, пороги **0.55 / 0.70**. Файл `champion_return.pt`.

По Кальмару **сид 77**, файл `champion_calmar.pt`. В логе у него +62.6% / MDD −17.2% / пороги 0.60 / 0.60 — это пара доходности. Пара Кальмара другая: **0.70 / 0.85**, доходность **+40.2%**, MDD **−2.1%**, Кальмар **19.5**. 2026 для сида 77 не смотрели.

Сид 4, пороги не с 2026:

| режим | окно | пороги | ret | MDD | сделки |
|---|---|---|---:|---:|---:|
| таргет | 2026 | 0.55 / 0.70 | +13.5% | −20.1% | 226 |
| FLIP | фев–дек 2025 | 0.60 / 0.65 | +88.5% | −29.6% | 82 |
| FLIP | 2026 | те же 0.60 / 0.65 | +44.8% | −18.7% | 51 |

2026 таргет помесячно: янв +7.9, фев +3.2, мар −7.7, апр +5.1, май −0.1, июн −8.1, июл +7.1, авг −4.9, сен +12.3.

2026 FLIP помесячно: янв +13.5, фев +8.3, мар +8.0, апр +2.9, май +2.1, июн −5.9, июл −5.1, авг +0.1, сен +16.2.

На этом сиде FLIP на валидации слабее таргета (+88% против +184%), а нетронутый 2026 проходит лучше и с меньшей просадкой (+44.8% / −18.7% против +13.5% / −20.1%). Это первый честный прогон в серии, где 2026 не красный и просадка около −20%. Веса `runs/champion_1h_bch_target/seed4.pt`.

**Артефакты:** `transformer2602/optuna_1h_qnt.py`, `optuna_1h_bch_target.py`, `champion_1h_qnt.py`, `champion_1h_bch_target.py`, `qnt_seeds_2026.py`, `qnt_seed2_2026.py`, `bch_batch1024.py`, `bch_seed4_2026.py`; каталоги `runs/optuna_1h_qnt_target/`, `champion_1h_qnt_target/`, `optuna_1h_bch_target/`, `champion_1h_bch_target/`.

## EXP-088 — состояния прямых метрик → EV (2026-09-28)

**Статус:** NO-GO. Ни одно состояние не прошло заранее зафиксированные ворота.

**Метод.** Не прогноз цены и не ML. Состояние — не больше двух прямых метрик, шкала causal: квантиль предыдущих 720 часовых баров, текущий бар не входит. Сторона против толпы. Вход на open следующей свечи, выход ровно через H ∈ {6, 24, 72} часов. Одно срабатывание на восходящий фронт, удержания не пересекаются. EV_net = средний gross − 8 bps round-trip. Фолды EXP-045: 2024H2, 2025H1, 2025H2, 2026H1, 2026H2_YTD. Train до 2024-07-01 в выбор не входит.

Символ проходит горизонт, если n ≥ 30, EV_net > 0 на стыкованном OOS, плюс хотя бы на 3 из 5 фолдов, и ни один фолд не хуже −15% по сложной доходности. Состояние — GO, только если так проходят минимум 2 инструмента. Пороги после просмотра не двигались.

**Чего нет в кэше.** Open interest и long/short ratio — S3, S4, S5 не считались. Фандинг BTC и ETH отсутствует, поэтому S1/S2 только DOGE. Волатильность 168h считается из доходностей, но у неё нет стороны толпы, в состояние не входила. Кэш DOGE 1h обрывается **2026-08-09 17:00** (BTC и ETH доходят до 2026-09-26). Фандинг DOGE: 2020-07-10 … 2026-08-09.

**Зафиксированные состояния.** S1 funding > q99 → шорт. S2 funding < q01 → лонг. S6 Fear & Greed < 10, значение видно со следующего дня → лонг. R1 price/BTC > q99 → шорт (DOGE, ETH). R2 price/BTC < q01 → лонг. V1 объём > 2× медианы 30 дней и доходность 24h > 0 → шорт.

### OOS, EV_net

| состояние | H | DOGE | BTC | ETH | вердикт |
|---|---:|---|---|---|---|
| S1 funding hi | 6 | n=7, −0.70%, 1/5, худший −6.3% | нет фандинга | нет фандинга | NO-GO |
| S1 | 24 | n=7, −3.30%, 1/5, −26.7% | — | — | NO-GO |
| S1 | 72 | n=6, −15.1%, 0/5, −71.7% | — | — | NO-GO |
| S2 funding lo | 6 | n=29, +0.07%, 1/5, −5.0% | — | — | NO-GO |
| S2 | 24 | n=28, +1.00%, 3/5, −5.2% | — | — | NO-GO |
| S2 | 72 | n=25, +1.47%, 3/5, −12.8% | — | — | NO-GO |
| S6 F&G < 10 | 6 | n=11, −0.08%, 0/5 | n=11, +0.04%, 1/5 | n=11, −0.01%, 0/5 | NO-GO |
| S6 | 24 | n=11, +0.67%, 1/5 | n=11, +0.22%, 1/5 | n=11, +0.22%, 1/5 | NO-GO |
| S6 | 72 | n=8, +0.62%, 1/5 | n=8, +0.50%, 1/5 | n=8, +0.84%, 1/5 | NO-GO |
| R1 ratio hi | 6 | n=69, −0.65%, 0/5, −21.9% | не определено | n=68, −0.58%, 1/5, −18.2% | NO-GO |
| R1 | 24 | n=40, −1.80%, 1/5, −55.5% | — | n=44, −1.22%, 1/5, −25.4% | NO-GO |
| R1 | 72 | n=26, −4.16%, 2/5, −96.8% | — | n=27, −1.43%, 1/5, −28.0% | NO-GO |
| R2 ratio lo | 6 | n=180, −0.52%, 0/5, −45.7% | — | n=215, **+0.16%, 3/5, −3.9%** | NO-GO |
| R2 | 24 | n=107, −0.13%, 3/5, −28.2% | — | n=121, −0.13%, 2/5, −33.1% | NO-GO |
| R2 | 72 | n=57, +0.02%, 2/5, −37.0% | — | n=66, −0.08%, 2/5, −21.5% | NO-GO |
| V1 volume up | 6 | n=582, −0.12%, 1/5, −36.9% | n=645, −0.08%, 2/5, −21.9% | n=681, −0.09%, 1/5, −31.0% | NO-GO |
| V1 | 24 | n=312, −0.40%, 1/5, −59.0% | n=360, −0.11%, 3/5, −31.3% | n=366, −0.10%, 2/5, −27.7% | NO-GO |
| V1 | 72 | n=165, −0.36%, 3/5, −67.0% | n=190, −0.15%, 2/5, −45.7% | n=192, −0.29%, 2/5, −48.8% | NO-GO |

Единственная ячейка, которая сама по себе проходит все ворота, — ETH, R2, 6h. На DOGE то же правило в минусе и с фолдом −46%. Это не подтверждение, а расхождение инструментов.

S2 на 24h и 72h имеет плюсовой EV и 3 зелёных фолда, но событий 28 и 25, и проверить BTC/ETH нельзя: фандинга в кэше нет. Ворота n ≥ 30 и «2 из 3 инструментов» не пройдены. Порог не снижался.

V1 на всех трёх инструментах: gross около нуля, net отрицательный, худший фолд от −22% до −67%. Та же картина, что у depth/flow в NOTE-085: частоты есть, кармана над 8 bps нет.

**Вывод.** Ветка прямых метрик, которые реально лежат в кэше, — NO-GO. OI и L/S этой проверкой не закрыты: их истории в кэше нет.

**Артефакты:** `exp088_ev_states.py`, `tests/test_exp088_ev_states.py`, `causal_flip/reports/exp088_ev_states/summary.json`, `state_ev.csv`, `fold_results.csv`.

## EXP-089 — обнуление фич живого FLIP-бота (2026-09-29)

**Статус:** диагностический. Совместное выключение «плохих» фич 2026 не улучшило.

**Метод.** Чекпоинт `prod_doge_1h_v1`, пороги 0.48 / 0.52, динамическая температура, сборка фич бота с кэша `2024-01-01` (129 колонок). Вход на open следующей часовой свечи. Комиссия 4 б.п. и проскальзывание 2 б.п. на сторону, доля позиции 0.931. 2025 и 2026 — отдельные счета со старта 10 000. Данные до 2026-09-22 12:00. Обнуление — поставить фичу в 0 после скейла, то есть в среднее обучения. Это не минутный causal-replay из `analysis_flip_causal_2025_2026/summary.json`.

**По одной фиче.** База: 2025 **+4490.3%** (250 сделок), 2026 **+38.2%** (172 сделки). Плюс в колонке дельты значит, что без этой фичи год лучше. Минус — фича нужна.

Убирание бьёт по обоим годам только у 8 фич: `macro_doge_sp500_corr_20d` (−2722 / −23 п.п.), `macro_nasdaq_ret_1d` (−26 / −23), `eth_stoch_k` (−453 / −18), `corr_doge_btc_24` (−734 / −18), `doge_adx` (−704 / −14), `ctx4h_rsi_14` (−1326 / −6), `btc_ema50_ratio` (−932 / −5), `doge_ema50_ratio` (−1899 / −2). На 2026 сильнее всего проседает `btc_di_plus`: год из +38% в −14% (−52 п.п.), при этом 2025 чуть растёт.

64 фичи по отдельности поднимают оба года. Крупнейшие плюсы 2026 при выключении: `btc_obv_z168` +133 п.п., `doge_macd_line` +131, `btc_d_ema50_ratio` +123. `funding_rate` и `funding_ma8` в боевом наборе уже нули, дельта 0.

На 2025 сложный процент раздувает ранний сдвиг сделки до тысяч процентов. 2026 на свежем счёте читается прямее.

**Все 64 вместе.** Правило отбора: дельта 2025 > 0 и дельта 2026 > 0 в поштучном тесте. Смешанные фичи не трогались. Те же годы и разметили фичи, и приняли этот прогон — это не чистый тест.

| | все 129 | без 64 |
|---|---:|---:|
| 2025 | +4490%, 250 сделок | **+902.0%**, MDD −39.7%, 170 сделок, винрейт 60%, эквити 100 204 |
| 2026 | +38.2%, 172 сделки | **−43.5%**, MDD −66.6%, 102 сделки, винрейт 45%, эквити 5 651 |

2025 помесячно без 64: янв +52.4, фев +53.1, мар −12.1, апр +53.5, май +18.1, июн +15.1, июл +21.2, авг +21.0, сен +4.8, окт +34.8, ноя +18.2, дек −4.4.

2026 помесячно: янв −33.9, фев −27.8, мар +0.9, апр −14.5, май −1.0, июн −0.2, июл −15.5, авг +41.7, сен +15.8.

**Вывод.** Поштучный плюс не складывается. Фичи, которые по одной выглядели лишними, вместе держат 2026: выключение уводит год из +38% в −44%.

**Артефакты:** `causal_flip/flip_bot_feature_ablation.py`, `causal_flip/flip_bot_zero_bad_features.py`; `analysis_flip_causal_2025_2026/feature_ablation.csv`, `feature_ablation_ranked.csv`, `zero_bad_features.json`.

## EXP-100 — 20 фич класса B, цель H6, флип после чемпиона (2026-09-29)

**Вопрос.** Хватает ли 20 фич, без которых серверный реплей терял больше всего, чтобы заново найти край: обучение до 2025, калибровка на 2025, проверка на 2026.

**Зафиксировано до прогона.** DOGEUSDT 1h, бары с 2019-07-05. Веса и скейл только по барам до 2025-01-01. Ранняя остановка — последние 15% этого окна, метка горизонта не заходит в 2025. Цель Optuna — доходность фиксированных 6 часов на календарном 2025, 60 триалов, 5 сидов, пространство как в `optuna_1h` (при d_model 64 и nhead 8 головы становятся 4). Чемпион: 100 сидов тех же гиперпараметров. Сетка флипа 0.45–0.95 шаг 0.05 выбирается по доходности 2025. Сид чемпиона — максимальный кальмар 2025 среди сидов с просадкой не глубже −30%. Пороги на 2026 заморожены. 2026 не входит в обучение, скейл, Optuna и выбор сида.

**20 фич.** `eth_range_pct`, `eth_macd_line`, `macro_doge_sp500_corr_20d`, `ctx4h_ema50_r`, `doge_ema50_ratio`, `btc_d_trend_ma20`, `eth_macd_signal`, `month_sin`, `eth_ema100_ratio`, `eth_rsi_14`, `macro_nasdaq_ret_1d`, `btc_di_plus`, `fear_greed_z30`, `doge_di_plus`, `eth_stoch_k`, `doge_d_ema200_ratio`, `eth_d_trend_ma20`, `ctx4h_rsi_14`, `doge_adx`, `btc_atr_pct_14`.

**GO, если всё сразу.** Медиана доходности 2026 по 100 сидам > 0. Доля сидов с 2026 > 0 не меньше 60%. Медиана просадки 2026 не глубже −35%. Медиана 2026 не хуже такого же прогона на всех 129 фичах. База 129 — те же правила разреза и флипа, гиперпараметры живой обучалки (`runs/optuna_1h/best_params.json`), без нового поиска.

**Статус.** Прогон запущен. Цифры дописываются после него.

**Артефакты:** `transformer2602/optuna_1h_20feat.py`, `transformer2602/champion_1h_20feat.py`, `analysis_exp100_20feat/`.

## EXP-101 — жадное зануление фич живого FLIP на серверном реплее (2026-09-29)

**Вопрос.** Можно ли по одной выключить фичи, которые по отдельности портят оба года, и оставить только те нули, которые не ломают ни 2025, ни 2026.

**Метод.** Тот же счёт, что в серверном реплее до 2026-09-29: кэш бота, пороги 0.48/0.52, динамическая температура, минутные сделки DOGEUSDC, фандинг, комиссия и проскальзывание. Ноль — колонка после скейла равна 0, то есть среднее обучения. Кандидаты — 52 фичи, у которых поштучное выключение поднимало и 2025, и 2026; порядок по силе плюса 2026. Ноль остаётся, только если оба года не хуже текущего счёта и хотя бы один лучше. Иначе колонка возвращается до следующей фичи. В прод не переносилось.

**База.** 2025 **+4536.7%**, просадка **−56.5%**, эквити 463 675. 2026 **+81.4%**, просадка **−42.3%**, эквити 840 926.

**Оставлены три нуля:** `doge_di_minus`, затем `btc_ma_ratio_24`, затем `eth_obv_chg_z24`. Остальные 49 на уже зануленном наборе хотя бы один год ухудшали и были возвращены.

| | база | три нуля |
|---|---:|---:|
| 2025 | +4536.7%, MDD −56.5% | **+7290.9%**, MDD −29.5%, эквити 739 090 |
| 2026 | +81.4%, MDD −42.3% | **+218.8%**, MDD −36.3%, эквити 2 356 549 |

Помесячно, доходность и просадка месяца (от пика внутри месяца, включая вход):

| месяц | база % | 3 нуля % | база MDD | 3 нуля MDD |
|---|---:|---:|---:|---:|
| 2025-01 | +195.7 | +189.1 | −16.9 | −11.4 |
| 2025-02 | +110.5 | +74.6 | −26.0 | −26.0 |
| 2025-03 | +29.2 | +20.9 | −24.2 | −28.9 |
| 2025-04 | +28.9 | +36.8 | −22.7 | −18.6 |
| 2025-05 | +52.7 | +46.3 | −15.7 | −19.3 |
| 2025-06 | −7.0 | +7.4 | −22.8 | −23.3 |
| 2025-07 | +111.0 | +134.3 | −11.0 | −11.0 |
| 2025-08 | −2.7 | +9.6 | −30.5 | −26.7 |
| 2025-09 | +20.3 | +30.8 | −17.1 | −18.1 |
| 2025-10 | +2.5 | +34.0 | −56.5 | −29.0 |
| 2025-11 | +8.0 | +10.5 | −19.9 | −20.8 |
| 2025-12 | +15.1 | +13.3 | −22.3 | −22.0 |
| 2026-01 | +7.7 | +19.6 | −18.4 | −18.4 |
| 2026-02 | −15.5 | −1.2 | −35.3 | −34.0 |
| 2026-03 | +0.2 | +12.9 | −16.2 | −16.0 |
| 2026-04 | −16.6 | −7.3 | −23.4 | −19.5 |
| 2026-05 | +12.3 | +7.0 | −9.3 | −13.9 |
| 2026-06 | +28.7 | +29.5 | −9.1 | −11.7 |
| 2026-07 | −2.4 | +6.5 | −10.5 | −8.6 |
| 2026-08 | +26.8 | +25.1 | −14.9 | −14.9 |
| 2026-09 | +33.2 | +39.8 | −15.8 | −14.3 |

**Вывод.** На этом реплее три нуля поднимают оба года и срезают октябрьскую яму 2025 с −56.5% до −29.0%. В прод не писать по этому прогону: 2026 участвовал в решении «оставить». Другой порядок тех же 52 мог оставить другой набор. Решение по живому боту отложено.

**Артефакты:** `causal_flip/flip_server_greedy_zero.py`, `causal_flip/flip_server_greedy_months.py`; `analysis_flip_causal_2025_2026/server_greedy_zero.json`, `server_greedy_zero_months.json`.

## EXP-102 — XRP, цель H6, перенос во флип и жадные нули (2026-10-01)

**Вопрос.** Держит ли чемпион XRP, обученный как фиксированное удержание, исполнение флипом, и какие фичи можно выключить, не ухудшив ни 2025, ни 2026.

**Обучение.** XRPUSDT 1h. К свечам Рипла добавлены BTC и ETH. 138 фич: прежние 129 плюс девять (`doge_cvd_z24`, `doge_cvd_slope12`, `doge_vwap_dev24`, `doge_vol_regime`, `doge_candle_eff`, `doge_price_accel`, `doge_ret_ac12`, `doge_ret_ac24`, `doge_btc_ratio_dev`). Колонки `doge_` — это Рипл, скопированный в слоты DOGE, не Dogecoin. Обучение до 2025-02-01, метка горизонта не заходит в февраль. Optuna: 60 триалов, 5 сидов, счёт — максимальная доходность фиксированного удержания на февраль–декабрь 2025. Сетка порогов цели 0.50–0.95 шаг 0.05. 2026 в массив поиска не входил. Батч в поиске: 64, 128, 256, 512, 1024.

Лучший триал 52, счёт **+300.1%**, горизонт 6. Веса: d_model 64, 5 слоёв, ff 4, seq 168, dropout 0.2, lr 5.97e-5, batch 512. В файле параметров nhead записан как 8; при d_model 64 обучение шло с 4 головами.

**Чемпион.** 100 сидов тех же гиперпараметров. По доходности валидации победил сид 1: фиксированное удержание **+300.1%**, просадка −32.7%, пороги цели 0.50/0.50. Файл `transformer2602/runs/champion_1h_xrp_target/champion_return.pt`.

**Флип без нулей.** Те же веса. Сетка 0.45–0.65 шаг 0.01, пара выбирается по доходности 2025 при числе сделок не меньше 5. 2026 на этом шаге только смотрели. Каждый год стартует с 10 000. Комиссия флипа — две стороны по 4 б.п.

Победитель **0.50 / 0.45**. 2025 **+733.3%**, просадка −40.5%, 114 сделок. 2026 **−49.2%**, просадка −52.9%, 40 сделок. Октябрь 2026 неполный: данные до 2026-10-01.

2025 по месяцам: −3.2, +47.6, +136.8, −1.1, +27.7, −0.6, +60.1, +44.7, −5.6, +2.0, −7.4, −5.0.

2026 по месяцам: −25.1, +2.5, −6.7, −1.8, −12.8, +11.0, −1.9, −11.3, −15.0, +0.7.

**Нули.** Пороги флипа заморожены. Правило то же, что в EXP-101. Кандидат — фича, чьё одиночное обнуление поднимает и 2025, и 2026. Порядок по плюсу 2026. Ноль остаётся, только если оба года не хуже текущего счёта и хотя бы один лучше. Иначе колонка возвращается. Ноль — значение после скейла равно 0, то есть среднее обучения. Таких кандидатов 47 из 138.

Оставлены четыре, в этом порядке: `doge_d_vol_z` (дневной объём Рипла), `doge_d_trend_ma20` (дневной тренд Рипла), `macro_sp500_vol_10d`, `doge_ma_ratio_24` (цена Рипла к MA24). Остальные 43 на уже зануленном наборе портили хотя бы один год.

| | флип | четыре нуля |
|---|---:|---:|
| 2025 | +733.3%, MDD −40.5%, 114 сделок | **+1425.0%**, MDD −31.4%, 108 сделок |
| 2026 | −49.2%, MDD −52.9%, 40 сделок | **+122.1%**, MDD −45.3%, 44 сделки |

Путь: после первого нуля 2025 +945.9% / 2026 +14.7%; после второго +1168.3% / +55.2%; после третьего +1261.0% / +82.0%; после четвёртого +1425.0% / +122.1%.

**Вывод.** На этом счёте четыре нуля поднимают оба года и смягчают просадку. В прод не писать: 2026 участвовал в решении, какие нули оставить. Пороги флипа после нулей не перебирались. Живой DOGE-бот не менялся.

**Артефакты:** `transformer2602/optuna_1h_alt_target.py`, `transformer2602/champion_1h_alt_target.py`, `transformer2602/xrp_flip_test.py`, `transformer2602/xrp_greedy_both_years.py`; `analysis_exp100_20feat/xrp_flip_test.json`, `xrp_greedy_both_years.json`.

## EXP-103 — XRP, выкинуть 47 фич и учить чемпиона заново (2026-10-01)

**Вопрос.** Если 47 фич по одной поднимали и 2025, и 2026 на флипе замороженных весов, станет ли чемпион сильнее, когда эти колонки убрать до обучения.

**Метод.** Те же гиперпараметры триала 52, новая Optuna не запускалась. Список — лог жадного прохода EXP-102, все 47 кандидатов, не только четыре оставленных нуля. Осталось 91 фича. Скейл заново на барах до 2025-02-01. Сид по доходности фиксированного удержания, февраль–декабрь 2025. 2026 в обучение и в выбор сида не входил, но входил в отбор этих 47. Старые веса `champion_1h_xrp_target` не перезаписывались.

**Прогон.** Перезагрузка оборвала обучение после сида 9. Сид 10 в таблицу не попал. Веса чемпиона не сохранились: они пишутся в конце сотни. Доучивание не запускалось.

Лучший из десяти — сид 0: **+115.3%**, просадка −35.2%. Чемпион на 138 фичах на той же проверке: **+300.1%**, просадка −32.7%. Медиана этих десяти около +36%. Медиана первых десяти старого прогона около +23%. Обычные сиды на месте, чемпионский хвост в этих десяти не появился.

**Вывод.** Выкидывать все 47 и учить заново не стоит. Поштучный плюс на замороженном флипе не говорит, какие колонки удалять до нового обучения. Рабочий флип остаётся на старых весах с четырьмя нулями из EXP-102.

**Артефакты:** `transformer2602/runs/champion_1h_xrp_target_drop47/leaderboard.json`, `dropped_features.json`.

## EXP-104 — XRP, три модели: бык, медведь, флет (2026-10-02)

**Вопрос.** Станет ли фиксированное удержание лучше, если учить три сети на барах своего режима и на каждом часе брать модель этого режима.

**Режим.** Дневной ADX Рипла с прошлого закрытого дня, сдвиг как у дневных фич, без заглядывания вперёд. Флет, если ADX ниже 20. Бык, если ADX от 20 и +DI выше −DI. Медведь, если ADX от 20 и −DI выше +DI. Равенство DI и прогрев в обучение не входят: 640 часовых баров обучения без режима. Порог 20 заморожен, по 2025 и 2026 не подбирался. Новой Optuna нет: гиперпараметры триала 52, все 138 фич, горизонт 6, 100 сидов на режим. Сид и пороги — по своему режиму, февраль–декабрь 2025, фиксированное удержание. 2026 в выбор сида и порогов не входил.

| режим | обучение | проверка 2025 | 2026 |
|---|---:|---:|---:|
| бык | 18 759 | 1 152 | 1 817 |
| медведь | 22 306 | 3 090 | 2 112 |
| флет | 17 353 | 3 768 | 2 640 |

**Чемпионы режимов.** Отдельный счёт, вход только на барах своего режима. Пороги быка и медведя 0.55/0.50, флета 0.50/0.50.

| режим | сид | 2025 | 2026 |
|---|---:|---|---|
| бык | 65 | +86.6%, MDD −12.2%, 137 сделок | −10.1%, MDD −25.1%, 204 сделки |
| медведь | 20 | +165.7%, MDD −18.3%, 332 сделки | −33.0%, MDD −36.1%, 218 сделок |
| флет | 78 | +78.9%, MDD −20.2%, 471 сделка | −6.0%, MDD −20.8%, 330 сделок |

**Общий счёт.** Один капитал, каждый час берёт модель своего режима. На момент прогона, данные до 2026-10-01: 2025 **+770.6%**, просадка −25.4%, 938 сделок. 2026 **−40.6%**, просадка −49.9%, 751 сделка. Прогон 6 ч 34 мин.

**Одна модель на тех же окнах.** Пороги 0.50/0.50, те же даты, свечи до 2026-10-02. 2025 **+300.1%**, просадка −32.7%, 1000 сделок. 2026 **−43.2%**, просадка −48.3%, 823 сделки. Это цель H6, не флип EXP-102.

**Ноль.** Правило EXP-101 на общем счёте: колонка гасится у всех трёх моделей сразу. Пороги заморожены. К утру 2026-10-02 база 2025 та же +770.6%, 2026 сдвинулся до −41.0% и 753 сделок из-за новых часов. Кандидат был один, `btc_vol_z_24` (объём биткоина за 24 часа). После него 2025 **+857.6%**, просадка −20.5%, 937 сделок. 2026 **−27.8%**, просадка −47.5%, 752 сделки. 2026 участвовал в решении оставить этот ноль.

**Вывод.** На фиксированном удержании разбиение лучше одной модели: 2025 с +300% до +771%, просадка с −32.7% до −25.4%. Само разбиение 2026 почти не двигает (−43% против −41%). Ноль объёма биткоина доводит 2026 до −28%, год остаётся в минусе. Флет не сидит в кэше. В прод не писать. Живой DOGE-бот не менялся.

**Артефакты:** `transformer2602/champion_1h_xrp_regimes.py`, `transformer2602/xrp_regime_greedy_both_years.py`; `transformer2602/runs/champion_1h_xrp_regimes/summary.json`; `analysis_exp100_20feat/xrp_regime_greedy_both_years.json`.

## EXP-105 — forward pilot трёх нулей (2026-09-30 → ...)

**Дата деплоя:** 2026-09-30 02:21, время сервера. Сервис `bot_binance_flip` перезапущен, в строке старта `zero=doge_di_minus,btc_ma_ratio_24,eth_obv_chg_z24`.

**Baseline:** `prod_doge_1h_v1`, пороги 0.48/0.52. Веса не менялись. Ноль ставится после скейла.

**Изменение:** три нуля — `doge_di_minus`, `btc_ma_ratio_24`, `eth_obv_chg_z24`.

**Ожидание (из бэктеста):** 2026 +219%, MDD −36.3%. Это EXP-101: после трёх нулей 2026 +218.8%, просадка −36.3%. 2026 там участвовал в решении, какие нули оставить.

**Реальность:** forward P&L пилота ещё копится. С 30 сентября смены предикта не было. Локальный пересчёт живого пути (окно включает закрытый бар, динамическая температура, пороги 0.48/0.52) с 2026-09-29 21:00 по 2026-10-02 09:00, время UTC+3: все часы LONG. Ближе всего 2026-10-01 09:00: вверх 0.509, вниз 0.491. Короткий порог 0.52 не достигался, длинный не проваливался через 0.48. Без трёх нулей тоже все часы LONG, ближе всего 2026-10-01 14:00: вверх 0.508, вниз 0.492. Залипший лонг — это модель, не сбой прода.

**Критерий:** через 4–6 недель, окно 2026-10-28 … 2026-11-11, сравнить forward P&L с тем, что дал бы baseline без этих нулей. Если forward лучше baseline — подтверждено. Если хуже — откат.

## EXP-106 — прод-модель DOGE на топ-28 монетах (2026-10-02)

**Вопрос.** Держит ли замороженная прод-модель собаки плюс на других монетах, если в слоты DOGE подставить свечи этой монеты.

**Метод.** Веса `prod_doge_1h_v1`, три нуля EXP-105, пороги 0.48/0.52, динамическая температура по часовым свечам самой монеты, окно как у живого бота, флип со следующего открытия, комиссия 4 б.п. на сторону, капитал целиком. BTC, ETH, макро и часы не меняются. Фандинг пустой, как у живого сборщика фич. Список — текущий топ-28 USDT по объёму за 24 часа на споте Binance, без стейблов и без UP/DOWN/BULL/BEAR. Это сегодняшний объём, не капитализация на начало года. Каждый год стартует с 10 000. Пороги под монету не подбирались. 2026 в выбор монет не входил: смотрели оба года.

**Контроль DOGE на этом же пути.** 2025 **+10166.4%**, просадка −30.8%, 217 сделок. 2026 **+244.6%**, просадка −37.0%, 179 сделок.

**Оба года в плюсе:** VTHO, DOGE, ZRO, XRP, LTC, BTC, SUI, LINK.

VTHO: 2025 **+4185.8%**, просадка −54.8%, 170 сделок. 2026 **+279.2%**, просадка −54.4%, 225 сделок. HYPE и XAUT в 2025 не считались: истории на год нет. HYPE в 2026 — одна сделка, +2.1%.

**Вывод.** Замороженное правило собаки на чужих свечах не разваливается на всех монетах сразу. VTHO в этом прогоне второй по 2026 после самого себя. Это не обучение на VTHO и не подбор порогов под VTHO. Сколько часов сигнал совпадает с собакой, не считали, поэтому долю биткоина и эфира в этом плюсе не отделяли. В прод ничего не переносилось.

**Артефакты:** `bot_binance_flip/coin_pass_top28.py`; `analysis_exp100_20feat/doge_model_coin_pass.json`.

## EXP-107 — VTHO, цель H6, флип и жадные нули (2026-10-02)

**Вопрос.** Если обучить VTHO так же, как XRP, удержит ли флип 2026, и какие фичи можно выключить по правилу обоих лет.

**Обучение.** VTHOUSDT 1h, 138 фич, колонки `doge_` — это VTHO. Обучение до 2025-02-01. Optuna 60 триалов, 5 сидов, 8 ч 11 мин. Счёт — максимальная доходность фиксированного удержания на февраль–декабрь 2025. 2026 в массив поиска не входил. Бары с 2020-07-17.

Лучший триал 4: счёт **+640.1%**, среднее по пяти сидам +184.0%, горизонт 6, d_model 64, 4 головы, 2 слоя, ff 2, seq 60, dropout 0.2, lr 2.45e-4, batch 128. Второй — триал 21: счёт **+634.0%**, среднее **+201.9%**, те же H6, d 64, seq 60, но 5 слоёв и dropout 0.35. Поиск берёт максимум сида, не среднее, поэтому победил триал 4.

**Чемпион триала 4.** 100 сидов. По валидации сид 2: **+640.1%**, просадка −33.1%, 707 сделок, пороги цели 0.60/0.50. Файл `transformer2602/runs/champion_1h_vtho_target/champion_return.pt`.

**Флип.** Сетка 0.45–0.65 шаг 0.01, пара по доходности 2025, сделок не меньше 5. Победитель **0.53 / 0.45**. 2025 **+13661.4%**, просадка −47.9%, 130 сделок. Январь один **+809.4%**. 2026 **−100.0%**, просадка −100%, 75 сделок: сентябрь обнулил счёт.

2025 по месяцам: +809.4, +93.8, +35.2, +171.5, +10.7, +7.2, +4.7, +6.9, +16.0, +26.5, −4.8, +14.7.

2026 по месяцам: +52.0, +1.8, −1.3, −15.6, −23.9, +11.7, +0.1, −16.3, −100.0, +1.7. Октябрь неполный, данные около 2026-10-02.

**Нули.** Пороги 0.53/0.45 заморожены. Правило EXP-101. Кандидатов 27. Оставлены два, по порядку: `doge_range_pct` (часовой диапазон VTHO, (high−low)/close) и `eth_wick_lower` (нижняя тень эфира). После второго: 2025 **+23818.9%**, просадка −47.8%, 134 сделки. 2026 **−30.7%**, просадка −93.0%, 85 сделок. 2026 участвовал в решении оставить эти нули. Просадка 2026 остаётся почти полным сливом.

**Вывод.** Своя модель VTHO на флипе 2026 не живёт. Плюс +279% в EXP-106 — другое исполнение: чужие замороженные веса, пороги 0.48/0.52, динамическая температура. Здесь пороги и сид выбраны по максимуму 2025, и январь этот максимум раздувает. В прод не писать. Веса триала 4 не перезаписывались следующим прогоном.

**Артефакты:** `transformer2602/vtho_pipeline.py`, `transformer2602/alt_flip_then_greedy.py`; `transformer2602/runs/optuna_1h_vtho_target/best_params.json`, `trial21_params.json`; `analysis_exp100_20feat/vtho_flip_test.json`, `vtho_greedy_both_years.json`.

## EXP-108 — VTHO, триал 21, валидация до 31 мая 2026 (2026-10-02)

**Вопрос.** Если сид выбирать не по февралю–декабрю 2025, а по окну до 31 мая 2026, выживет ли лучший сид на месяцах после этой даты.

**Метод.** Гиперпараметры триала 21, не триала 4. 100 сидов. Обучение по-прежнему до 2025-02-01. Валидация 2025-02-01 … 2026-05-31, это явное включение января–мая 2026 в выбор сида и порогов цели. Июнь и дальше в выбор не входят. Веса триала 4 не трогались. Прогон остановлен после сида 42: готово 43 сида из 100, доучивание не запускалось.

**Лучший из этих 43 — сид 18.** Фиксированное удержание на валидации **+414.1%**, просадка −38.8%, 1064 сделки, пороги 0.60/0.50. Первые пять сидов, по которым оптуна мерила триал 21, здесь все ниже +152%.

**Тест.** Те же пороги, окно 2026-06-01 … 2026-10-02. **−66.7%**, просадка −78.3%, 299 сделок. Месяцы: июнь −18.6%, июль +1.1%, август −15.2%, сентябрь −54.8%, октябрь +5.7% (не закрыт). Валидация при повторном обучении того же сида совпала: +414.1%.

**Вывод.** Удлинение валидации до мая меняет победителя, но следующий кусок года этот победитель не держит. Это фиксированное удержание, не флип EXP-107. В прод не писать.

**Артефакты:** `transformer2602/vtho_trial21_seed_test.py`; `transformer2602/runs/champion_1h_vtho_target_trial21/seed18.pt`, `leaderboard.json`; `analysis_exp100_20feat/vtho_trial21_seed_test.json`.

## EXP-109 — LTC, оптуна цели H6, идёт (2026-10-02)

**Вопрос.** Есть ли у лайткоина свой триал на том же разрезе, что у XRP и VTHO.

**Метод.** LTCUSDT 1h, 138 фич, колонки `doge_` — это LTC. Обучение до 2025-02-01. 30 триалов, 10 сидов. Счёт — максимальная доходность фиксированного удержания на февраль–декабрь 2025. Бары поиска обрезаны на 2026-01-01. Старт 2026-10-02 23:37, время местное. На момент записи первый триал ещё не завершён.

**Вывод.** Пока нет. Чемпион, флип и нули не запускались. Живой DOGE-бот не менялся.

**Артефакты:** исследование `ltc1h_target_plus_h3to6`, каталог `transformer2602/runs/optuna_1h_ltc_target/`.

## EXP-110 — стационарный бутстрап сделок 1h DOGE (2026-10-03)

**Вопрос.** Отделяется ли годовая доходность живого флипа от шума, если соседние сделки не рвать по одной.

**Метод.** Стационарный бутстрап Политиса–Романо (1994). Ряд — доходности закрытых сделок по времени входа. Длина блока — Политис–Уайт (2004) с поправкой Паттона–Политиса–Уайта (2009), для среднего сделки. Интервал — процентильный, 2.5% и 97.5%, на сложной доходности года. 10 000 повторов, сид 20261003. Средняя длина блока вышла около 1, то есть зависимости между закрытыми сделками метод не нашёл и ряд почти как обычный бутстрап.

**Без нулей.** Файл `analysis_flip_causal_2025_2026/trades.csv`. Пороги 0.48/0.52, три нуля не стоят. Данные до 2026-09-22.

| | сделок | факт | 95% | доля годов ≤ 0 |
|---|---:|---:|---|---:|
| 2025 | 264 | +6550.4% | от +1089% до +40982% | 0 |
| 2026 | 169 | +46.1% | от −38.9% до +262.3% | 0.202 |
| оба | 433 | +9616.3% | от +1237% до +79467% | 0 |

2025 ноль не накрывает, но ширина огромная, и этот год выбирал сид и пороги. 2026 ноль накрывает: примерно каждый пятый искусственный год в минусе.

**С тремя нулями.** `doge_di_minus`, `btc_ma_ratio_24`, `eth_obv_chg_z24`. Тот же серверный счёт, что в EXP-101: непрерывный капитал даёт 2025 **+7290.9%**, 2026 **+218.8%**, весь путь **+23465.5%**. Бутстрап считает сделки, открытые внутри года, поэтому 2026 по сделкам **+196.8%** (66 сделок), а не +218.8%: позиция через Новый год сидит в годовом срезе счёта, не в этом списке. Данные фич до 2026-09-29 04:00.

| | сделок | факт | 95% | доля годов ≤ 0 |
|---|---:|---:|---|---:|
| 2025 | 218 | +7840.2% | от +1373% до +49013% | 0 |
| 2026 | 66 | +196.8% | от +41.0% до +547.4% | 0.0017 |
| оба | 284 | +23465.5% | от +3659% до +163624% | 0 |

**Вывод.** Без нулей 2026 от шума не отделяется. С нулями нижняя граница 2026 **+41%**, в минус попали 17 искусственных годов из 10 000. Это устойчивость уже отобранного пути, не чистая проверка: 2026 участвовал в решении оставить эти три нуля. В прод ничего нового не писалось.

**Артефакты:** `analysis_flip_causal_2025_2026/stationary_bootstrap.py`, `bootstrap_three_zeros.py`; `stationary_bootstrap.json`, `stationary_bootstrap_three_zeros.json`; `trades_three_zeros.csv`.

## EXP-111 — LTC, оптуна закончена, 2026 только в строке (2026-10-03)

**Вопрос.** Продолжение EXP-109. Есть ли триал, чей лучший сид 2025 ещё и жив в 2026.

**Метод.** Тот же разрез: обучение до 2025-02-01, счёт — максимальная доходность фиксированного удержания на февраль–декабрь 2025, 30 триалов, 10 сидов, 138 фич. Первый прогон сняли после триала 0: в строке не было 2026. Поиск начат заново. После каждого триала лучший сид 2025 прогоняется по 2026 с теми же порогами. Эта цифра печатается и в счёт оптуны не входит. 15 ч 20 мин.

**Победитель по счёту.** Триал 4: **+250.9%**, среднее по 10 сидам +68.0%. 2026 этого сида **−72.5%**, просадка −75.2%, 825 сделок. Горизонт 6, d=64, 2 слоя, seq 60, batch 128, dropout 0.20.

**Плюс в 2026 только у двух триалов.** Триал 10: счёт **+189.6%**, среднее +88.1%, 2026 **+23.9%**, просадка −22.8%, 405 сделок. Горизонт 5, d=64, 3 слоя, ff 4, seq 60, batch 64, dropout 0.30, lr 4.11e-5. Лучший из десяти сидов поиска — сид 2, пороги 0.55/0.55. Триал 23: счёт +150.4%, 2026 **+6.8%**, просадка −23.0%, 381 сделка. Остальные 28 триалов в 2026 в минусе. Ближе всех к нулю триал 5, −1.5%.

**Вывод.** Поиск по 2025 выбирает триал 4, и он в 2026 не живёт. Плюс 2026 есть у триала 10 и слабо у триала 23. Живой DOGE-бот не менялся.

**Артефакты:** `transformer2602/runs/optuna_1h_ltc_target/train.log`, `best_params.json`, `trial10_params.json`.

## EXP-112 — LTC, сид 2 триала 10, флип и жадные нули (2026-10-03)

**Вопрос.** Держит ли флип сида, который на фиксированном удержании дал плюс и в 2025, и в 2026, и какие фичи можно выключить по правилу обоих лет.

**Чемпион.** 100 сидов триала 10. Валидация февраль–декабрь 2025. Для каждого сида 2026 печатается с порогами этой валидации и сид не выбирает. Прогон остановлен после сида 39. Веса сотни не записаны.

Сид 2 повторил оптуну: фиксированное удержание **+189.6%**, просадка −16.1%, пороги цели 0.55/0.55, 2026 **+23.9%**, просадка −22.8%, 405 сделок. Повторное обучение того же сида дало те же +189.6% и 0.55/0.55. Среди уже посчитанных 40 сидов по 2025 выше были сид 20 (+219.5%, 2026 −65.6%) и сид 21 (+190.0%, 2026 −32.2%). Взяли сид 2, потому что 2026 у него в плюсе. Это выбор глазами, не счёт оптуны.

**Флип.** Сетка 0.45–0.65 шаг 0.01, пара по доходности 2025, сделок не меньше 5. Победитель **0.48 / 0.45**. 2025 **+927.0%**, просадка −39.5%, 170 сделок. 2026 **+39.4%**, просадка −33.1%, 137 сделок. Октябрь 2026 неполный, данные около 2026-10-03.

2025 по месяцам: +144.6, +80.0, +20.3, +21.5, +2.0, +0.3, −16.3, +17.2, −13.4, +16.6, +20.6, +30.5.

2026 по месяцам: −14.0, +13.5, +11.8, +4.9, −5.4, +3.8, +18.2, −18.0, +30.1, −1.7.

**Нули.** Пороги 0.48/0.45 заморожены. Правило EXP-101. Кандидатов 12 из 138. Оставлены четыре, по порядку: `doge_vol_regime` (режим объёма лайткоина), `rel_str_btc_24` (сила к биткоину за 24 часа), `eth_macd_signal`, `doge_cvd_slope12` (наклон CVD лайткоина). Колонки `doge_` — это LTC. После четвёртого: 2025 **+1415.1%**, просадка −37.3%, 160 сделок. 2026 **+124.9%**, просадка −23.8%, 143 сделки. 2026 участвовал в решении оставить эти нули.

Путь: после первого 2025 +967.1% / 2026 +105.1%; после второго +1059.2% / +120.7%; после третьего +1215.4% / +123.3%; после четвёртого +1415.1% / +124.9%.

**Вывод.** Флип сида 2 в 2026 в плюсе и без нулей. Четыре нуля поднимают оба года. В прод не писать: сид выбран по плюсу 2026, и 2026 же голосовал за нули. Живой DOGE-бот не менялся.

**Артефакты:** `transformer2602/ltc_trial10_seed2.py`, `transformer2602/alt_flip_then_greedy.py`; `transformer2602/runs/champion_1h_ltc_target_trial10/seed2.pt`, `leaderboard.json`; `analysis_exp100_20feat/ltc_flip_test.json`, `ltc_greedy_both_years.json`.

## EXP-113 — XRP, оптуна 30×10, в счёте таргет, флип 2026 только в строке (2026-10-04)

**Вопрос.** Переделать рипл так, чтобы выживаемость 2026 была видна сразу, и при этом флип не выбирал архитектуру.

**Метод.** Новая оптуна, старая `xrp1h_target_plus_h3to6` и чемпион сида 1 не тронуты. Обучение до 2025-02-01, 138 фич, горизонт 3..6, 30 триалов, 10 сидов. Счёт — максимальная доходность фиксированного удержания на февраль–декабрь 2025. После триала лучший сид 2025 печатается дважды и ни одна цифра в счёт не входит: таргет 2026 с порогами валидации и флип 2026. Пороги флипа с календаря 2025, сетка 0.45–0.65 шаг 0.01, сделок не меньше 5, пара по доходности 2025. 15 ч 16 мин. База `transformer2602/runs/optuna_1h_xrp_target_y2026`.

**Победитель по счёту.** Триал 28: **+252.0%**, среднее +90.6%. Таргет 2026 **−61.6%**, просадка −64.0%, 660 сделок. Флип 2026 **−49.0%**, просадка −63.7%, 124 сделки. Горизонт 6, d=64, 5 слоёв, seq 60, batch 128, dropout 0.35.

**Флип 2026 в плюсе у 13 триалов из 30.** Между сильными по счёту и живыми во флипе выбор глазами не складывается: гиперпараметры разные. Четыре кандидата:

| Триал | Счёт 2025 | Таргет 2026 | Флип 2026 | Просадка флипа | Сделок | Что за модель |
|------:|----------:|------------:|----------:|---------------:|-------:|---|
| 8 | +193.8% | −27.7% | +35.6% | −41.4% | 98 | d=256, 5 слоёв, seq 120, dropout 0.05 |
| 19 | +63.9% | −12.3% | +68.3% | −31.9% | 49 | d=128, горизонт 3, seq 120, batch 64 |
| 23 | +103.5% | +36.5% | +40.5% | −29.0% | 52 | d=64, 6 слоёв, seq 60, dropout 0.35 |
| 25 | +145.7% | +2.9% | +45.6% | −47.2% | 65 | d=64, 4 слоя, seq 168, batch 512 |

У ширины 64 в файле поиска голова записана как 8, в обучении принудительно 4. Общее у этих четырёх только ff=2. Похожи друг на друга триалы 13, 21 и 22: d=64, 5 слоёв, seq 60, batch 128, dropout около 0.30, горизонт 6. Счёт 2025 у них +168…+231%, флип 2026 слабый, +3.6…+21%.

**Выбор.** В чемпионы взят триал 23 как самый ровный: и таргет 2026 **+36.5%** (просадка −23.7%, 210 сделок), и флип 2026 **+40.5%** (пороги 0.55/0.56, просадка −29.0%, 52 сделки). Среднее по 10 сидам +46.5%. Лучший сид поиска — 8. Это выбор глазами по 2026, не счёт оптуны. Чемпион: 100 сидов, сид по таргету февраль–декабрь 2025, строка 2026 сид не выбирает. Флип и нули после чемпиона. Живой DOGE-бот не менялся.

**Артефакты:** `transformer2602/runs/optuna_1h_xrp_target_y2026/train.log`, `best_params.json` (триал 28), `trial23_params.json`. Чемпион: `transformer2602/runs/champion_1h_xrp_target_trial23`.

## EXP-114 — плотность DOGE 1h сверх волатильности (2026-10-06)

**Вопрос.** Даёт ли маленькая условная плотность информацию сверх реализованной волатильности, и хватает ли этой добавки на издержки.

**Метод.** Зафиксирован до запуска в `analysis_density_doge/PREREG.md`. DOGEUSDT 1h. Горизонты 1, 3, 6, 12, 24. Таргет r = ln(close_{t+h}/close_t) / σ_t, σ_t — стандартное отклонение часовых лог-доходностей за 24 бара. 41 корзина от −4 до +4, хвосты в крайних. Нулевая модель: Стьюдент с центром 0, масштаб и степени свободы с train, 2 параметра. Добавка: 9 параметров, среднее и логарифм масштаба линейны по mom6, mom24 и отношению волатильности 6/24. λ из сетки 0, 0.001, 0.01, 0.1, 1 выбирается на валидации. Пять полугодовых тестов с 2023-07-01 по 2026-01-01, train растущий, 2026 закрыт. Прирост правдоподобия: блоки длины h, 4000 повторов. Экономика открывается только если правдоподобие прошло ворота. Издержки 4 bps на сторону, 2 bps проскальзывания, funding. Вход по open следующего бара. Живой бот не менялся.

**Правдоподобие.** Прирост на бар и 95% интервал по сшитым фолдам:

| h | Фолды в плюсе | Среднее | Интервал | Ворота |
|--:|:-------------:|--------:|----------|:------:|
| 1 | 5/5 | +0.0080 | [+0.0059, +0.0100] | да |
| 3 | 5/5 | +0.0053 | [+0.0031, +0.0076] | да |
| 6 | 5/5 | +0.0024 | [+0.00004, +0.0046] | да |
| 12 | 4/5 | +0.0021 | [−0.0011, +0.0052] | нет |
| 24 | 3/5 | −0.0025 | [−0.0064, +0.0017] | нет |

CRPS почти не двигается. На h=1 сдвиг порядка 0.594 → 0.591, в двух фолдах добавка чуть хуже нулевой.

**Калибровка.** PIT нулевой модели не плоский ни в одном фолде: хи-квадрат отвергает равномерность везде. На h=1 добавка выравнивает PIT в большинстве фолдов (один фолд всё ещё кривой). С h=3 и дальше PIT добавки тоже кривой. С h=6 квантиль 5% уже лежит в хвостовой корзине: σ_t — часовая вола, поэтому r растёт с горизонтом и сетка ±4 перестаёт разрешать тело распределения. Это свойство зафиксированного σ, прогон не переигрывался.

**Экономика.** Открыта на h=1, 3 и 6. Плюс со средним net > 0 и не меньше 30 сделок только в одном фолде из пяти на каждом из этих горизонтов. Худший фолд: h=1 **−72.9%**, h=3 **−73.1%**, h=6 **−64.7%**. Ворота «3 из 5» и «не ниже −15%» не пройдены. На h=12 и h=24 экономика не открывалась.

**NO-GO по всем пяти горизонтам.** Ветку закрыть. Как инструмент ширины остаётся нулевая модель. На часе степени свободы около 3, масштаб около 0.72, множитель стандартного отклонения r около **1.24**. Лог-доходность горизонта имеет ширину 1.24·σ_t, размер позиции пропорционален 1/(1.24·σ_t)². Квантиль 5% на часе около **−1.64·σ_t**. На длинных горизонтах для стопа брать непрерывный Стьюдент, не корзину ±4.

**Артефакты:** `analysis_density_doge/PREREG.md`, `run_density.py`, `results.json`.

## EXP-116 — теория лент, edge на уровнях до ML (2026-10-06)

**Вопрос.** Есть ли у касания EMA50, EMA200 или swing high/low статистический край в пользу отскока, после 8 bps round-trip.

**Метод.** Зафиксировано до счёта. Активы DOGE, BTC, ETH, XRP, LTC. Таймфреймы 1h, 4h, 1d, 1w. k = 0.5, порог исхода X = 10 bps. Сетка по k и X не запускалась. Swing — фрактал 4 бара влево и 4 вправо, живёт N баров после подтверждения (168 / 42 / 7 / 4). Касание по close, одно на вход в зону. Уровень и ширина заморожены на баре касания. Если на одном баре сняты оба барьера, событие неоднозначно: оно в числе событий и ни в одной вероятности. Ожидание 24 часа, 4 дня, 7 дней, 4 недели. Пять фолдов 2023H2…2025H2. 2026 не голосует. ML не обучалась.

**Результат.** Ни один уровень не кандидат ни на одном активе. GO нет: для этого нужно два актива, не набралось ни одного.

Там, где событий больше 100, EV_net на сшитом окне отрицательный. Ближе всех к нулю EMA50 на часе у XRP: 1851 событие, P(отскок) 0.42, P(пробой) 0.45, средний отскок 59 bps, средний пробой 70 bps, EV **−14.4 bps**. Плюсовых фолдов 0. На часе по пяти активам картина одна: пробой чуть чаще и чуть крупнее отскока, EV от −14 до −27 bps. День и неделя чаще не добирают 100 событий. Плюсы там есть только на выборках из 2–20 касаний, в зачёт они не идут.

**NO-GO.** Ветку закрыть. Пороги после просмотра не двигались.

**Артефакты:** `causal_flip/exp116_level_bands.py`, `causal_flip/reports/exp116_level_bands/results.json`.

## EXP-117 — сетка ширины ленты, отскок и пробой (2026-10-06)

**Вопрос.** Меняется ли знак края, если ширину зоны не фиксировать на k = 0.5, и есть ли край у ставки на пробой.

**Метод.** Зафиксировано до счёта в `causal_flip/reports/exp117_lenta_width_grid/prereg.json`. Те же уровни, активы, фолды и барьеры, что в EXP-116. X = 10 bps, издержки 8 bps. Считаются оба ожидания: ставка на отскок и ставка на пробой. Они в сумме всегда −16 bps, вместе в плюс выйти не могут. Кандидат — это уровень, одна ширина и одна сторона. GO, если эта же сторона проходит на двух и более активах. Вариант A: k = 0.25, 0.5, 1, 2, 3. Варианты B и C запускаются только если у A нет GO. 2026 не голосует. ML нет. k = 0.5 на часовой EMA50 собаки повторил EXP-116: 1717 событий, отскок −18.12 bps.

**Вариант A дал GO, B и C не открывались.** 34 кандидата. День и неделя не прошли. На часе и на 4h знак зависит от ширины.

Узкая зона, k = 0.25 и 0.5: в плюсе пробой. Широкая, k = 2 и 3: в плюсе отскок. На k = 1 час в основном не проходит, на 4h проходит точечно.

Часовая EMA50, сшитое окно, bps:

| k | DOGE отскок / пробой | BTC | ETH | XRP | LTC |
|--:|---|---|---|---|---|
| 0.25 | −25.1 / **+9.1** | −16.9 / **+0.9** | −21.4 / **+5.4** | −21.9 / **+5.9** | −20.9 / **+4.9** |
| 0.5 | −18.1 / **+2.1** | −15.8 / −0.2 | −19.0 / **+3.0** | −14.4 / −1.6 | −21.9 / **+5.9** |
| 1 | +3.4 / −19.4 | −6.7 / −9.3 | −7.8 / −8.2 | +1.0 / −17.0 | −4.7 / −11.3 |
| 2 | **+26.9** / −42.9 | **+9.7** / −25.7 | **+9.6** / −25.6 | **+21.6** / −37.6 | **+17.5** / −33.5 |
| 3 | **+55.3** / −71.3 | **+18.2** / −34.2 | **+24.4** / −40.4 | **+36.5** / −52.5 | **+34.7** / −50.7 |

Жирный — сторона, которая прошла ворота на этом активе. У XRP при k = 1 отскок на сшитом окне +1.0 bps, но плюсовых фолдов меньше трёх, поэтому не GO.

Фолды часовой EMA50, пробой при k = 0.25. Собака, все пять в плюсе: +0.1, +13.5, +16.0, +7.3, +9.0. Биткоин: −1.3, +3.9, −0.1, +1.8, +0.2, три из пяти, худший −1.3 bps. Порог −15% (−1500 bps) нигде не сработал.

**Почему знак переворачивается.** В узкой зоне барьер пробоя близко к цене входа, барьер отскока — на дальней стороне. В широкой зоне цена входит у края, и отскок — это возврат за этот край, а пробой — путь через всю зону до уровня. Средний отскок при k = 3 большой, потому что в зачёт идёт экстремум бара, который вышел из зоны.

**Вердикт: GO по замороженным воротам варианта A.** В прод не писать: это ещё не исполнение, а классификация барьеров. Список из 34 строк в `summary.json`.

**Артефакты:** `causal_flip/exp117_lenta_width_grid.py`, `tests/test_exp117_lenta_width_grid.py`; `causal_flip/reports/exp117_lenta_width_grid/summary.json`, `events.csv`, `fold_results.csv`, `width_grid.csv`, `prereg.json`.

## EXP-118 — величина EV, семь фолдов, режимы (2026-10-06)

**Вопрос.** У прошедших в EXP-117 ширин есть ли величина, а не только знак. Держится ли она на семи фолдах, включая 2026. Меняется ли в тренде, пиле и шоке. Что на дне и неделе.

**Метод.** Зафиксировано в `causal_flip/reports/exp118_lenta_ev_folds_regimes/prereg.json`. Барьеры те же, что в EXP-116: отскок выходит из зоны, пробой проходит уровень на 10 bps. Ширины A: k = 0.25, 0.5, 1, 2, 3. B открывается только если у A ноль GO. Ворота жёстче: n ≥ 100, EV > 0 на сшитых семи фолдах, плюс минимум на 4 фолдах из 7, худший фолд не ниже −20 bps, медиана фолдов с событиями > 0, Sharpe = среднее pnl сделки / его стандартное отклонение > 0.1. 2026 входит в эти 4/7 и ширину не выбирает. Режим на баре касания: ADX(14) > 25 тренд, ADX < 20 пила, ATR > 2× медианы ATR за предыдущие 168 баров. Режим в ворота не входит. ML нет.

**Величина.** Тонкий пробой часовых EMA из EXP-117 ворота не прошёл: Sharpe около 0.1 не набирается. Прошли 25 пар, все на 1h и 4h.

Узкий пробой, k = 0.25, 4h EMA50, сшитое окно: собака **+35.2 bps**, Sharpe 0.19, 7/7. Биткоин **+13.7**, 0.13, 6/7. Эфир **+27.4**, 0.22, 5/7. Рипл **+15.5**, 0.11, 5/7. Лайткоин **+27.2**, 0.21, 7/7.

Широкий отскок, k = 2, часовая EMA50: собака **+24.8 bps**, Sharpe 0.24, 7/7. Биткоин **+10.0**, 0.16. Эфир **+9.9**, 0.13. Рипл **+20.0**, 0.19. Лайткоин **+17.9**, 0.19. Все семь фолдов в плюсе. При k = 3 те же уровни дают больше: часовая EMA50 собаки **+47.6 bps**, Sharpe 0.34.

**Фолды.** Часовая EMA50 собаки, k = 2, отскок: худший фолд **+11.8 bps**, медиана **+20.6**, лучший **+41.0**. Пробой swing high на часе у собаки, k = 0.25: худший **+3.4**, медиана **+13.3**, лучший **+18.3**, тоже 7/7.

**Распределение сделки, не фолда.** У этого пробоя квантили pnl: 5% −100 bps, 25% и 50% **−8 bps**, 75% +40, 95% +164. Медиана сделки — это голые издержки: много касаний неоднозначны и в плюс среднего их вытягивает хвост. У широкого отскока той же собаки медиана сделки уже **+21.6 bps**, 5% −112, 95% +180. Здесь плюс не только в хвосте.

**Режимы.** Знак не переключается. Часовая EMA50, k = 2, отскок: собака тренд +23.9 (1037), пила +16.5 (192), шок +57.6 (64). Биткоин: +10.3, +8.1, +24.4. Узкий пробой тоже одного знака в тренде и в пиле. Шок — десятки событий, отдельный режим из него не делается.

**День и неделя.** Тот же знак, но n < 100. Дневная EMA50 собаки, k = 2, отскок +138 bps, Sharpe 0.44, 7 фолдов в плюсе, событий 76. Неделя — 7–11 событий. Ворота по числу не пройдены.

**Вердикт: GO, 25 пар варианта A.** B не открывался. В прод не писать: у узкого пробоя типичная сделка не в плюсе, плюс сидит в хвосте. Список пар в `summary.json`.

**Артефакты:** `causal_flip/exp118_lenta_ev_folds_regimes.py`, `tests/test_exp118_lenta_ev_folds_regimes.py`; `causal_flip/reports/exp118_lenta_ev_folds_regimes/summary.json`, `events.csv`, `fold_results.csv`, `regime_results.csv`, `ev_distribution.csv`, `prereg.json`.

## EXP-119 — полная сетка лент, X и планка 20 bps (2026-10-06)

**Вопрос.** Какая величина у края, если порог исхода не фиксирован на 10 bps. Проходит ли он планку EV > 20 bps. Где сильнее: час, 4 часа, день или неделя. Держится ли знак в тренде, пиле и шоке.

**Метод.** Зафиксировано до счёта в `causal_flip/reports/exp119_lenta_full_grid/prereg.json`. Уровни, активы, ширины A и барьеры те же, что в EXP-116 и EXP-118. Отскок — выход из зоны на X bps. Пробой — проход уровня на X bps, не зеркальный выход с другой стороны зоны. X = 5, 10, 20, 50 bps. Издержки 8 bps. Ворота EXP-118 плюс сшитый EV > 20 bps. 2026 входит в 4 из 7 и не выбирает ни k, ни X: считаются все клетки. Режим в ворота не входит. Варианты B и C не запускались. ML нет. На X = 10 часовая EMA50 собаки, k = 2, отскок повторил EXP-118: **+24.8 bps**.

**Вердикт: GO, 73 пары.** Все на 1h и 4h. День и неделя — ноль пар: ни одна клетка не набрала 100 событий. По X пар 16 / 19 / 19 / 19. Отскок 48, пробой 25. Знак тот же, что раньше: узкая зона (k = 0.25 и 0.5) — пробой на 4h, широкая (k = 2 и 3) — отскок на 1h и 4h.

**Величина, часовая EMA50, отскок, сшитое окно, bps.** Строка — X = 5 / 10 / 20 / 50. Жирным — клетка прошла планку 20 bps и остальные ворота.

| k | DOGE | BTC | ETH | XRP | LTC |
|--:|---|---|---|---|---|
| 2 | **22.5 / 24.8 / 27.8 / 35.2** | 7.8 / 10.0 / 14.5 / **21.3** | 9.1 / 9.9 / 12.3 / **22.4** | 17.1 / **20.0 / 24.4 / 33.9** | 17.3 / 17.9 / **21.6 / 28.3** |
| 3 | **45.6 / 47.6 / 53.0 / 68.7** | 16.4 / 17.5 / **23.5 / 35.0** | **22.7 / 24.0 / 28.5 / 42.8** | **31.8 / 34.2 / 38.6 / 57.7** | **32.0 / 34.8 / 38.5 / 47.8** |

При k = 3 все пять активов на X = 20 и 50. Sharpe собаки на k = 3, X = 10: **0.34**, 7/7, худший фолд +15.3, медиана фолдов +50.4. На X = 50 Sharpe **0.42**, худший фолд +28.0.

**Узкий пробой, 4h EMA50, k = 0.25.** Собака **+34.4 / +35.2 / +35.5 / +31.8**, все четыре X прошли, Sharpe около 0.15–0.19. Эфир и лайткоин тоже на всех X. Рипл проходит только на 20 и 50 (**+27.5 / +31.2**). Биткоин **+12.3…+16.3**, Sharpe около 0.12, планку 20 не берёт. Число событий от X не зависит: касание определяется зоной, не порогом исхода.

**Таймфреймы.** Прошедших клеток актива: 4h — 216, 1h — 66, 1d — 0, 1w — 0. День того же знака и с большим средним, но выборка мала. Дневная EMA50, k = 2, отскок, X = 10: собака +138 bps, Sharpe 0.44, 76 событий. Эфир +191, 55 событий. В GO не входят.

**Режимы.** Знак не переключается. Собака, часовая EMA50, k = 3, отскок, X = 10: тренд +48.9 (748), пила +19.8 (19), шок +112 (76). На X = 50: +70.4 / +52.6 / +145. Пила здесь — десятки событий, вывод держит тренд. Узкий пробой 4h, X = 10: тренд +46.8 (96), пила +33.6 (228), шок +98 (14).

**Распределение.** По всем клеткам с n ≥ 100 медиана EV **−8 bps**: типичная клетка сетки не в плюсе, GO — правый хвост. У 282 прошедших клеток актива квантили EV: 5% **21.6**, 25% **26.2**, 50% **33.5**, 75% **44.3**, 95% **70.7**. Sharpe этих клеток: медиана 0.22, 5% 0.13. Медиана самой сделки у широкого отскока положительная (собака, k = 3, X = 10, q50 **+29.9 bps**). У узкого пробоя q50 остаётся **−8 bps** до X = 50: среднее вытягивает хвост.

Рост EV вместе с X — не отдельный край. Барьер дальше, в зачёт идёт больший экстремум бара. От X = 5 до X = 50 отскок собаки на k = 2 вырос с +22.5 до +35.2, а не на 45 bps.

**В прод не писать.** 73 пары в `summary.json`. Пороги после просмотра не двигались.

**Артефакты:** `causal_flip/exp119_lenta_full_grid.py`, `tests/test_exp119_lenta_full_grid.py`; `causal_flip/reports/exp119_lenta_full_grid/summary.json`, `events.csv`, `fold_results.csv`, `regime_results.csv`, `ev_distribution.csv`, `tf_comparison.csv`, `prereg.json`.

## EXP-120 — ленты в деньгах и корреляция с FLIP (2026-10-06)

**Вопрос.** Сколько bps на сделку, сколько сделок в год и не дублирует ли это дневной P&L флипа.

**Метод.** Зафиксировано в `causal_flip/reports/exp120_lenta_money/prereg.json`. Отскок только на k = 2 и 3. Пробой только на k = 0.25 и 0.5. X = 5, 10, 20, 50. Таймфреймы 1h, 4h, 1d. Барьер пробоя — проход уровня, как в EXP-116. Издержки 8 bps. Частота = число касаний / 3.25 года окна 2023-07-01…2026-10-01. EV в год = EV × частота. Ворота EXP-119 плюс EV в год > 500 bps и Пирсон с флипом < 0.5. Спирмен пишется, ворота не ставит. 2026 входит в 4 из 7 и k с X не выбирает. ML не обучалась.

Дневной флип: собака — готовая эквити `bot_binance_flip` с 2025-01-01. Рипл и лайткоин — прогон уже сохранённых весов с порогами из json. Сверка: доход 2025 должен совпасть с json в пределах 1 п.п., иначе корреляции нет. Биткоин и эфир сохранённого флипа не имеют. Дни без сделки ленты в перекрытии равны нулю.

**Сверка флипа.** Лайткоин 2025 **+926.98%**, совпал. Рипл вышел **+127.30%** против сохранённых **+128.40%**. Это больше 1 п.п., порог не сдвигал, корреляция рипла пустая.

**Вердикт: GO, 50 пар.** Все пары — собака и лайткоин, потому что только у них есть дневной ряд, прошедший сверку. 131 клетка актива прошла целиком. Пирсон этих клеток: 5% −0.13, медиана **+0.006**, 95% +0.19. Спирмен медиана −0.007. Все 131 имеют метку диверсификации. Дублирования (> 0.7) нет.

**Деньги, X = 10.** Часовая EMA50, k = 3, отскок:

| | n/год | EV, bps | медиана сделки | Sharpe | EV/год, bps | win | PF | Пирсон |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| DOGE | 253 | +47.6 | +29.9 | 0.34 | 12 058 | 0.81 | 4.30 | +0.01 |
| LTC | 246 | +34.8 | +21.1 | 0.32 | 8 546 | 0.79 | 3.46 | +0.17 |
| ETH | 254 | +24.0 | +17.6 | 0.28 | 6 096 | 0.78 | 2.51 | нет ряда |
| XRP | 250 | +34.2 | +22.8 | 0.28 | 8 537 | 0.79 | 3.04 | сверка не прошла |
| BTC | 291 | +17.5 | +14.7 | 0.30 | 5 111 | 0.79 | 2.59 | нет ряда, и EV < 20 |

Пробой 4h EMA50, k = 0.25, X = 10: собака EV **+35.2**, 135 сделок в год, EV/год 4 762 bps, Пирсон +0.14. Медиана сделки **−8 bps**, win 0.40. Лайткоин EV +27.2, 144/год, Пирсон +0.08, медиана тоже −8.

**Топ по EV/год** — часовые swing, k = 3, отскок, X = 50. Собака swing low: EV +37.7, **1 933 касания в год**, EV/год 72 862 bps, Пирсон −0.06. Это сумма перекрывающихся ставок: около пяти касаний в сутки при ожидании 24 часа. На один счёт без плеча на каждое касание столько не складывается.

**День.** Ни одна клетка не набрала 100 событий. Дневная EMA50, k = 2, отскок, X = 10: собака +138 bps, 76 событий, EV/год 3 230 bps. В GO не входит.

**В прод не писать.** EV/год в таблице — это EV, умноженный на частоту касаний, не доход одного депозита. Список 50 пар в `summary.json`.

**Артефакты:** `causal_flip/exp120_lenta_money.py`, `tests/test_exp120_lenta_money.py`; `causal_flip/reports/exp120_lenta_money/summary.json`, `events.csv`, `fold_results.csv`, `ev_frequency.csv`, `flip_correlation.csv`, `prereg.json`.

## EXP-121 — slippage, просадка, фолды, paper (2026-10-06)

**Вопрос.** Жив ли край, если на сторону заложить реалистичный slippage, и какая при этом просадка.

**Метод.** Зафиксировано в `causal_flip/reports/exp121_lenta_slippage_mdd/prereg.json`. Те же семьи, что в EXP-120. Slippage на сторону 2, 5, 10, 15 bps. Round-trip, который вычитается: **8, 14, 24, 34 bps**. Это 4 bps комиссии на круг плюс slippage с обеих сторон. GO считается только при slippage 10 (24 bps на круг). Остальные три стоимости пишутся и в отбор не входят. Просадка — последовательное реинвестирование каждой сделки в полный equity, в порядке касания. 5000 bps = 50% этой кривой. Calmar = EV/год ÷ эта просадка. Оба числа со сложенных ставок, не с одного депозита на все одновременные касания. Paper — три верхние строки EXP-120 по EV/год, список не пересортирован: часовой swing low собаки, swing high собаки, swing low рипла, все k = 3, отскок, X = 50. Окно 2026-10-06…2026-10-20. Свечи не качались. Флип тот же, что в EXP-120. Рипл снова не прошёл сверку 2025 (+127.30 против +128.40), порог 1 п.п. не двигался.

**Топ-10 EXP-120, EV в bps при slippage 2 / 5 / 10 / 15.** Сдвиг ровно на разницу издержек.

| | 2 | 5 | 10 | 15 | MDD при 10 | Calmar | фолды | прошёл |
|--|--:|--:|--:|--:|--:|--:|--:|--|
| DOGE swing low, X=50 | 37.7 | 31.7 | **21.7** | 11.7 | 4723 | 8.9 | 7/7, худший +11.4 | один актив |
| DOGE swing high, X=50 | 39.5 | 33.5 | **23.5** | 13.5 | 4443 | 9.3 | 7/7, худший +13.5 | один актив |
| XRP swing low, X=50 | 36.4 | 30.4 | 20.4 | 10.4 | 3694 | 10.6 | 7/7 | нет корреляции |
| LTC swing high, X=50 | 35.1 | 29.1 | 19.1 | 9.1 | 6217 | 5.5 | 7/7 | EV < 20 и MDD > 50% |
| XRP swing high, X=50 | 33.3 | 27.3 | 17.3 | 7.3 | 8980 | 3.4 | 6/7 | нет |
| LTC swing low, X=50 | 30.3 | 24.3 | 14.3 | 4.3 | 6298 | 4.3 | 7/7 | нет |
| DOGE swing low, X=20 | 28.7 | 22.7 | 12.7 | 2.7 | 5102 | 4.8 | 7/7 | нет |
| DOGE swing high, X=20 | 29.8 | 23.8 | 13.8 | 3.8 | 4297 | 5.7 | 7/7 | EV < 20 |
| ETH swing high, X=50 | 30.1 | 24.1 | 14.1 | 4.1 | 2947 | 8.3 | 7/7 | нет ряда флипа |
| DOGE swing low, X=10 | 25.9 | 19.9 | 9.9 | −0.1 | 5562 | 3.4 | 5/7 | нет |

При slippage 15 ни одна из этих десяти не держит EV > 20. У двух собак при X = 50 медиана сделки на slippage 10 всё ещё плюс (+40.7 и +43.0). Хвост 5% около −186 bps. Худшая сделка около −1670 bps. Серия убытков 11–13.

**Вердикт: GO, 9 пар, все собака и лайткоин, slippage 10.** Это не тот топ по частоте. Часовые swing из топ-10 парой не прошли: у лайткоина EV ниже 20 или просадка больше 50%, у рипла нет корреляции.

Пары при slippage 10, bps:

| | DOGE EV / MDD / Calmar | LTC | фолды DOGE | медиана сделки |
|--|--|--|--|--|
| 4h swing low, k=2, отскок, X=5 | +30.5 / 2205 / 3.4 | +20.1 / 3399 / 1.4 | 7/7, худший +2.4 | +17.6 / +15.5 |
| 1h EMA50, k=3, отскок, X=20 | +37.0 / 893 / 10.5 | +22.5 / 1155 / 4.8 | 7/7, худший +7.0 | +22.7 / +13.6 |
| 4h EMA200, k=3, отскок, X=20 | +51.8 / 1669 / 2.3 | +44.8 / 2227 / 1.6 | 6/7, худший −12.5 | +39.3 / +41.3 |
| 1h EMA50, k=3, отскок, X=50 | +52.7 / 830 / 16.1 | +31.8 / 1132 / 6.9 | 7/7, худший +12.0 | +50.5 / +43.2 |
| 4h swing low, k=2, отскок, X=50 | +40.0 / 3547 / 2.8 | +24.3 / 3588 / 1.6 | 7/7, худший +10.4 | +56.5 / +51.6 |
| 4h EMA200, k=3, отскок, X=50 | +61.4 / 1542 / 3.0 | +54.7 / 2227 / 1.9 | 7/7, худший +0.6 | +60.3 / +70.3 |
| 4h swing low, k=0.25, пробой, X=10 | +23.3 / 1146 / 4.5 | +20.2 / 1456 / 3.0 | 6/7, худший −2.9 | **−24 / −24** |
| то же, X=20 | +25.3 / 1958 / 2.9 | +22.5 / 1312 / 3.7 | 6/7, худший −0.6 | **−24 / −24** |
| то же, X=50 | +30.4 / 1594 / 4.2 | +24.0 / 1650 / 3.1 | 7/7, худший +2.3 | +33.4 / +30.0 |

Пирсон этих пар от −0.07 до +0.14. Просадка на сложенном пути 8–36%, порог 50% не пробит. У узкого пробоя на X = 10 и 20 медиана сделки равна голым издержкам −24 bps: плюс среднего сидит в хвосте. У широкого отскока EMA50 медиана сама в плюсе.

**Paper.** Три правила записаны. В кэше последняя свеча **2026-09-30 23:00**. В окне 6–20 октября касаний **0**. Сравнивать с бэктестом нечего. Бэктест этих трёх при slippage 10: +21.7, +23.5, +20.4 bps на сделку.

**В прод не писать.** При slippage 15 bps топ по частоте уже ниже 20 bps на сделку. Список 9 пар в `summary.json`.

**Артефакты:** `causal_flip/exp121_lenta_slippage_mdd.py`, `tests/test_exp121_lenta_slippage_mdd.py`; `causal_flip/reports/exp121_lenta_slippage_mdd/summary.json`, `events.csv`, `fold_results.csv`, `slippage_grid.csv`, `mdd_results.csv`, `forward_paper.csv`, `prereg.json`.

## EXP-122 — ML-фильтр касаний, только OHLCV (2026-10-06)

**Вопрос.** Отличает ли LightGBM хороты.

**Метод.** Зафиксировано в `causal_flip/reports/exp122_lenta_ml_filter/prereg.json`. Активы DOшее касание от плохого после 24 bps на круг, лучше ли это нефильтрованной ленGE, XRP, LTC, только 1h. Отскок: четыре уровня, k = 3, X = 50. Пробой: те же уровни, k = 0.25, X = 10, 20, 50. Цель: 1, если pnl после 24 bps больше нуля. 30 признаков, список не расширялся. Модель на каждый актив отдельно: LightGBM, 500 деревьев, lr 0.02, 31 лист, min_child_samples 50, сид 42, без ранней остановки. Обучение на касаниях до 2024-12-31, с отступом 24 часа, чтобы метка не залезла в 2025. Порог из 0.5, 0.6, 0.7 выбран по средней pnl на 2025 при n ≥ 100. 2026 в выбор порога не входил. База — те же тестовые касания без фильтра, те же 24 bps. Фолды 2023–2024 в зачёт «4 из 7» не идут: модель их видела. Transformer не обучался.

**Порог.** На 2025 у всех трёх активов EV растёт с порогом. Сводный максимум — **0.7**. Он и стоит в вердикте.

**Тест 2026, порог 0.7.**

| | n | EV | база | Sharpe | EV/год | фолды после 2024 |
|--|--:|--:|--:|--:|--:|--|
| DOGE | 1991 | +18.7 | −4.9 | 0.18 | 49 894 | 4/4, худший 2026H2 +15.7 |
| XRP | 1674 | **+25.7** | −5.2 | 0.27 | 57 669 | 4/4, худший 2026H2 +21.8 |
| LTC | 1613 | **+31.7** | −7.4 | 0.32 | 68 424 | 4/4, худший 2025H2 +30.3 |

Собака бьёт базу и держит 4 фолда, но не берёт планку 20 bps. Рипл и лайткоин проходят всё.

**Вердикт: GO на XRP и LTC.** Порог после просмотра 2026 не менялся.

**Что именно отфильтровалось.** Главный признак по gain — **k**, затем x_bps. При пороге 0.7 модель оставляет почти только отскок: доля отскока 0.995–0.999. База отрицательная, потому что в неё замешан узкий пробой, который при 24 bps в минусе. Это сравнение завышает пользу фильтра. Внутри одного отскока (k = 3, X = 50) подъём меньше, но он есть: собака +13.8 → +18.7, рипл +17.4 → +26.0, лайткоин +19.4 → +31.7. Следующие признаки после k и x: доход биткоина за час, расстояние до swing low и swing high, z-score объёма за 24 бара, ATR в процентах, ширина Боллинджера, стохастик K, расстояние до EMA50.

EV/год снова сумма перекрывающихся касаний: около 1600–2000 сделок за девять месяцев 2026. На один депозит это не складывается.

**В прод не писать.** Transformer в этом прогоне не запускался.

**Артефакты:** `causal_flip/exp122_lenta_ml_filter.py`, `tests/test_exp122_lenta_ml_filter.py`; `causal_flip/reports/exp122_lenta_ml_filter/summary.json`, `events.csv`, `fold_results.csv`, `feature_importance.csv`, `threshold_grid.csv`, `prereg.json`.

## EXP-124 — полный бэктест ленты (2026-10-06)

**Вопрос.** Какая комбинация ленты держит полную кривую equity: доход, просадку, месяцы и фолды сразу.

**Метод.** Зафиксировано в `causal_flip/reports/exp124_lenta_full_backtest/prereg.json` до счёта. 93 правила: сетка 1h из задания, плюс все GO и топ-10 по EV/год из EXP-120/121 на своём таймфрейме, включая 4h. Пять активов. ML — книга EXP-122 при пороге 0.7, порог не двигался. Каждое касание садится на 100% текущего equity, по порядку касания, после 24 bps. Это сложенный путь: одновременные касания не делят один депозит. Calmar = CAGR в процентах / просадка в процентах. Sharpe ворот — средний pnl сделки / его стандартное отклонение. В зачёт фолдов идут пять окон до 2026, нужно 4 положительных сложных доходности. Два фолда 2026 записаны и в проход, в сортировку и в выбор правила не входят. Длительность просадки — дни от пика худшей просадки до её дна, порог строго меньше 180.

**Вердикт: GO, 31 правило на двух и более активах.** Из 468 клеток проходят 103. Выбор по заранее записанному правилу (максимум из минимального Calmar прошедших активов): **1h swing high, k = 3, отскок, X = 50**, собака и эфир. Минимальный Calmar пары 37.6, у собаки 200.7. Лайткоин на этом же правиле имеет Calmar 37, но просадка 6217 bps и ворота не берёт.

Calmar и полный доход возглавляют частые книги, около 1800 сделок в год, потому что каждая ставка реинвестируется в полный equity. Это не доход одного счёта. По Sharpe и по маленькой просадке впереди другое: **1h EMA50, k = 3, отскок, X = 50**. Оно проходит на всех пяти активах. У собаки Sharpe 0.35, просадка 830 bps, Calmar 45.6, 656 сделок, 5 из 5 фолдов, худший месяц −6.5%, медиана месяца +12.4%.

Фильтр EXP-122 по Calmar стоит выше всех трёх монет и ворота не проходит: вероятности есть только на 2025, положительных фолдов до 2026 только два. 2023–2024 модель видела, отдельной книги там нет.

**Топ по Calmar до 2026** (просадка в bps, Calmar, прошёл ли). Собака ML 3791 / 257 / нет. Лайткоин ML 2111 / 212 / нет. Собака swing high X=50: 4443 / 201 / да. Собака swing low X=50: 4723 / 183 / да. Рипл ML 4749 / 146 / нет. Рипл swing low X=50: 3694 / 129 / да. Собака EMA50 k=3 X=50: 830 / 46 / да. Эфир swing high X=50: 2759 / 38 / да. Лайткоин swing high X=50: 6217 / 37 / нет. Собака swing high X=20: 4297 / 31 / да.

**Топ по Sharpe.** Собака EMA50 1h k=3 X=50: 0.35. Собака EMA200 4h k=3 X=50: 0.35. Собака EMA50 4h k=2 X=20: 0.31. Дальше собака ML 0.30 (ворота нет) и широкие отскоки 4h около 0.29. Все десять, кроме ML, проходят.

**Топ по просадке.** Меньше всех у собаки 4h EMA200, k=0.25, пробой, X=10: 492 bps, Calmar 2.8, 109 сделок. Дальше широкие отскоки EMA200 на 4h, просадка 517–767 bps. Биткоин EMA200 k=3 X=20 имеет просадку 517 bps, но длительность 274 дня и не проходит.

**Три верхние по Calmar, месяцы до 2026.** Собака ML: худший месяц 0 (пустые месяцы до 2025), лучший +455%, медиана 0. Лайткоин ML: лучший +453%, медиана 0. Собака swing high X=50: худший −13.8%, лучший +122%, медиана +41.5%, std 38.5. Кривые: `equity_curves/DOGEUSDT_ml_exp122_p07.csv`, `LTCUSDT_ml_exp122_p07.csv`, `DOGEUSDT_swing_high_3_bounce_50.csv`. 2026 у этих трёх напечатан и порядок не выбирал: сложенный доход +3605%, +15187%, +410%.

**2026, отдельно, правила уже были выбраны.** Средний pnl одной сделки, bps. Собака / рипл / лайткоин / биткоин / эфир. Часовой EMA50, k=3, отскок, X=50: +18.9 (168), +28.8 (188), +37.5 (197), +18.4 (202), +31.8 (175). Оба полугодия в плюсе, кроме эфира во втором: −5.5 на 48 сделках. Просадка сложенного пути 557–1220 bps. Часовой swing high, тот же k и X: +14.1, +17.0, +20.2, +9.2, +12.5, сделок 1200–1400, просадка 2060–2825 bps. Swing low: +12.8, +15.8, +16.4, +8.1, +9.9. Фильтр 0.7 совпал с тестом EXP-122: собака +18.7, рипл +25.7, лайткоин +31.7. 2026 ничего не переставлял.

**В прод не писать.** Сложный Calmar частых касаний — не размер счёта.

## EXP-125 — одна ставка, выход по тику (2026-10-06)

**Вопрос.** Останется ли плюс у часового EMA50, k = 3, отскок, X = 50, если не открывать вторую ставку, пока первая не закрыта, и выходить по первой сделке на барьере, а не по краю часовой свечи.

**Метод.** Зафиксировано в `causal_flip/reports/exp125_lenta_tick_one_slot/prereg.json`. Пять пар. Касание по-прежнему на закрытии часа. Вход — первая aggTrade после этого закрытия. Выход — первая сделка, которая доходит до барьера отскока или пробоя, крайний срок 24 часа. Комиссия 4 bps на круг. Отдельные 20 bps проскальзывания не добавлялись: цена тика и есть исполнение. Лента на диске кончается 2026-09-11. 2026 правило не выбирал.

**Результат.** Перекрытие у этой книги маленькое: собака пропустила 34 касания из примерно 800. Средняя сделка в bps, один депозит:

| | сделок | средняя | счёт | просадка |
|--|--:|--:|--:|--:|
| DOGE | 774 | −8.1 | −56% | 67% |
| XRP | 750 | −8.1 | −53% | 59% |
| LTC | 729 | −15.7 | −72% | 76% |
| BTC | 817 | +2.8 | +17% | 33% |
| ETH | 756 | −14.7 | −71% | 75% |

Доля плюсовых сделок около 70%. Средняя всё равно минус: выигрыш закрывается у барьера, около цели, а проигрыш проходит всю ширину ленты. Часовая свеча раньше записывала край часа и раздувала выигрыш. 2026 у всех пяти в минусе или около нуля. Биткоин за всю историю чуть в плюсе, за 2026 тоже в минусе (−11.9 и −7.7).

**В прод не писать.** На одном депозите и на тиках эта книга не держит плюс.

**Артефакты:** `causal_flip/exp125_lenta_tick_one_slot.py`, `tests/test_exp125_lenta_tick_one_slot.py`; `causal_flip/reports/exp125_lenta_tick_one_slot/summary.json`, `trades.csv`, `fold_results.csv`, `prereg.json`.

**Артефакты:** `causal_flip/exp124_lenta_full_backtest.py`, `tests/test_exp124_lenta_full_backtest.py`; `causal_flip/reports/exp124_lenta_full_backtest/summary.json`, `prereg.json`, `fold_results.csv`, `top10_calmar.csv`, `top10_sharpe.csv`, `top10_return.csv`, `top10_mdd.csv`, `equity_curves/`, `monthly_returns/`.

## EXP-126 — все правила, тики, лимитные сделки (2026-10-06)

**Вопрос.** Останется ли плюс у книг EXP-124, если держать один рабочий ордер и входить лимитом в закрытие свечи касания, а прибыль снимать лимитом на барьере.

**Метод.** Зафиксировано в `causal_flip/reports/exp126_lenta_tick_limits/prereg.json` до счёта. Те же 93 правила, пять пар, 465 строк. Пока лимит стоит или позиция открыта, следующее касание пропускается. Неисполненный лимит занимает слот до конца окна. Покупка исполняется агрессивными продажами по цене не хуже лимита, продажа зеркально. Объём должен покрыть 9310 USDT. Цена входа — сам лимит, не тик. Прибыльный выход — такой же лимит на барьере, комиссия мейкера 2 bps. Стоп и таймаут — по цене сделки, тейкер 4 bps. Вход всегда мейкер. На цель круг 4 bps, на стоп и таймаут 6 bps. Старые 20 bps проскальзывания не добавлялись. Окно 24 бара: 24 часа на 1h и 96 часов на 4h. Лента кончается 2026-09-11. 2026 в файл записан и порядок строк не выбирал.

**Результат.** 1 199 231 касание. Исполнено 875 645, пропущено занятым слотом 315 465, лимит не собрал объём в 8 121 случаях. Промах около 1% от попыток: цена почти всегда возвращалась к закрытию. Медиана средней сделки −9.7 bps. Средняя выше нуля у 22 книг из 465. У большинства из этих 22 счёт всё равно в минусе: плюс среднего меньше размера стопа. Sharpe нигде не дотягивает до 0.1, лучший 0.078.

Часовой EMA50, k = 3, отскок, X = 50, рядом с EXP-125. Лимит картину не починил:

| | исполнено | средняя | счёт | просадка | 2026, средняя |
|--|--:|--:|--:|--:|--:|
| DOGE | 741 | −8.7 | −56% | 67% | −21.2 (140) |
| XRP | 712 | −15.2 | −71% | 72% | −16.8 (145) |
| LTC | 679 | −25.1 | −84% | 87% | −9.1 (151) |
| BTC | 768 | −0.4 | −10% | 35% | −14.3 (155) |
| ETH | 713 | −18.4 | −76% | 77% | −7.0 (145) |

Часовой swing high, k = 3, отскок, X = 50, тоже в минусе на всех пяти: −7.1, −4.7, −15.0, −9.3, −9.1 bps.

Единственный счёт, который заметно вырос, — собака, 4h swing high, k = 3, отскок. Средняя +19.0 / +21.7 / +16.6 / +23.4 bps при X = 5 / 10 / 20 / 50, сделок 407–477, счёт +71%…+131%, просадка 32–46%, Sharpe 0.056–0.078. Та же семья на остальных четырёх монетах около нуля или в минусе. 2026 у этой собаки тоже в плюсе (+9…+16 bps) и правило не выбирал: у рипла, лайткоина, биткоина и эфира 2026 этой семьи в минусе.

**В прод не писать.** Лимит на закрытии исполняется, плюс часовых книг не возвращается. Одна монета на 4h с Sharpe ниже 0.1 — не книга.

**Артефакты:** `causal_flip/exp126_lenta_tick_limits.py`, `tests/test_exp126_lenta_tick_limits.py`; `causal_flip/reports/exp126_lenta_tick_limits/prereg.json`, `limit_results.csv`, `summary.json`, `width_geometry.csv`.

**Ширина ленты.** Полоса — это уровень ± k×ATR. Цифра ниже — медиана расстояния от уровня до края, в bps. Полная ширина от края до края вдвое больше. Считалось по тем же касаниям, что и лимитный прогон, без нового отбора.

Час, k = 3: собака 404, рипл 309, лайткоин 304, эфир 274, биткоин 197. На 4h у swing high k = 3 край ещё дальше: собака 697, рипл 569, лайткоин 546, эфир 511, биткоин 375. Узкая лента k = 0.25 на часе — около 21 bps до края, полная ширина около 40 bps.

Отскок k = 3, X = 50 входит у края. До цели от лимита медиана 73–99 bps на часе. До стопа 216–386 bps, почти вся полуширина. Чтобы это окупалось, нужно около 80% выигрышей. На тиках у часового EMA50 было около 70%. У биткоина лента самая узкая из пяти, поэтому его средняя около нуля, а не −20.

## EXP-123 — Optuna, MLP и GRU (2026-10-06)

**Вопрос.** Есть ли на 2025 фильтр касаний лучше LightGBM из EXP-122, если учить MLP и GRU на тех же 30 признаках.

**Метод.** Зафиксировано в `causal_flip/reports/exp123_lenta_transformer_flip/prereg.json`. Метка: 1, если барьерный pnl после 24 bps больше нуля. Активы DOGE, XRP, LTC, только 1h. Обучение до 2024-12-31, проверка 2025, тест 2026 в счёт не входил: ни в score, ни в раннюю остановку, ни в порог. Score — средняя EV 2025 по активам, у которых n ≥ 100 и Sharpe > 0.1; таких активов нужно минимум два, иначе −1e9. Порог из 0.5 / 0.6 / 0.7 по сводному 2025. 20 испытаний, по 3 сида, в зачёт идёт лучший сид испытания. Transformer открывается только если ни MLP, ни GRU не ближе 2 bps к лучшему и Sharpe не хуже. Чемпион на 100 сидов и чтение 2026 не запускались.

**Результат.** 20 из 20 испытаний завершились. Лучший — испытание 14, MLP, hidden 128, 3 слоя, dropout 0.4, lr 5.75e-5, batch 128. Score **39.9 bps**, средний Sharpe 0.32. Порог 0.7. Лучший сид 1141. Три сида этого испытания: 38.1, 39.9, 39.0. На 2025, перекрывающиеся касания, после 24 bps:

| | n | EV | Sharpe |
|--|--:|--:|--:|
| DOGE | 2599 | +42.1 | 0.35 |
| XRP | 2155 | +38.5 | 0.26 |
| LTC | 2450 | +39.1 | 0.34 |

Лучший GRU — испытание 0, score 23.4, Sharpe 0.16. MLP впереди больше чем на 2 bps, transformer не открыт. В `summary.json` вердикт остаётся `PENDING_TEST`: 2026 эта модель не читала.

**Это не тики и не один депозит.** Score — средняя барьерная сделка 2025, те же перекрывающиеся касания, что в EXP-122 и EXP-124. Часовая свеча записывала край часа. EXP-125 и EXP-126 на правилах ленты этот край не подтвердили: на тиках выигрыш закрывается у барьера, а стоп платит ширину ленты. Сеть на тиках и на одном ордере не прогонялась. Чемпиона и transformer из этого не запускать, пока нет отдельного решения.

**В прод не писать.**

**Артефакты:** `causal_flip/exp123_lenta_transformer_flip.py`, `tests/test_exp123_lenta_transformer_flip.py`; `causal_flip/reports/exp123_lenta_transformer_flip/summary.json`, `prereg.json`, `PLAN.md`; `causal_flip/runs/optuna_lenta/study.db`.

## EXP-127 — тот же MLP на тиках (2026-10-06)

**Вопрос.** Останется ли 39.9 bps испытания 14, если исполнить отобранные касания лимитом по ленте.

**Метод.** Зафиксировано в `causal_flip/reports/exp127_lenta_mlp_tick/prereg.json` до счёта. Заново обучен тот же MLP: hidden 128, 3 слоя, dropout 0.4, batch 128, сид 1141. Порог 0.7 не перевыбирался. Веса сохранены. В тики идут только касания 2025 и 2026 с p > 0.7. Вход — лимит в закрытие свечи, один рабочий ордер на актив, прибыль лимитом на барьере, стоп по сделке. Мейкер 2 bps, тейкер 4 bps. Лишние 20 bps не добавлялись. 2026 модель не принимал.

**Повтор обучения.** Score снова 39.90, порог снова 0.7. Собака 2599 / +42.10, рипл 2155 / +38.50, лайткоин 2450 / +39.11. Это барьерная метка часовой свечи, та же, что в EXP-123.

**Тики, все отобранные касания.** Средняя сделка в bps. Рядом — что было на свече.

| | 2025, свеча | 2025, тик | 2026, тик |
|--|--:|--:|--:|
| DOGE | +42.1 (2599) | −7.5 (2572) | −22.1 (1264) |
| XRP | +38.5 (2155) | −1.4 (2125) | −11.7 (1107) |
| LTC | +39.1 (2450) | −11.7 (2423) | −6.0 (1163) |

Доля плюсовых сделок на тиках около 75%. Модель оставила только отскок, пробоев в отобранном нет. Выигрыш закрывается у барьера, стоп платит ширину ленты. Промах лимита маленький: у собаки на 2025 не исполнились 27 из 2599.

**Один ордер.** 2025: собака −3.4 bps, счёт −35%, просадка 54%. Рипл −2.4, счёт −24%, просадка 50%. Лайткоин −22.1, счёт −81%, просадка 84%. 2026 у собаки и рипла тоже в минусе. У лайткоина на одном ордере в 2026 средняя +6.2 на 359 сделках, счёт +20%, просадка 30%, Sharpe 0.04. Это не отбор: без слота та же книга в 2026 −6.0.

**В прод не писать.** 39.9 на тиках не держатся.

**Артефакты:** `causal_flip/exp127_lenta_mlp_tick.py`, `tests/test_exp127_lenta_mlp_tick.py`; `causal_flip/reports/exp127_lenta_mlp_tick/prereg.json`, `summary.json`, `tick_summary.csv`, `trades.csv`, `mlp_trial14_seed1141.pt`.

## EXP-128 — тот же MLP флипом, без барьера (2026-10-06)

**Вопрос.** Если барьерный стоп платит ширину ленты, поможет ли другая полоса или удержание до переворота.

**Другая полоса уже есть в EXP-126.** k = 0.25 и 0.5, все таймфреймы и стороны, 240 книг, те же лимиты и один ордер. Средняя выше нуля ни у одной. Медиана около −10 bps. Лучшая, эфир 4h swing low k = 0.25 пробой X = 50, −1.8 bps. Узкий стоп скальп не спасает.

**Флип.** Зафиксировано в `causal_flip/reports/exp128_lenta_mlp_flip/prereg.json`. Те же веса EXP-127, порог 0.7, без дообучения. Одна позиция на актив. Знак — сторона последнего отобранного касания. Тот же знак ничего не делает. Противоположный закрывает и переворачивает. Цена — первая aggTrade на закрытии свечи сигнала. Тейкер 4 bps на вход и 4 bps на выход. Барьера нет. Незакрытый хвост в конце ленты в сделки не входит. 2026 книгу не принимал.

**Результат.** Это уже один депозит, не сумма касаний. Медиана удержания 5–7 часов.

| | 2025, сделок | средняя | счёт | просадка | 2026, средняя | 2026, счёт |
|--|--:|--:|--:|--:|--:|--:|
| DOGE | 829 | +4.3 | −4% | 45% | +0.1 | −13% |
| XRP | 721 | +21.7 | +266% | 38% | −17.0 | −55% |
| LTC | 739 | −21.7 | −85% | 90% | +12.9 | +44% |

У рипла в 2025 средняя и медиана рядом, около +22 bps, Sharpe 0.08. В 2026 медиана ещё +15, а средняя −17: хвост убытков толще. Лайткоин в 2025 наоборот. Собака около нуля. Три актива в одну сторону не встают, 2026 рипл не подтверждает.

**В прод не писать.**

**Артефакты:** `causal_flip/exp128_lenta_mlp_flip.py`, `tests/test_exp128_lenta_mlp_flip.py`; `causal_flip/reports/exp128_lenta_mlp_flip/prereg.json`, `summary.json`, `flip_summary.csv`, `trades.csv`.

## EXP-129 — полоса как накопление, вилка на пробой (2026-10-06)

**Вопрос.** Если полоса — место аккумуляции, а не цель в X пунктов, живёт ли пробой: вилка на обоих краях.

**Метод.** Зафиксировано в `causal_flip/reports/exp129_lenta_fork/prereg.json` до счёта. Те же ленты EXP-124, 28 полос на пять пар, 140 книг. Сторона и X больше не выбираются. Полоса замораживается в момент входа цены внутрь. Бай-стоп на верхнем крае, селл-стоп на нижнем. Кто пробит первым, тот и вход, второй снимается. Стоп — противоположный край. Цель — ещё одна высота полосы за пробитым краем. До комиссий риск и прибыль равны высоте полосы. Пробой набирает 9310 USDT и исполняется по принту, тейкер 4 bps. Цель — лимит мейкера 2 bps. Стоп — первая сделка за дальним краем, тейкер 4 bps. На пробой 24 бара, потом ещё 24 бара на цель или стоп. Пока вилка стоит или позиция открыта, следующее касание этой книги пропускается. 2026 файл не сортировал и книгу не выбирал.

**Результат.** Это один депозит на книгу. Исполнено 160 429, пропуск занятым слотом 178 987, без пробоя 2 427. Медиана доли выигрышей 48%, медиана средней сделки **−8.4 bps**. Около половины побед на симметричной вилке не покрывает тейкер.

Средняя выше нуля у 16 книг из 140. Крупные плюсы сидят на одной монете. Собака, 4h swing low, k = 3: +106 bps, 157 сделок, счёт +168%, просадка 51%, Sharpe 0.11. Та же вилка на остальных четырёх в минусе или около нуля. Рипл, 4h swing high, k = 3: +95 bps, счёт +168%, просадка 44%. На битке та же вилка −101 bps, на эфире −84, на лайткоине −65, на собаке −19. У обеих «больших» книг 2026 в минусе, около −25 bps на 30 сделках.

Биткоин и эфир, 4h swing low, k = 2, оба скромно в плюсе: +24 и +23 bps, счета +39% и +17%, просадка 36–37%, Sharpe 0.06 и 0.04. Собака на этой вилке −7, лайткоин −88, рипл +2 при счёте −33%.

**В прод не писать.** Вилка выровняла стоп и цель, доля побед села около половины, после тейкера типичная книга в минусе. Плюс, который остался, не повторяется на второй монете.

**Артефакты:** `causal_flip/exp129_lenta_fork.py`, `tests/test_exp129_lenta_fork.py`; `causal_flip/reports/exp129_lenta_fork/prereg.json`, `summary.json`, `fork_results.csv`.

## Линия лент закрыта (2026-10-06)

**Решение.** После графика цены с лентой линия закрыта. Нового прогона нет.

На графике 4h с 2023-07-01 по конец кэша, около 7100 баров, и час за последние 120 дней. Центр — EMA, края — k×ATR(14). Это широкая полоса вокруг средней: цена большую часть времени сидит внутри, выход за край ничем не отмечен. Картинка: `<local canvas file>`.

Тики это уже посчитали. Отскок на одном слоте и лимитом в минусе (EXP-125, EXP-126, EXP-127). Узкая полоса k = 0.25 и 0.5 тоже в минусе, 0 из 240 (EXP-126, записано в EXP-128). Флип без барьера не держится на трёх монетах (EXP-128). Вилка на пробой — около половины побед и медиана −8.4 bps (EXP-129). Плюсы на свечах, включая 39.9 у MLP и Calmar EXP-124, были краем часовой свечи и сложением перекрывающихся касаний. На один депозит и на тик они не переносятся.

**Дальше по ленте не идти.** Transformer и чемпиона на 100 сидов из EXP-123 не открывать. Вердикт в `summary.json` остаётся `PENDING_TEST`. Живой бот DOGE FLIP этой линией не трогать.

## EXP-130 — уровни DOGE 4h, отскок и стоп (2026-10-07) · подумать

**Тег: подумать.** Это не лента EXP-116–129 и не живой бот. В прод не писать. 2026 порог, стоп и правило не выбирал.

**Разметка.** Горизонтальные уровни, без наклонных коридоров и без заливки. Свинг 8 баров влево и 8 вправо, цена — экстремум. На картинке линия на всю ширину окна, ближайшие слиты, если диаметр кучи не больше 1.5 ATR, цена линии — медиана. Для модели и книги уровень живёт иначе: появляется после подтверждения и умирает, когда закрытие уходит за дальний край. Ширина полосы 0.5 ATR в каждую сторону, ближний край до дальнего — 1 ATR. Новый свинг той же стороны в пределах max(своя полоса, 1 ATR) тот же уровень и цену не двигает. Картинка: `<local canvas file>`.

**Метка.** Касание: прошлое закрытие снаружи ближнего края, текущее внутри. Отскок — за 24 бара 4h закрытие возвращается за ближний край раньше, чем закрытие уходит за дальний. Иначе пробой. Таймаут выброшен. Доля отскока: обучение 64.3% (810), 2025 — 57.7% (130), 2026 — 62.4% (109).

**LightGBM.** Те же 129 признаков собаки на часе, известные за час до открытия 4h бара, плюс approach_up, dist_atr, age_bars, band_atr. 500 деревьев, lr 0.02, 31 лист, min_child_samples 50, сид 42. Порог из 0.50–0.70 по винрейту 2025, n ≥ 40, выше доли отскока 2025. Выбран 0.65: на 2025 винрейт 58.3% при базе 57.7%, из них 4 вызова пробоя. Точность отскока на этом пороге 57.5%, ниже базы. 2026 при 0.65: винрейт 58.3% против всегда-отскока 62.4%. Модель базу не обошла. ATR, волатильность, RSI, импульс, возраст и расстояние до EMA50/200 долю отскока с ~64% до ~80% не поднимают.

**Чужая книга, не логика сделки.** Вход в закрытие касания, выход в закрытие разрешения, один слот. При 8 bps обучение −91%, 2025 −36%, 2026 −12%. Это не книга со страховкой. В зачёт стратегии не ставить.

**Книга по логике.** Один депозит, DOGE 4h, всегда отскок. Лимит переворота на ближнем крае, мейкер 0, гэп — тейкер 4. Страховка на дальнем крае, тейкер 4. Комиссия DOGEUSDC этого ключа: мейкер 0, тейкер 4 bps. Старые эксперименты на своих 24 bps не переписывать.

Трейлинг 0.5 ATR книгу убил. 5361 исполнение: 2468 переворотов, 677 страховок, 2216 трейлингов. Средняя закрытая нога −13.7 bps. К 14 августа 2022 от 1.0 осталось около 0.01, к 2026 — пыль. Файл `bounce_book.json` этот прогон уже не хранит: его перезаписал фиксированный стоп.

**Фиксированный стоп.** Трейлинг снят. После выхода цены из полосы стоп стоит на цене переворота и не двигается. После страховки у новой стороны один стоп на 0.5 ATR, тоже неподвижный. 5281 исполнение: 2574 переворота, 2036 стопов, 671 страховка. До следующего уровня дошли 537 ног, медиана +414 bps. Стоп: медиана −4 bps, это комиссия. Страховка: медиана −267 bps. Среднее по закрытым ногам +4.6 bps. Депозит: 14 августа 2022 — 0.021, начало 2025 — 0.012, начало 2026 — 0.0075, конец — 0.009. Обучение −98.7%, просадка −99.8%. 2025 на остатке −39%, просадка −63%. Плюс 2026 — движение пыли.

**Сетка стопа**, расстояние от входа в ATR уровня. Дальний край — 1 ATR. 2026 расстояние не выбирал. К началу 2025 от доллара:

| Стоп | К августу 2022 | К началу 2025 | На конце |
|---|---:|---:|---:|
| 0, на входе | 0.021 | 0.012 | 0.009 |
| 0.25 | 0.000 | 0.000 | 0.000 |
| 0.5 | 0.000 | 0.000 | 0.000 |
| 0.75 | 0.000 | 0.000 | 0.000 |
| 1, на дальнем крае | 0.032 | 0.004 | 0.003 |
| 1.5 | 0.045 | 0.013 | 0.006 |
| 2 | 0.042 | 0.018 | 0.008 |
| 3 | 0.061 | 0.015 | 0.005 |
| 4 | 0.039 | 0.013 | 0.003 |

Стоп внутри полосы режет сделку раньше следующего уровня. Дальше 1 ATR на отскоке первой бьёт страховка. Ни одно расстояние депозит не оставило.

**Индикатор пробоя.** Глубина закрытия в полосе: 0 на ближнем крае, 1 на дальнем. Пробой — закрытие глубже 0.5. Доля пробоев: обучение 52% против базы 36% (164), 2025 — 74% (27), 2026 — 68% (25). Объём, тело свечи, ADX и DI сторону подхода эту долю не отделяют. В книге лимит на ближнем крае остаётся; если бар закрылся за серединой, отскок закрывается по этому закрытию и уровень остывает. Срабатываний 229, страховок 487 вместо 671. Депозит: 0.024 к августу 2022, 0.014 к началу 2025, 0.004 к началу 2026, 0.009 на конце. 2025 на остатке −72%. Правило счёт не спасло.

**Артефакты:** `causal_flip/exp130_doge_level_bounce.py`, `causal_flip/exp130_bounce_book.py`; `causal_flip/reports/exp130_doge_level_bounce/summary.json`, `bounce_book.json`, `stop_sweep.json`.

## EXP-131 — трансформер, три головы (2026-10-07) · подумать

**Тег: подумать.** Отдельный прогон, не EXP-123. Чемпиона на 100 сидов нет. В прод не писать. 2026 не в лоссе, не в ранней остановке и не в пороге.

**Головы.** Направление — лонг, если закрытие через 24 бара 4h выше. Стоп — ход против этой стороны, в ATR, обрезка метки на 8. Ширина — ход в сторону сделки, в тех же единицах. Вход — последовательность 24 часовых векторов собаки до касания, d_model 64, nhead 4, 2 слоя, ff 128, dropout 0.1, сид 42, lr 3e-4. Ранняя остановка по совместному лоссу 2025, терпение 10, не больше 50 эпох. Порог из 0.50–0.70 по депозиту 2025, n ≥ 40. Сделка пропускается, если предсказанная ширина не больше стопа. Вход по закрытию касания, тейкер 4. Стоп тейкер 4. Цель по своей цене мейкер 0, гэп сквозь неё тейкер 4.

**Результат.** Направление хуже монетки: 45.7% на 2025, 44.4% на 2026. Стоп и ширина сидят на средних обучения, около 1.3 ATR и 4.3 ATR. Порог встал на 0.5, потому что строже 40 сделок на 2025 нет. Книга 2025: 76 сделок, депозит 0.46, −54%, стоп 58, цель 10. Книга 2026 при том же пороге: 61 сделка, депозит 0.78, −22%, стоп 43, цель 12. Порог 0.7 сделок не открывает и депозит не двигает.

**Артефакты:** `causal_flip/exp131_level_heads.py`; `causal_flip/reports/exp131_level_heads/summary.json`.

## EXP-132 — forward трёх нулей и ошибка ретроспективного OOS (2026-11-05)

**Связь с EXP-105.** Это отдельная проверка пилота, исходная запись EXP-105 не переписывается. Окно критерия до 11 ноября ещё открыто. Счёт ниже — по 2026-11-05 07:00. Пороги и нули на этом отрезке не выбирались. Живой бот не останавливался.

**Сервер, нули включены.** Сделки из `stats_binance_flip.sqlite`. Лонг, открытый 28 сентября в 06:00 по 0.0933, закрыт 3 октября в 15:00 по 0.0932: на ноге около −0.2%. Дальше 19 кругов, все закрыты в октябре, 13 в плюс. Размер бота 95% с буфером 2%. На этом размере закрытый счёт **+7.0%**, просадка **−5.0%**. На полном номинале было бы +7.4% и −5.3%. С 29 октября 21:00 открыт лонг с 0.07531. Свеча 5 ноября 07:00 закрылась 0.07643: нога около +1.4%, к счёту на размере бота это ещё около +1.3%. Вместе с закрытыми примерно **+8.4%**. Ноябрь закрытых сделок не имеет. Последний тик 5 ноября 08:00 UTC, закрытый час 07:00: вверх 0.5786, сигнал лонг.

| вход | сторона | нога |
|---|---|---:|
| 10-03 15:00 | шорт | +0.05% |
| 10-03 20:00 | лонг | −0.08% |
| 10-03 21:00 | шорт | +6.87% |
| 10-08 15:00 | лонг | −1.10% |
| 10-10 07:00 | шорт | +0.04% |
| 10-10 08:00 | лонг | +0.07% |
| 10-10 09:00 | шорт | +1.62% |
| 10-12 18:00 | лонг | −3.00% |
| 10-16 09:00 | шорт | +0.67% |
| 10-16 13:00 | лонг | −3.04% |
| 10-18 02:00 | шорт | +0.21% |
| 10-19 12:00 | лонг | +0.86% |
| 10-20 04:00 | шорт | +0.96% |
| 10-20 11:00 | лонг | +5.48% |
| 10-21 16:00 | шорт | +0.03% |
| 10-21 18:00 | лонг | −1.38% |
| 10-21 23:00 | шорт | +1.26% |
| 10-22 02:00 | лонг | −4.48% |
| 10-25 00:00 | шорт | +2.80% |

**Ошибочный барный пересчёт, не использовать.** Те же веса, пороги 0.48/0.52, динамическая температура, вход на открытии следующего часа, комиссия 4 б.п. на сторону, капитал целиком. Окно 2026-09-30 02:00 … 2026-11-05 07:00, 870 часов. Пересчёт дал:

| | без нулей | три нуля |
|---|---:|---:|
| счёт | **−0.2%**, MDD −11.2%, 20 сделок, 10 в плюс | **−6.0%**, MDD −18.2%, 20 сделок, 10 в плюс |
| 30 сентября | +0.7% | +0.7% |
| октябрь | −2.0% | −7.7% |
| 1–5 ноября | +1.1% | +1.1% |
| открытый лонг | +0.3% | +1.3% |

**Ошибка.** Пересчёт строил прошлые часы из сегодняшних целиком обновлённых кэшей SPY/QQQ/VIX и Fear & Greed. Живой бот видел эти дневные источники с задержкой и обновлял их пачками. Например, до закрытого часа 2026-10-21 14:00 у сервера было 704 строки macro и 1023 Fear & Greed. Перед часом 15:00 кэши обновились до 706 и 1024. В этот же момент крупное расхождение вероятностей исчезло. Если на час 2026-10-21 14:00 обрезать сегодняшние данные до фактически доступных тогда размеров, локальная вероятность вверх меняется с ошибочных 0.4699 на **0.6225** при серверных **0.6215**. На 2026-10-20 14:00: с 0.4518 на **0.5222** при серверных **0.5220**. На 2026-10-20 22:00: с 0.4283 на **0.5098** при серверных **0.5097**.

Значит, −6.0% с нулями и −0.2% без нулей — **не OOS и не сравнение вариантов**, а пересчёт с данными, которых в те часы ещё не было. Обе цифры аннулированы. Реальный серверный forward с тремя нулями остаётся около **+7.0% по закрытым кругам**, либо около **+8.4% с открытым лонгом по закрытию 5 ноября 07:00.** Чистого серверного counterfactual без нулей нет. Для него нужны сохранённые as-of снимки внешних кэшей или почасовые векторы фич.

Дополнительно исправлена диагностическая проверка сигнала: она ошибочно требовала `prob_up >= prob_down` даже когда только long-порог был пройден. Живой код сравнивает стороны только когда прошли оба порога. На P&L ошибочного барного пересчёта это не влияло: его симулятор использовал правильное правило.

**Артефакты:** `causal_flip/exp132_zero_oos.py`; `causal_flip/reports/exp132_zero_oos/summary.json`, `server_trades.txt`.

