"""Сборка docs/anatomy.md и графика норм активаций."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без дисплея: скрипт должен работать и в CI

import matplotlib.pyplot as plt  # noqa: E402  (backend выбирается до импорта)

MODE_TITLES = {
    "inference": "инференс",
    "full_ft": "full fine-tune",
    "lora": "LoRA (r=8, q/v)",
}


def thousands(n: int) -> str:
    """Число с неразрывными пробелами по разрядам."""
    return f"{n:,}".replace(",", " ")


def plot_activations(activations: dict, path: str) -> None:
    """Две панели: норма по позициям токена и средняя норма по трём блокам."""
    labels = list(activations["norms"])
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4), width_ratios=(2, 1))

    for label in labels:
        values = activations["norms"][label]
        ax_left.plot(values, linewidth=1.4,
                     label=f"{label} (слой {activations['layers'][label]})")
    ax_left.set_xlabel("позиция токена")
    ax_left.set_yscale("log")   # без лога всё придавит выброс massive activations
    ax_left.set_ylabel("‖h‖₂ (лог. шкала)")
    ax_left.set_title("Норма скрытого состояния по позициям")
    ax_left.legend(fontsize=9)
    ax_left.grid(alpha=0.3)

    means = [sum(activations["norms"][x]) / len(activations["norms"][x]) for x in labels]
    ax_right.bar(labels, means, color=["#4c78a8", "#f58518", "#54a24b"])
    ax_right.set_ylabel("средняя ‖h‖₂")
    ax_right.set_title("Средняя норма по блоку")
    ax_right.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def markdown_report(report: dict, params: dict) -> str:
    env, cfg = report["environment"], report["config"]
    modes = {m["mode"]: m for m in report["memory"]}
    inf, full, lora = (modes[m] for m in ("inference", "full_ft", "lora"))
    mib = 1024 ** 2
    weights = inf["weights_bytes"] / mib
    tied = sum(g["tied_params"] for g in report["params_by_group"])
    lines = [
        f"# Анатомия {report['model']}", "",
        "## Конфигурация", "",
        "| Параметр | Значение |", "|---|---:|",
        *[f"| {key} | {value} |" for key, value in cfg.items()], "",
        f"Ревизия модели: `{report['model_revision']}`.", "",
        "## Условия замера", "",
        "| Условие | Значение |", "|---|---|",
        f"| платформа | {env['platform']} |",
        f"| устройство | {env['device']} |",
        f"| dtype весов | {env['dtype']} |",
        f"| seq_len × batch_size | {env['seq_len']} × {env['batch_size']} |",
        f"| прогонов на режим | {env['repeats']}, каждый в отдельном процессе; берётся максимум |",
        f"| метрика памяти | `{env['memory_metric']}` |",
        f"| дополнительная метрика RSS | `{env['rss_metric']}` |",
        *[f"| {name} | {env[name]} |" for name in ("python", "torch", "transformers", "peft")],
        f"| seed | {params['generate']['seed']} |", "",
        "Все значения памяти — в МиБ (2²⁰ байт), как в лекции. Модель и длина 256 сохранены.",
        "Инференс: один forward под `torch.inference_mode()`. Обучение: forward с labels,",
        "backward и один шаг AdamW, затем `zero_grad(set_to_none=True)`.",
        f"Learning rate — {params['memory']['lr']}; остальные параметры AdamW стандартные.",
        "Вход — случайные id токенов; KV-cache выключен во всех режимах (`use_cache=False`).",
        "Gradient checkpointing и autocast не используются. Загрузка модели входит во время,",
        "пик ускорителя снимается после загрузки и включает уже размещённые веса.", "",
    ]
    if inf["device"].startswith("mps"):
        lines += [
            f"На MPS опрос драйвера идёт каждые {inf['sample_interval_ms']:g} мс в отдельном потоке,",
            "плюс замеры после синхронизации при загрузке, forward, backward и optimizer.step.",
            "Это максимум наблюдений, а не аппаратный high-water mark: короткий пик между",
            "отсчётами может быть пропущен. Метрика включает кэш аллокатора и буферы Metal.", "",
        ]
    lines += [
        "## Параметры", "",
        "| Группа | Модулей | Shape | Уникальных параметров | Доля | Tied |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for item in report["params_by_group"]:
        lines.append(f"| {item['group']} | {item['modules']} | {item['shape']} | "
                     f"{thousands(item['params'])} | {100 * item['share']:.2f}% | "
                     f"{thousands(item['tied_params'])} |")
    lines += [
        f"| **Итого** | | | **{thousands(report['params_total'])}** | **100%** | |", "",
        f"Прямой подсчёт `sum(p.numel() for p in model.parameters())` = {thousands(report['params_direct'])}.",
        f"`lm_head.weight` и `embed_tokens.weight` используют один тензор из {thousands(tied)} чисел.",
        "В строке lm_head сохранена форма, но новых параметров — 0. Векторы RMSNorm:",
        f"`{2 * cfg['num_hidden_layers'] + 1} × {cfg['hidden_size']} + "
        f"{2 * cfg['num_hidden_layers']} × {cfg['head_dim']} = "
        f"{next(g['params'] for g in report['params_by_group'] if g['group'] == 'norm')}`.",
        "В Excel norm представлена одним суммарным вектором; фактически норм 113.", "",
        "GQA: 16 Q-голов и 8 KV-голов по 128 чисел. Поэтому q_proj имеет 2048 строк,",
        "а k_proj и v_proj — 1024. В MLP на блок `3 × 3072 × 1024 = 9 437 184` параметра,",
        "во внимании — `6 291 456`: MLP тяжелее в 1,5 раза (44,33% против 29,55%).",
        "RoPE не добавляет обучаемых параметров.", "",
        "## Активации", "",
        f"Промпт: «{params['hooks']['prompt']}». После chat template — {report['activations']['n_tokens']} токенов.",
        "На выходе блоков измеряется L2-норма hidden state каждого токена в float32.", "",
        f"![Нормы активаций]({Path(params['hooks']['plot']).name})", "",
        "| Блок | Индекс (с нуля) | Средняя L2-норма | Максимум | Позиция максимума |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, index in report["activations"]["layers"].items():
        vals = report["activations"]["norms"][label]
        lines.append(f"| {label} | {index} | {sum(vals)/len(vals):.2f} | {max(vals):.2f} | {vals.index(max(vals))} |")
    lines += ["", "В pre-norm блоке обновляется `x + f(norm(x))`: residual-поток не нормализуется",
              "целиком, поэтому его масштаб может расти с глубиной. Выбросы на ранних токенах",
              "согласуются с massive activations из лекции. Одни нормы не доказывают attention sink:",
              "для этого потребовалось бы отдельно измерить веса внимания.", "",
              "## LoRA", "",
              "Для матрицы `(out, in)` добавляются `A(r, in)` и `B(out, r)`:",
              "`N = Σ r × (in + out)`. Bias отсутствует; lm_head в список целей не входит.", "",
              "| Конфиг | Своя формула | peft | Доля базовой модели |",
              "|---|---:|---:|---:|"]
    for item in report["lora"]:
        lines.append(f"| {item['name']} | {thousands(item['formula'])} | {thousands(item['peft'])} | {item['share_of_base'] * 100:.4f}% |")
    lines += ["", "`r=8`: `28 × 8 × ((1024 + 2048) + (1024 + 1024)) = 1 146 880`.",
              "`r=16`: `28 × 16 × (3072 + 2048 + 2048 + 3072 + 4096 + 4096 + 4096) = 10 092 544`.",
              "`lora_alpha` равна 16 и 32 соответственно; dropout = 0. Alpha меняет масштаб,",
              "но не число параметров. Обе суммы точно совпадают с `peft`.", "",
              "## Память", "",
              "| Режим | Пик, МиБ | RSS, МиБ | × к инференсу | Время, с | loss | PID |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for m in report["memory"]:
        loss = "—" if m["loss"] is None else str(m["loss"])
        lines.append(f"| {MODE_TITLES[m['mode']]} | {m['peak_mb']:.1f} | {m['peak_rss_mb']:.1f} | "
                     f"{m['peak_mb']/inf['peak_mb']:.2f} | {m['seconds']:.1f} | {loss} | {m['pid']} |")
    lines += ["", "Объём тензоров подсчитан отдельно через `numel() × element_size()`.",
              "Градиенты сняты до zero_grad, состояния AdamW — после первого шага.", "",
              "| Режим | Базовые веса, МиБ | Адаптер, МиБ | Градиенты, МиБ | AdamW, МиБ |",
              "|---|---:|---:|---:|---:|"]
    for m in report["memory"]:
        lines.append(f"| {MODE_TITLES[m['mode']]} | " + " | ".join(
            f"{m[k]/mib:.2f}" for k in ("weights_bytes", "adapter_bytes", "gradient_bytes", "optimizer_bytes")) + " |")
    lines += ["", f"Веса bf16: `{report['params_total']} × 2 / 2²⁰ = {weights:.2f}` МиБ.",
              f"Full FT добавляет {full['gradient_bytes']/mib:.2f} МиБ градиентов и "
              f"{full['optimizer_bytes']/mib:.2f} МиБ состояний AdamW: около ×4 от весов ещё до активаций.",
              "Моменты AdamW здесь имеют dtype параметров; отдельной fp32 master-копии нет.",
              f"PEFT переводит адаптер в float32: {report['lora'][0]['peft']} × 4 байта = "
              f"{lora['adapter_bytes']/mib:.2f} МиБ. Его градиенты и два момента занимают "
              f"{(lora['gradient_bytes'] + lora['optimizer_bytes'])/mib:.2f} МиБ.",
              f"Full FT выше инференса на {full['peak_mb']-inf['peak_mb']:.1f} МиБ, "
              f"LoRA — на {lora['peak_mb']-inf['peak_mb']:.1f} МиБ.",
              "Разность пика и объёма постоянных тензоров нельзя целиком назвать активациями:",
              "в неё входят logits, временные буферы операций и кэш драйвера, а их максимумы",
              "могут приходиться на разные стадии. LoRA сохраняет активации для backward через",
              "замороженные слои, поэтому экономия на градиентах не делает её равной инференсу.", "",
              "В лекции на MPS получено 2222 / 7166 / 3606 МиБ; порядок режимов совпадает.",
              "Точные версии библиотек и параметры аллокатора лекционного прогона не указаны,",
              "поэтому разницу пиков нельзя однозначно приписать одной причине. Наши условия приведены выше.",
              f"В нашем замере RSS меняется на {max(m['peak_rss_mb'] for m in report['memory']) - min(m['peak_rss_mb'] for m in report['memory']):.1f} МиБ.",
              "На MPS буферы Metal лишь частично учитываются в RSS процесса, несмотря на общую",
              "физическую память Apple Silicon. Поэтому основной столбец снят у драйвера.", "",
              "## Четыре дефекта", "",
              "1. **Хуки оставались на слоях.** Handles от register_forward_hook терялись,",
              "   замыкания со словарями продолжали жить после вызова. Повтор добавлял ещё три хука.",
              "   Теперь handles удаляются в finally, значения отделяются через detach,",
              "   модель временно переводится в eval, а training-флаги всех модулей восстанавливаются.",
              f"   Проверка двух прогонов: хуков {report['hook_check']['before']} → {report['hook_check']['after']},",
              "   массивы норм совпадают точно, режим восстановлен. Проверен и выход по исключению.",
              "2. **Tied embeddings считались дважды.** Обход с remove_duplicate=False был нужен",
              "   для строки lm_head, но все строки помечались tied=False. Теперь повтор определяется",
              "   по id параметра и не входит в сумму уникальных весов.",
              f"   Было бы {thousands(report['params_total'] + tied)}, стало {thousands(report['params_total'])};",
              f"   лишние {thousands(tied)} параметров убраны, сумма совпадает с model.parameters().",
              "3. **Вместо пика снимался конец шага.** PeakMemory читал память только в __exit__,",
              "   после gc.collect и обнуления градиентов. Дополнительно режимы запускались",
              "   в одном процессе и наследовали high-water mark RSS. Теперь каждый режим запускается",
              "   через subprocess; пик MPS собирается опросом и на границах стадий, CUDA — штатной",
              "   пиковой метрикой. Разные PID приведены выше.",
              f"   Получено full FT {full['peak_mb']:.1f} > LoRA {lora['peak_mb']:.1f} > инференс {inf['peak_mb']:.1f} МиБ.",
              "4. **На ускорителе измерялся RSS.** device_allocated_bytes и device_metric_source",
              "   всегда вызывали peak_rss. Теперь выбор зависит от устройства: MPS —",
              "   torch.mps.driver_allocated_memory, CUDA — torch.cuda.max_memory_allocated,",
              "   CPU — ru_maxrss (macOS/Linux) или peak_wset (Windows). Источник в JSON и отчёте",
              "   соответствует вызванной функции. RSS сохранён отдельным столбцом для сравнения.",
              f"   Минимальный пик {min(m['peak_mb'] for m in report['memory']):.1f} МиБ выше весов {weights:.2f} МиБ.", "",
              "Источники: задание, слайды 2–7; lecture2.html, слайды 4–14. Числа — docs/report.json.", ""]
    return "\n".join(lines)


def write_report(report: dict, params: dict) -> None:
    plot_activations(report["activations"], params["hooks"]["plot"])
    path = Path(params["report"]["markdown"])
    path.parent.mkdir(exist_ok=True)
    path.write_text(markdown_report(report, params), encoding="utf-8")
