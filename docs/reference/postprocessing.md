# Постобработка: подготовлено в PR #21

::: warning Статус реализации
Этот раздел описывает [PR #21](https://github.com/vinneyto/4da_3dgs_pipeline/pull/21), который ещё не включён в документируемый `main`. Поля и команды ниже нельзя считать работающими в текущей основной ветке. После слияния обновите основной config/CLI/API reference вместе с кодом.
:::

## Место в пайплайне

Конвертация — отдельная завершающая стадия: после reconstruction/Rerun и перед итоговым manifest. Исходный full SH PLY остаётся результатом Splatfacto, а производные форматы живут в `postprocessing/`.

```json
{
  "postprocessing": {
    "splat_conversion": {
      "enabled": true,
      "formats": ["compressed_ply", "spz", "sog"]
    }
  }
}
```

Это root fragment рядом с `pipeline`, `artifacts`, `aws_worker`. Секция опциональна и совместима со schema v7: старые документы не добавляют conversion passes.

| Настройка | Назначение |
| --- | --- |
| `enabled` | Включить конвертацию |
| `formats` | Форматы `ply`, `compressed_ply`, `spz`, `sog`; `ply` — копия исходника |

## Результаты и checkpoints

Для каждого кадра предусмотрен отдельный `splat-convert:frame_NNN`, output `postprocessing/splat_conversion/frame_NNN/` и независимый S3 bundle. Итоговый manifest включает `postprocessing.splat_conversion` с картой `exported_artifacts`.

Изменение списка форматов или повтор не требует повторного обучения при валидных reconstruction checkpoints. Отсутствующие/пустые converted outputs инвалидируют conversion checkpoint. Старые неиспользуемые S3 objects могут остаться; актуальные файлы определяет committed map.

## CLI и окружение PR

```bash
recon-splat-convert \
  --input /data/runs/demo/splatfacto/frame_060/exports/splat.ply \
  --output /data/runs/demo/postprocessing/splat_conversion/frame_060 \
  --formats compressed_ply spz sog \
  --splat-transform /opt/splat-transform/bin/splat-transform
```

Нужны Node.js 22+ и pinned `@playcanvas/splat-transform` 3.10.0. PR добавляет `RECON_SPLAT_TRANSFORM_PREFIX`, `RECON_SPLAT_TRANSFORM_VERSION`, установку отдельного Node prefix и проверки executable/node/version в environment checker. Conversion использует `--quiet --gpu cpu`; training GPU env не требуется.

Generator PR добавляет `--splat-conversion`, `--no-splat-conversion`, `--splat-conversion-formats`. Он также уменьшает поток training progress в CloudWatch до изменения целого процента, сохраняя raw rows в `train.log`.

Эксперимент без маски и clipping результатов — отдельные идеи; этот PR не является их реализацией.
