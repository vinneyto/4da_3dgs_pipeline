# Окружение и проверки

## Требования

Полная установка через `setup_environment.sh` рассчитана на **Linux x86_64**. Для inference и Splatfacto нужна NVIDIA GPU с рабочим драйвером и доступом к CUDA. Подготовить JSON и собрать документацию можно без GPU; это не проверка готовности реконструкции.

| Ресурс | Требование |
| --- | --- |
| Система установки | Linux x86_64, Bash, системный Python 3, Git; curl для установки Conda |
| Python пакета | `>=3.11`; installer по умолчанию выбирает 3.11 |
| GPU | PyTorch CUDA import, `torch.cuda.is_available()` и выделение тензора проходят в обоих вычислительных env |
| Компиляция gsplat | CUDA toolkit / `nvcc`, C++ compiler, ninja; одних CUDA runtime wheel недостаточно |
| Headless Nerfstudio | Open3D, EGL/OpenGL/libusb внутри Conda prefix |
| 4DAnyone | Pinned repository revision, инициализированный GVHMR submodule, dependencies и model assets |
| Workspace | Доступные для записи `input`, `models`, `runs`, `jobs`, `environment` |
| AWS | Credentials/role SDK, доступ к выбранному S3 bucket/prefix; дополнительные сервисы согласно конфигу |
| Документация | Node.js 22+, npm; отдельные от CUDA зависимости |

Код не задаёт универсального минимума VRAM, RAM или диска. Расход зависит от видео, числа ракурсов, экспортируемых кадров и размера Gaussian-сцены. Checker проверяет доступность GPU, а не достаточность памяти для конкретного эксперимента.

## Изолированные инструменты

| Окружение | Что выполняет | Defaults installer |
| --- | --- | --- |
| `RECON_CONDA_ENV` | 4DAnyone, AWS worker, dataset exports и dataset Rerun | Python 3.11, torch 2.8.0, torchvision 0.23.0, cu126 |
| Prefix над `RECON_NERFSTUDIO_BIN` | RGBA preparation, `ns-train`, `ns-export` | torch 2.2.2, torchvision 0.17.2, cu121, Nerfstudio 1.1.5, gsplat 1.4.0, toolkit 12.1.1 |
| `RECON_SPLATFACTO_RERUN_PYTHON` | CPU PLY import и Rerun обученной сцены | rerun-sdk 0.36.x |

Это значения [installer в коде](../reference/environment.md), а не рекомендации обновить всё до последней версии. Разные torch и Rerun зависимости разделены по окружениям.

## Сначала проверить существующую машину

Из корня checkout:

```bash
./scripts/check_environment.sh --require-cuda
./scripts/check_environment.sh --json --require-cuda > environment-check.json
```

Checker использует системный Python и запускает probes в настроенных env. Он не читает run JSON, не обращается к AWS, не устанавливает пакеты, не создаёт workspace и не компилирует gsplat. Отчёт перечисляет `OK`, `WARN`, `FAIL`; exit code `1` означает хотя бы один `FAIL`.

Без `--require-cuda` отсутствие GPU становится предупреждением. Сломанный import PyTorch всё равно ошибка. Проверяются все настроенные инструменты, даже если конкретный запуск отключил соответствующую стадию.

Проверки включают переменные `RECON_*`, абсолютность путей, checkout/revision 4DAnyone, GVHMR, каталоги, lock-файл, версии пакетов, CLI executable, модельные файлы, CUDA compiler/version и C++ compiler. Open3D действительно импортируется и создаёт пустой point cloud: это обнаруживает отсутствующую `libEGL.so.1`.

## Полная установка

Сначала замените bucket и SageMaker placeholders в копии примера:

```bash
cp config/run.example.json config/run.json
# Отредактируйте config/run.json перед установкой весов.
./scripts/setup_environment.sh --config config/run.json
source "$HOME/.config/recon-pipeline/environment.sh"
./scripts/check_environment.sh --require-cuda
```

Installer конфигурирует env, при необходимости устанавливает Miniforge, создаёт workspace, устанавливает 4DAnyone/GVHMR, Splatfacto и Rerun. С `--config` дополнительно запускается скачивание весов. Лицензированный архив SMPL-X должен заранее находиться в выбранном bucket; installer не получает лицензию вместо пользователя.

Без `--config` инструменты устанавливаются отдельно от весов; завершающий checker может завершиться ошибкой из-за отсутствующих assets:

```bash
./scripts/download_4danyone_models.sh config/run.json
```

## Переиспользовать рабочий 4DAnyone

Если inference, GPU и модельные assets уже работают:

```bash
./scripts/setup_environment.sh --reuse-4danyone --log-file environment-setup.log
source "$HOME/.config/recon-pipeline/environment.sh"
./scripts/check_environment.sh --require-cuda
```

Этот режим пропускает checkout/requirements/PyTorch 4DAnyone, но устанавливает worker extras `[aws,rerun]`, сохраняет lock и устанавливает остальные инструменты. Существующие Conda prefixes переиспользуются.

## Только сохранить настройки

```bash
./scripts/setup_environment.sh --configure-only
source "$HOME/.config/recon-pipeline/environment.sh"
```

Configure-only не устанавливает инструменты, не создаёт workspace и не обнаруживает старые custom paths автоматически. Сначала экспортируйте свои пути, затем выполните configure. Приоритет: текущие exported переменные → сохранённый env file → installer defaults. Runtime loading defaults не имеет: отсутствие переменной выдаёт ошибку.

`--env-file FILE` или `RECON_ENV_FILE` меняет путь к файлу. По умолчанию это `~/.config/recon-pipeline/environment.sh`; source добавляется в Bash startup files, но текущий терминал надо обновить вручную. Сохраняются `PATH`, `CUDA_HOME`, `CXX`, `LD_LIBRARY_PATH`; при наличии Conda CUDA headers добавляется `CPATH`.

## Диагностика установки

Полная установка пишет timestamped `environment-setup-*.log` в checkout. Ошибка показывает стадию, exit status и путь. Configure-only такого лога не создаёт.

```bash
tail -n 100 environment-setup.log
```

Для отсутствующих headless native libraries:

```bash
conda install --prefix "$(dirname "$RECON_NERFSTUDIO_BIN")" \
  -c conda-forge libegl libgl libusb -y
./scripts/check_environment.sh --require-cuda
```

После исправления среды перезапускайте тот же конфиг без `--force`, чтобы сохранить завершённые результаты.

## Отдельно проверить AWS

Проверка окружения и AWS preflight решают разные задачи. Preflight не доказывает, что CUDA inference работает, а checker не доказывает доступность S3.

```bash
recon-aws-preflight --bucket YOUR_BUCKET --region us-east-1 \
  --input-key input/video.MOV --models-prefix models --write-prefix runs
```

Preflight использует STS `GetCallerIdentity`, S3 bucket/object HEAD, LIST моделей и write probe (PUT с попыткой DELETE). Для worker с shutdown также проверяется SageMaker App `JupyterLab` в статусе `InService`. LIST моделей здесь ограничен одним объектом: это проверка доступа, а не полноты assets.

IAM должен разрешать чтение входа/моделей, LIST нужных prefixes, PUT/GET/DELETE артефактов и markers при облачном сохранении. Для дополнительных функций нужны доступ к выбранной CloudWatch группе (`DescribeLogStreams`, `CreateLogStream`, `PutLogEvents`), SNS и `DescribeApp`/`DeleteApp` SageMaker согласно используемой операции. Worker не создаёт CloudWatch группу автоматически.

Новые требования для splat-transform описаны отдельно в [PR #21](../reference/postprocessing.md).
