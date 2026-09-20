# 4DAnyone → 3DGS pipeline

Python-обвязка для воспроизводимого запуска [4DAnyone](https://github.com/ant-research/4DAnyone) в AWS SageMaker Studio и экспорта одного синхронизированного момента в формат Nerfstudio/3DGS.

Текущий пайплайн делает следующее:

1. получает одно монокулярное видео;
2. запускает 4DAnyone с заданной схемой виртуальных камер;
3. экспортирует выбранный кадр в `transforms.json`, изображения, маски и point cloud;
4. при необходимости синхронизирует результат в S3;
5. в фоновом режиме сохраняет статус, пишет лог, отправляет SNS email и может остановить JupyterLab App.

> Сейчас это подготовка **статического** датасета 3DGS для выбранного момента времени. Экспорт всех 121 временных кадров и обучение динамического 4DGS в эту версию не входят.

## Стандартная конфигурация камер

Значения по умолчанию повторяют проверенный Colab-прогон:

| Параметр | Значение |
|---|---:|
| `views_per_layer` | 24 |
| `layer_pitches` | `-15, 0, 15` |
| Общее число камер | 72 |
| `start_yaw` | 0° |
| `yaw_span` | 360° |
| `target_fps` | 30 |
| `seed` | 42 |
| модель | turbo |
| экспортируемый кадр | 60 |

`RERUN_VIEW_COUNT=4` из Colab не является параметром inference или экспорта 4DAnyone. Это настройка последующей визуализации и в данный CLI не включена.

## Структура на постоянном диске Space

```text
~/work/4DAnyone/              upstream 4DAnyone
~/work/4da_3dgs_pipeline/     этот репозиторий
~/4danyone-data/
├── input/
├── models/
├── runs/
└── jobs/                     запросы, статусы и логи фоновых задач
```

Все эти каталоги находятся на 50-гигабайтном EBS-томе Space и сохраняются между остановками JupyterLab App. Сам GPU-инстанс оплачивается только пока App имеет статус `InService`/запущен; плата за EBS-хранилище продолжает начисляться и при остановленном App.

## Установка окружения

```bash
cd "$HOME/work"
git clone https://github.com/vinneyto/4da_3dgs_pipeline.git
cd 4da_3dgs_pipeline

chmod +x scripts/*.sh
./scripts/setup_4danyone_env.sh
```

Скрипт создаёт изолированное окружение `$HOME/.conda/envs/4danyone`, ставит совместимую пару CUDA PyTorch/Torchvision, заменяет GUI OpenCV на headless-версию, устанавливает FFmpeg и этот CLI. Его можно выполнить на CPU-инстансе: CUDA-сборка будет проверена после запуска GPU. На GPU скрипт дополнительно проверяет реальный CUDA-вызов.

Рекомендуемый фрагмент `~/.bashrc`:

```bash
export CP_4DA_ENV="$HOME/.conda/envs/4danyone"
export CP_4DA_REPO_ROOT="$HOME/work/4DAnyone"
export CP_4DA_DATA_ROOT="$HOME/4danyone-data"
export CP_4DA_MODEL_DIR="$CP_4DA_DATA_ROOT/models"
export CP_4DA_INPUT_DIR="$CP_4DA_DATA_ROOT/input"
export CP_4DA_JOBS_DIR="$CP_4DA_DATA_ROOT/jobs"
export PYTHONNOUSERSITE=1

source /opt/conda/etc/profile.d/conda.sh
conda activate "$CP_4DA_ENV"
```

## Загрузка моделей

Сначала положите лицензированный архив SMPL-X в:

```text
s3://${CP_4DA_BUCKET}/models/smplx/models_smplx_v1_1.zip
```

Затем выполните:

```bash
cd "$HOME/work/4da_3dgs_pipeline"
./scripts/download_4danyone_models.sh
```

Эта стадия тоже CPU-safe. Скрипт синхронизирует уже имеющиеся файлы из S3, устанавливает SMPL-X и скачивает недостающие 4DAnyone/GVHMR/VGG-19/BiRefNet assets.

## Блокирующий запуск

Это первый режим для проверки конфигурации: терминал занят до завершения.

```bash
fourda-pipeline \
  --video "$CP_4DA_INPUT_DIR/leo.MOV" \
  --experiment-name leon_video_72views_01 \
  --views-per-layer 24 \
  --layer-pitches=-15,0,15 \
  --start-yaw 0 \
  --yaw-span 360 \
  --target-fps 30 \
  --seed 42 \
  --frame-indices 60
```

Если задан `CP_4DA_BUCKET`, результат автоматически попадёт в:

```text
s3://${CP_4DA_BUCKET}/runs/leon_video_72views_01/
```

Отключить загрузку можно флагом `--no-s3-upload`; задать другой путь — `--s3-output-uri`.

Результат выбранного момента:

```text
~/4danyone-data/runs/leon_video_72views_01/nerfstudio/frame_060/
├── transforms.json
├── sparse_pcd.ply
├── images/
└── masks/
```

Флаг `--resume` переиспользует успешно созданные промежуточные результаты. Без него существующий каталог защищён от случайной перезаписи.

## Фоновый запуск и статус

```bash
fourda-job start \
  --video "$CP_4DA_INPUT_DIR/leo.MOV" \
  --experiment-name leon_video_72views_01 \
  --views-per-layer 24 \
  --layer-pitches=-15,0,15 \
  --start-yaw 0 \
  --yaw-span 360 \
  --target-fps 30 \
  --seed 42 \
  --frame-indices 60 \
  --shutdown-on success
```

После `start` можно закрыть терминал, VS Code и вкладку браузера. Отдельный worker продолжает работать внутри JupyterLab App.

```bash
fourda-job status leon_video_72views_01
fourda-job status leon_video_72views_01 --json
fourda-job logs leon_video_72views_01 --lines 200
fourda-job logs leon_video_72views_01 --follow
fourda-job stop leon_video_72views_01
```

Процент основан на нативных стадиях 4DAnyone и отражает этап выполнения, а не точную оценку оставшегося времени. Статус атомарно сохраняется в `~/4danyone-data/jobs/<job-id>/status.json`.

### Важное ограничение фонового режима

Worker живёт **внутри SageMaker JupyterLab App**. Если App будет остановлен вручную или политикой Idle Shutdown до завершения, процесс прекратится. Для долгого прогона задайте Idle Shutdown с запасом либо используйте `--shutdown-on success`: тогда App остановит себя сразу после результата, S3 upload и email.

Политики остановки:

- `never` — App не останавливается автоматически;
- `success` — остановить только после успешного прогона;
- `always` — остановить и после успеха, и после ошибки.

Остановка реализована через SageMaker `DeleteApp`. Она прекращает GPU compute, но не удаляет Space и его постоянный EBS-том.

## Email через Amazon SNS

Execution Role должна иметь `sns:CreateTopic`, `sns:Subscribe` и `sns:Publish`. Команду настройки можно выполнить в Space после выдачи этих прав:

```bash
fourda-job configure-email --email you@example.com
```

AWS отправит письмо `Subscription Confirmation`; откройте его и подтвердите подписку. Конфигурация сохраняется в `~/.config/4da-3dgs-pipeline/aws.json`. После этого фоновые задания отправляют письмо об успехе или ошибке перед автоматической остановкой App.

Для `--shutdown-on` также нужны `sagemaker:DeleteApp` и переменные:

```bash
export CP_SM_DOMAIN_ID="d-..."
export CP_SM_SPACE_NAME="cp-4da-jupyter-${CP_DEPLOYMENT_ID}"
export CP_SM_JUPYTER_APP_NAME="default"
export CP_AWS_REGION="us-east-1"
```

## Тесты

Тесты не запускают модели и не требуют GPU:

```bash
python -m pip install -e '.[dev]'
pytest
```
