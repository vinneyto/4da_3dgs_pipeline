import json
from dataclasses import replace

import pytest
from telebot import apihelper

from recon_pipeline.workers.aws import aws
from recon_pipeline.workers.aws.config import TelegramConfig
from test_aws_health import make_config


@pytest.fixture
def telegram_transport(monkeypatch, tmp_path):
    token = "123456:fixture-secret"
    worker = replace(
        make_config(tmp_path),
        sns=None,
        telegram=TelegramConfig(chat_id="123456", bot_token=token),
    )
    calls = []
    responses = {}

    class Response:
        status_code = 200

        def __init__(self, result):
            self.result = result
            self.text = json.dumps({"ok": True, "result": result})

        def json(self):
            return {"ok": True, "result": self.result}

    def request(method, url, **options):
        operation = url.rsplit("/", 1)[-1]
        files = options.get("files") or {}
        handles = [
            value[1] if isinstance(value, tuple) else value for value in files.values()
        ]
        contents = [handle.read() for handle in handles]
        calls.append((operation, options, handles, contents))
        return Response(responses[operation])

    monkeypatch.setattr(apihelper, "CUSTOM_REQUEST_SENDER", request)
    return worker, calls, responses


def message(message_id):
    return {
        "message_id": message_id,
        "date": 0,
        "chat": {"id": 123456, "type": "private"},
        "text": "hello",
    }


def test_telegram_message_can_be_replaced_in_place(telegram_transport):
    worker, calls, responses = telegram_transport
    responses.update(sendMessage=message(42), editMessageText=message(42))
    assert aws.publish_telegram(worker, "🔵 Pass started", "CPU: test") == 42
    aws.edit_telegram(worker, 42, "✅ Pass completed", "Duration: 1.0s")
    assert [call[0] for call in calls] == ["sendMessage", "editMessageText"]
    assert calls[0][1]["params"]["text"] == "🔵 Pass started\n\nCPU: test"
    assert calls[1][1]["params"]["text"] == "✅ Pass completed\n\nDuration: 1.0s"
    assert calls[1][1]["params"]["message_id"] == 42
    assert calls[0][1]["params"]["chat_id"] == worker.telegram.chat_id


def test_telegram_photos_are_sent_as_one_media_group(telegram_transport, tmp_path):
    worker, calls, responses = telegram_transport
    paths = tuple(tmp_path / f"camera-{index}.jpg" for index in range(4))
    for path in paths:
        path.write_bytes(b"jpeg")
    responses["sendMediaGroup"] = [message(80 + index) for index in range(4)]
    assert aws.publish_telegram_photos(worker, paths, "4DAnyone previews") == (
        80,
        81,
        82,
        83,
    )
    operation, options, handles, contents = calls[0]
    assert operation == "sendMediaGroup"
    assert options["params"]["chat_id"] == worker.telegram.chat_id
    media = json.loads(options["params"]["media"])
    assert len(media) == 4 and media[0]["caption"] == "4DAnyone previews"
    assert all("caption" not in item for item in media[1:])
    assert contents == [b"jpeg"] * 4
    assert all(handle.closed for handle in handles)


def test_telegram_access_and_command_polling_use_library(telegram_transport):
    worker, calls, responses = telegram_transport
    incoming = {
        **message(9),
        "from": {"id": 123456, "is_bot": False, "first_name": "User"},
        "text": "/shutdown job-01",
    }
    responses.update(
        getChat={"id": 123456, "type": "private"},
        getUpdates=[{"update_id": 101, "message": incoming}],
    )
    assert aws._check_telegram(worker) == "123456"
    assert aws.get_telegram_updates(worker, 100, 20) == [
        {"update_id": 101, "message": incoming}
    ]
    params = calls[-1][1]["params"]
    assert params["offset"] == 100 and params["timeout"] == 20
    assert json.loads(params["allowed_updates"]) == ["message"]
    assert calls[-1][1]["timeout"][1] >= 25


def test_telegram_token_from_environment_stays_in_worker(
    telegram_transport, monkeypatch
):
    worker, calls, responses = telegram_transport
    monkeypatch.setenv("FIXTURE_TELEGRAM_TOKEN", worker.telegram.bot_token)
    worker = replace(
        worker,
        telegram=TelegramConfig(
            chat_id="123456", bot_token_env="FIXTURE_TELEGRAM_TOKEN"
        ),
    )
    responses["getChat"] = {"id": 123456, "type": "private"}
    assert aws._check_telegram(worker) == "123456"
    assert calls[0][0] == "getChat"


def test_telegram_transport_failure_is_noncritical_and_hides_token(
    telegram_transport, monkeypatch
):
    worker, _, _ = telegram_transport

    def fail(method, url, **options):
        raise OSError(f"request failed: {url}")

    monkeypatch.setattr(apihelper, "CUSTOM_REQUEST_SENDER", fail)
    result = aws.publish_notification(worker, "subject", "body")
    assert result == {"telegram": "skipped: Telegram send_message failed: OSError"}
    assert worker.telegram.bot_token not in str(result)
    channels = aws.check_notification_channels(worker)
    assert (
        channels["telegram_status"] == "unavailable: Telegram get_chat failed: OSError"
    )
