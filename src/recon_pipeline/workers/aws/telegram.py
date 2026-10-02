"""Worker-side Telegram adapter backed by pyTelegramBotAPI."""

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

from .config import TelegramConfig


class TelegramClient:
    def __init__(self, config: TelegramConfig) -> None:
        try:
            from telebot import TeleBot
        except ImportError:
            raise RuntimeError("Telegram requires recon-pipeline[aws]") from None
        self.chat_id = config.chat_id
        self._token = config.resolve_bot_token()
        self._bot = TeleBot(self._token, threaded=False)

    def _call(self, method: str, **arguments: Any) -> Any:
        try:
            return getattr(self._bot, method)(**arguments)
        except Exception as error:
            # Transport exceptions may include a request URL containing the token.
            raise RuntimeError(
                f"Telegram {method} failed: {type(error).__name__}"
            ) from None

    def check_chat(self) -> str:
        return str(self._call("get_chat", chat_id=self.chat_id).id)

    def send_message(self, text: str) -> int:
        return int(
            self._call("send_message", chat_id=self.chat_id, text=text).message_id
        )

    def edit_message(self, message_id: int, text: str) -> None:
        self._call(
            "edit_message_text", chat_id=self.chat_id, message_id=message_id, text=text
        )

    def send_photos(self, paths: Sequence[Path], caption: str) -> tuple[int, ...]:
        from telebot.types import InputMediaPhoto

        photos = tuple(Path(path) for path in paths)
        if not 2 <= len(photos) <= 10:
            raise ValueError("Telegram media groups require between 2 and 10 photos")
        missing = [str(path) for path in photos if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Telegram photo files are missing: " + ", ".join(missing)
            )
        with ExitStack() as stack:
            media = [
                InputMediaPhoto(
                    stack.enter_context(path.open("rb")),
                    caption=caption[:1024] if index == 0 else None,
                )
                for index, path in enumerate(photos)
            ]
            result = self._call(
                "send_media_group", chat_id=self.chat_id, media=media, timeout=30
            )
        return tuple(int(message.message_id) for message in result)

    def get_updates(
        self, offset: int | None, timeout_seconds: int
    ) -> list[dict[str, Any]]:
        updates = self._call(
            "get_updates",
            offset=offset,
            timeout=max(timeout_seconds + 5, 15),
            # TeleBot treats zero as its default (20s); use a short initial poll.
            long_polling_timeout=max(timeout_seconds, 1),
            allowed_updates=["message"],
        )
        return [
            {
                "update_id": update.update_id,
                "message": update.message.json if update.message else {},
            }
            for update in updates
        ]
