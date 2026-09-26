from __future__ import annotations

import os
from dataclasses import dataclass

import requests


class MailUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Mailer:
    api_key: str
    sender: str
    dev_mode: bool
    api_base: str = "https://api.resend.com"

    def send_login_code(self, email: str, code: str) -> None:
        if not self.api_key:
            if self.dev_mode:
                return
            raise MailUnavailable("Email sending is not configured")
        try:
            response = requests.post(
                f"{self.api_base}/emails",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "from": self.sender,
                    "to": [email],
                    "subject": f"Mã đăng nhập Manga Translator: {code}",
                    "text": f"Mã đăng nhập của bạn là {code}. Mã có hiệu lực 10 phút.\n\n"
                            "Nếu bạn không yêu cầu mã này, hãy bỏ qua email.",
                },
                timeout=(5, 15),
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise MailUnavailable("Mail provider unreachable") from exc
        if not response.ok:
            raise MailUnavailable(f"Mail provider HTTP {response.status_code}")


def mailer_from_env() -> Mailer:
    return Mailer(
        api_key=os.getenv("GATEWAY_RESEND_API_KEY", ""),
        sender=os.getenv("GATEWAY_MAIL_FROM", "Manga Translator <login@example.com>"),
        dev_mode=os.getenv("GATEWAY_DEV_LOGIN", "0") == "1",
    )
