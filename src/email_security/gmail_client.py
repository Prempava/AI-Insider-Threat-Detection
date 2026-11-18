
import base64
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class GmailClient:
    """Read-only Gmail client for the Insider Threat Detection project."""

    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly"
    ]

    ROOT = Path(__file__).resolve().parents[2]

    CREDENTIALS_FILE = (
        ROOT / "credentials" / "gmail_credentials.json"
    )

    TOKEN_FILE = (
        ROOT / "credentials" / "token.json"
    )

    def __init__(self):
        self.service = None

    def authenticate(self):
        """Authenticate the user and create Gmail API service."""

        if not self.CREDENTIALS_FILE.exists():
            raise FileNotFoundError(
                f"OAuth credentials not found: {self.CREDENTIALS_FILE}"
            )

        credentials = None

        if self.TOKEN_FILE.exists():
            credentials = Credentials.from_authorized_user_file(
                str(self.TOKEN_FILE),
                self.SCOPES,
            )

        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

        if not credentials or not credentials.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.CREDENTIALS_FILE),
                self.SCOPES,
            )

            credentials = flow.run_local_server(
                port=0,
                access_type="offline",
                prompt="consent",
            )

        self.TOKEN_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

        self.service = build(
            "gmail",
            "v1",
            credentials=credentials,
        )

        return True

    def get_authenticated_email(self):
        """Return the email address of the authenticated Gmail account."""

        if self.service is None:
            self.authenticate()

        profile = (
            self.service.users()
            .getProfile(userId="me")
            .execute()
        )

        return profile.get("emailAddress", "").strip().lower()


    @staticmethod
    def _decode_body(data):
        """Decode Gmail API base64url message body."""

        if not data:
            return ""

        decoded = base64.urlsafe_b64decode(
            data + "=" * (-len(data) % 4)
        )

        return decoded.decode(
            "utf-8",
            errors="replace",
        )

    def _extract_body(self, payload):
        """Extract readable text from Gmail message payload."""

        mime_type = payload.get("mimeType", "")
        body_data = payload.get("body", {}).get("data")

        if mime_type == "text/plain" and body_data:
            return self._decode_body(body_data)

        parts = payload.get("parts", [])

        plain_text = []
        html_text = []

        for part in parts:

            # Ignore file attachments.
            if part.get("filename"):
                continue

            content = self._extract_body(part)

            if not content:
                continue

            part_type = part.get("mimeType", "")

            if part_type == "text/plain":
                plain_text.append(content)

            elif part_type == "text/html":
                html_text.append(content)

            else:
                plain_text.append(content)

        if plain_text:
            return "\n".join(plain_text)

        if html_text:
            return "\n".join(html_text)

        if mime_type == "text/html" and body_data:
            return self._decode_body(body_data)

        return ""

    def get_latest_emails(self, max_results=5, query=None):
        """
        Retrieve emails using an optional Gmail search query.

        Examples:
            get_latest_emails(5)
            get_latest_emails(5, 'subject:"Salary Processing"')
        """

        if self.service is None:
            self.authenticate()

        if not isinstance(max_results, int) or not 1 <= max_results <= 100:
            raise ValueError(
                "max_results must be an integer between 1 and 100"
            )

        request_params = {
            "userId": "me",
            "maxResults": max_results,
        }

        if query:
            request_params["q"] = query

        response = (
            self.service.users()
            .messages()
            .list(**request_params)
            .execute()
        )

        messages = response.get("messages", [])

        emails = []

        for message in messages:

            full_message = (
                self.service.users()
                .messages()
                .get(
                    userId="me",
                    id=message["id"],
                    format="full",
                )
                .execute()
            )

            payload = full_message.get("payload", {})

            headers = {}

            for header in payload.get("headers", []):

                header_name = header.get("name", "").lower()
                header_value = header.get("value", "")

                headers[header_name] = header_value

            email_data = {
                "id": full_message.get("id"),
                "thread_id": full_message.get("threadId"),
                "sender": headers.get("from", ""),
                "to": headers.get("to", ""),
                "subject": headers.get("subject", ""),
                "date": headers.get("date", ""),
                "snippet": full_message.get("snippet", ""),
                "body": self._extract_body(payload),
                "headers": headers,
            }

            emails.append(email_data)

        return emails
