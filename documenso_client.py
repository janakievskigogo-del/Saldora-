"""Thin synchronous client for the Documenso public API v1.

The client uses the Documenso v1 container flow:

1. ``POST /api/v1/documents`` creates the document container and returns
   its numeric ID.
2. ``POST /api/v1/documents/{id}/upload`` receives the PDF as multipart form
   data (``files={'file': ('ispratnica.pdf', bytes, 'application/pdf')}``).
3. ``POST /api/v1/documents/{id}/recipients`` adds the signer when the create
   response did not already include one.
4. ``POST /api/v1/documents/{id}/send`` distributes the document so the
   recipient's ``/sign/<token>`` URL becomes usable inside the embedded iframe.

Credentials come from the ``DOCUMENSO_API_KEY`` / ``DOCUMENSO_BASE_URL``
environment variables, or explicitly from a per-account Settings record.
"""

from __future__ import annotations

import hmac
import os
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from logger import logger

DEFAULT_BASE_URL = "https://app.documenso.com"
API_PATH = "/api/v1"

_PLACEHOLDER_KEYS = {
    "",
    "api_xxxxxxxxxxxxxxxx",
    "your_api_key_here",
    "your_documenso_api_key_here",
    "replace_me",
    "changeme",
}


class DocumensoError(RuntimeError):
    """Base class for every Documenso integration failure."""


class DocumensoConfigError(DocumensoError):
    """Raised when the API key or base URL is missing or malformed."""


class DocumensoAPIError(DocumensoError):
    """Raised when Documenso returns an error or cannot be reached."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def normalize_base_url(value: str | None = None) -> str:
    """Return the plain Documenso origin for API calls.

    Documenso URLs copied from the dashboard often contain team slugs
    (``/t/<slug>``), repeated API prefixes (``/api/v1``), document routes or
    trailing slashes. None of those belong in the API origin: the client always
    rebuilds the endpoint as ``<origin>/api/v1/...``. Keeping only
    ``scheme://host`` also guarantees we never construct paths such as
    ``/api/v1/t/my-team/api/v1/documents``.
    """

    raw = (value or os.environ.get("DOCUMENSO_BASE_URL") or DEFAULT_BASE_URL).strip()
    if not raw:
        raw = DEFAULT_BASE_URL
    if "://" not in raw:
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DocumensoConfigError(
            "DOCUMENSO_BASE_URL must be an absolute http(s) URL."
        )
    if parsed.username or parsed.password:
        raise DocumensoConfigError(
            "DOCUMENSO_BASE_URL must not embed credentials."
        )

    origin = f"{parsed.scheme}://{parsed.netloc}"
    if parsed.netloc.casefold() == "app.documenso.com" and parsed.scheme != "https":
        # The hosted service is HTTPS-only; never send the API key in clear text.
        origin = f"https://{parsed.netloc}"
    return origin


def is_configured_api_key(value: str | None = None) -> bool:
    """Return whether ``value`` (or the environment) holds a real API key."""

    key = (value or os.environ.get("DOCUMENSO_API_KEY") or "").strip()
    return bool(key) and key.casefold() not in _PLACEHOLDER_KEYS


def resolve_api_key(value: str | None = None) -> str:
    """Return a usable API key or raise a configuration error."""

    key = (value or os.environ.get("DOCUMENSO_API_KEY") or "").strip()
    if not is_configured_api_key(key):
        raise DocumensoConfigError(
            "DOCUMENSO_API_KEY is not configured. Add it in Поставки → "
            "Documenso e-Signature Конфигурација."
        )
    return key


def _recipient_signing_url(
    recipients: list[dict[str, Any]] | None, base_url: str
) -> str | None:
    """Extract the direct iframe ``/sign/<token>`` URL from a recipient list.

    Only genuine signing links are returned. Dashboard or document URLs that
    some Documenso versions place in ``signingUrl``/``directLink`` are ignored
    so the embedded iframe can never route the signer to the login screen.
    """

    for recipient in recipients or []:
        token = recipient.get("token")
        if isinstance(token, str) and token.strip():
            return f"{base_url}/sign/{token.strip()}"

        for field in ("directLink", "direct_link", "signingUrl", "signing_url"):
            value = recipient.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            candidate = value.strip()
            parsed = urlparse(candidate)
            path = parsed.path or candidate
            if "/sign/" in path:
                return candidate if parsed.netloc else f"{base_url}{path}"
            # A bare opaque value with no scheme/netloc/separator is a token.
            if not parsed.scheme and not parsed.netloc and "/" not in candidate:
                return f"{base_url}/sign/{candidate}"
    return None


def _first_nonempty_str(*values: Any) -> str | None:
    """Return the first non-empty string or integer in ``values``."""

    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return None


def _recipients_from(record: Any) -> list[dict[str, Any]]:
    """Extract recipient dicts from the shapes Documenso versions return."""

    if not isinstance(record, dict):
        return []

    for key in ("recipients", "Recipient", "data"):
        value = record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = value.get("recipients") or value.get("Recipient")
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if value.get("email"):
                return [value]

    if record.get("email"):
        return [record]
    return []


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None

    if isinstance(body, dict):
        for key in ("message", "error", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"Documenso returned HTTP {response.status_code}."


class DocumensoClient:
    """Synchronous Documenso API v1 client used by the FastAPI endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        # Trim stray leading/trailing whitespace and newlines that often sneak in
        # when the key is pasted into Settings or copied from a .env file.
        self.api_key = resolve_api_key(api_key).strip()
        self.base_url = normalize_base_url(base_url)
        self.api_root = f"{self.base_url}{API_PATH}"
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DocumensoClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        # Strip again at the boundary so a key assigned after construction still
        # cannot carry hidden whitespace/newlines into the Authorization header.
        return {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.api_root}{path}"
        try:
            response = self._client.request(
                method,
                url,
                headers=self._headers(),
                json=json_body,
            )
        except httpx.HTTPError as exc:
            raise DocumensoAPIError(
                "Documenso е недостапен во моментов. Обидете се повторно."
            ) from exc

        if not response.is_success:
            response_text = response.text[:2000]
            logger.error(
                f"Documenso target URL: {url} | Status: {response.status_code} | "
                f"Body: {response_text}"
            )
            raise DocumensoAPIError(
                _error_message(response),
                status_code=response.status_code,
                response_body=_safe_json(response),
            )

        data = _safe_json(response)
        if not isinstance(data, dict):
            raise DocumensoAPIError("Documenso returned an unexpected response.")
        return data

    def _resolve_upload_url(self, upload_url: str) -> str:
        """Return an absolute upload URL without inventing nested API routes.

        Presigned S3 URLs are already absolute and are used verbatim. Relative
        URLs are anchored to the Documenso origin (not to ``/api/v1``), which
        prevents malformed targets such as ``/api/v1/api/v1/documents/...``.
        """

        candidate = upload_url.strip()
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return candidate

        resolved = urljoin(f"{self.base_url}/", candidate)
        parsed = urlparse(resolved)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise DocumensoAPIError(
                "Documenso returned an invalid upload URL."
            )
        return resolved

    def _upload_headers(self) -> dict[str, str]:
        """Headers for multipart uploads.

        ``Content-Type`` is deliberately omitted so httpx/requests can add the
        multipart/form-data boundary itself.
        """

        return {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Accept": "application/json",
        }

    def _upload_document_pdf(self, document_id: int | str, pdf_bytes: bytes) -> None:
        """Upload the PDF bytes to ``/documents/{id}/upload`` as multipart data."""

        url = f"{self.api_root}/documents/{document_id}/upload"
        try:
            response = self._client.post(
                url,
                headers=self._upload_headers(),
                files={
                    "file": ("ispratnica.pdf", pdf_bytes, "application/pdf"),
                },
            )
        except httpx.HTTPError as exc:
            logger.error(f"Documenso upload failed for URL: {url} | {exc}")
            raise DocumensoAPIError(
                "Не успеа прикачувањето на PDF документот во Documenso."
            ) from exc

        if not response.is_success:
            response_text = response.text[:2000]
            logger.error(
                f"Documenso target URL: {url} | Status: {response.status_code} | "
                f"Body: {response_text}"
            )
            raise DocumensoAPIError(
                "Не успеа прикачувањето на PDF документот во Documenso "
                f"(HTTP {response.status_code}): {_error_message(response)}",
                status_code=response.status_code,
                response_body=_safe_json(response),
            )

        logger.info(
            f"[Documenso] Uploaded PDF bytes ({len(pdf_bytes)} bytes) to doc ID "
            f"{document_id}"
        )

    def _upload_pdf(self, upload_url: str, pdf_bytes: bytes) -> None:
        if not upload_url:
            raise DocumensoAPIError("Documenso did not return an upload URL.")

        url = self._resolve_upload_url(upload_url)
        try:
            response = self._client.put(
                url,
                content=pdf_bytes,
                headers={"Content-Type": "application/pdf"},
            )
        except httpx.HTTPError as exc:
            logger.error(f"Documenso upload failed for URL: {url} | {exc}")
            raise DocumensoAPIError(
                "Не успеа прикачувањето на PDF документот во Documenso."
            ) from exc

        if not response.is_success:
            response_text = response.text[:2000]
            logger.error(
                f"Documenso target URL: {url} | Status: {response.status_code} | "
                f"Body: {response_text}"
            )
            raise DocumensoAPIError(
                "Не успеа прикачувањето на PDF документот во Documenso "
                f"(HTTP {response.status_code}): {_error_message(response)}",
                status_code=response.status_code,
                response_body=_safe_json(response),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_document(
        self,
        pdf_bytes: bytes,
        title: str,
        recipient_email: str,
        *,
        recipient_name: str | None = None,
        external_id: str | None = None,
    ) -> dict[str, Any]:
        """Create, upload and distribute a document for embedded signing.

        Returns a dictionary with ``document_id``, ``signing_url`` and the
        normalised recipient list so callers can persist the audit metadata.
        """

        if not pdf_bytes:
            raise DocumensoAPIError("PDF документот е празен.")

        payload: dict[str, Any] = {
            "title": title.strip() or "Saldora e-Ispratnica",
            "recipients": [
                {
                    "name": (recipient_name or recipient_email.split("@")[0]).strip(),
                    "email": recipient_email.strip(),
                    "role": "SIGNER",
                }
            ],
        }
        if external_id:
            payload["externalId"] = external_id

        created = self._request("POST", "/documents", json_body=payload)

        # Documenso versions differ: some return the document fields at the top
        # level, others wrap them in ``document``/``data``.
        created_document = next(
            (
                created[key]
                for key in ("document", "data")
                if isinstance(created.get(key), dict)
            ),
            {},
        )
        upload_container = next(
            (
                created[key]
                for key in ("upload", "directUpload")
                if isinstance(created.get(key), dict)
            ),
            {},
        )

        raw_document_id = _first_nonempty_str(
            created.get("id"),
            created.get("documentId"),
            created_document.get("id"),
            created_document.get("documentId"),
        )
        if raw_document_id is None:
            raise ValueError("Documenso did not return a document ID.")
        try:
            document_id = int(raw_document_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Documenso returned a non-numeric document ID: {raw_document_id!r}"
            ) from exc
        logger.info(
            f"[Documenso] Document container created with numeric ID: {document_id}"
        )

        upload_url = _first_nonempty_str(
            created.get("uploadUrl"),
            created.get("upload_url"),
            created.get("directUploadUrl"),
            created.get("direct_upload_url"),
            created_document.get("uploadUrl"),
            created_document.get("upload_url"),
            created_document.get("directUploadUrl"),
            created_document.get("direct_upload_url"),
            upload_container.get("url"),
            upload_container.get("uploadUrl"),
        )

        # Primary upload path: multipart POST to the document's upload route.
        # Older/self-hosted instances that only return a presigned URL keep
        # working through the PUT fallback below.
        try:
            self._upload_document_pdf(document_id, pdf_bytes)
        except DocumensoAPIError as exc:
            if exc.status_code in {404, 405} and upload_url:
                logger.warning(
                    "Documenso upload endpoint is unavailable; falling back to the "
                    "presigned upload URL."
                )
                self._upload_pdf(upload_url, pdf_bytes)
            else:
                raise

        recipients = _recipients_from(created) or _recipients_from(created_document)
        if not recipients:
            # Some versions accept recipients in the create payload but only
            # return the container ID. Inspect the fresh document before adding
            # another recipient, so we never duplicate the signer.
            try:
                fetched = self._request("GET", f"/documents/{document_id}")
                recipients = _recipients_from(fetched)
            except DocumensoAPIError as exc:
                if exc.status_code not in {404, 405}:
                    raise

        if not recipients:
            try:
                created_recipient = self._request(
                    "POST",
                    f"/documents/{document_id}/recipients",
                    json_body={
                        "email": recipient_email.strip(),
                        "name": (
                            recipient_name or recipient_email.split("@")[0]
                        ).strip(),
                        "role": "SIGNER",
                    },
                )
                recipients = _recipients_from(created_recipient)
            except DocumensoAPIError as exc:
                if exc.status_code not in {404, 405}:
                    raise
                logger.warning(
                    "Documenso recipients endpoint is unavailable; continuing with "
                    "recipients supplied at document creation."
                )

        # ``sendEmail=False`` keeps delivery inside Saldora; the generated
        # signing URL is embedded in the modal instead of mailing the signer.
        sent = self._request(
            "POST",
            f"/documents/{document_id}/send",
            json_body={"sendEmail": False},
        )

        sent_document = next(
            (
                sent[key]
                for key in ("document", "data")
                if isinstance(sent.get(key), dict)
            ),
            {},
        )
        sent_recipients = _recipients_from(sent) or _recipients_from(sent_document)
        if sent_recipients:
            recipients = sent_recipients

        signing_url = _recipient_signing_url(recipients, self.base_url)
        if not signing_url:
            signing_url = _first_nonempty_str(
                sent.get("signingUrl"),
                sent.get("signing_url"),
                sent_document.get("signingUrl"),
                sent_document.get("signing_url"),
            )
        if not signing_url:
            signing_url = self.get_embed_url(document_id)

        return {
            "document_id": document_id,
            "signing_url": signing_url,
            "recipients": recipients,
            "status": sent.get("status") or sent_document.get("status") or "SENT",
        }

    def get_embed_url(self, document_id: int | str) -> str:
        """Return the direct iframe URL (``/sign/...``) for a document."""

        document = self._request("GET", f"/documents/{document_id}")
        recipients = _recipients_from(document)

        signing_url = _recipient_signing_url(recipients, self.base_url)
        if not signing_url:
            raise DocumensoAPIError(
                "Documenso не врати директен линк за потпишување."
            )
        return signing_url

    def get_document(self, document_id: int | str) -> dict[str, Any]:
        """Return the raw document record, including recipients and fields."""

        return self._request("GET", f"/documents/{document_id}")

    # ------------------------------------------------------------------
    # Webhook verification
    # ------------------------------------------------------------------
    @staticmethod
    def verify_webhook_secret(
        received_secret: str | None,
        expected_secret: str | None = None,
    ) -> bool:
        """Constant-time comparison of the ``X-Documenso-Secret`` header."""

        expected = (
            expected_secret
            if expected_secret is not None
            else os.environ.get("DOCUMENSO_WEBHOOK_SECRET", "")
        ).strip()

        if not expected:
            # Mirrors Documenso's documented behaviour: no configured secret
            # means the endpoint is open (acceptable for local development).
            return True
        if not received_secret:
            return False
        return hmac.compare_digest(received_secret.strip(), expected)


def _safe_json(response: httpx.Response) -> Any:
    if response.status_code == 204 or not response.content:
        return {}
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text[:1000]}
