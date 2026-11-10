"""Tests for the Documenso client upload + signing-URL flows."""

from __future__ import annotations

import httpx
import pytest

from documenso_client import (
    DocumensoAPIError,
    DocumensoClient,
    _recipient_signing_url,
    normalize_base_url,
)

BASE = "https://doc.test"
PDF = b"%PDF-1.7 fake pdf bytes"


def _client(handler) -> DocumensoClient:
    return DocumensoClient(
        api_key="test-key-123456",
        base_url=BASE,
        transport=httpx.MockTransport(handler),
    )


def test_create_uploads_pdf_and_returns_signing_url():
    """Create -> multipart upload -> send, using the numeric document ID."""

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/api/v1/documents":
            return httpx.Response(
                201,
                json={
                    "id": 42,
                    "recipients": [{"email": "s@example.com", "token": "tok-1"}],
                },
            )
        if (
            request.method == "POST"
            and request.url.path == "/api/v1/documents/42/upload"
        ):
            return httpx.Response(200, json={"success": True})
        if request.method == "POST" and request.url.path.endswith("/send"):
            return httpx.Response(
                200,
                json={
                    "status": "SENT",
                    "recipients": [{"email": "s@example.com", "token": "tok-1"}],
                },
            )
        return httpx.Response(
            404, json={"message": f"unexpected {request.method} {request.url.path}"}
        )

    with _client(handler) as client:
        result = client.create_document(
            PDF, "Тест", "s@example.com", external_id="ISP-1"
        )

    assert [r.method for r in requests] == ["POST", "POST", "POST"]
    assert [r.url.path for r in requests] == [
        "/api/v1/documents",
        "/api/v1/documents/42/upload",
        "/api/v1/documents/42/send",
    ]

    create_request, upload_request, _send_request = requests
    assert create_request.headers["Authorization"] == "Bearer test-key-123456"
    assert create_request.headers["Content-Type"] == "application/json"
    create_body = create_request.read().decode("utf-8")
    assert "files" not in create_body
    assert "base64" not in create_body

    upload_body = upload_request.read()
    assert upload_request.headers["Authorization"] == "Bearer test-key-123456"
    assert upload_request.headers["Accept"] == "application/json"
    assert upload_request.headers["Content-Type"].startswith("multipart/form-data")
    assert b'name="file"; filename="ispratnica.pdf"' in upload_body
    assert PDF in upload_body

    assert result["document_id"] == 42
    assert result["signing_url"] == f"{BASE}/sign/tok-1"
    assert result["status"] == "SENT"


def test_missing_or_non_numeric_document_id_raises_value_error():
    """A missing or malformed container ID is an explicit ValueError."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/documents":
            return httpx.Response(201, json={"id": "not-a-number"})
        return httpx.Response(404, json={"message": "unexpected"})

    with _client(handler) as client:
        with pytest.raises(ValueError, match="non-numeric document ID"):
            client.create_document(PDF, "Тест", "s@example.com")


def test_recipient_is_created_when_create_response_has_none():
    """If the container comes back without recipients, add the signer explicitly."""

    recipient_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/documents":
            return httpx.Response(201, json={"id": 50})
        if (
            request.method == "POST"
            and request.url.path == "/api/v1/documents/50/upload"
        ):
            return httpx.Response(200, json={"success": True})
        if request.method == "GET" and request.url.path == "/api/v1/documents/50":
            return httpx.Response(200, json={"id": 50, "recipients": []})
        if (
            request.method == "POST"
            and request.url.path == "/api/v1/documents/50/recipients"
        ):
            recipient_requests.append(request)
            return httpx.Response(
                201,
                json={
                    "id": 7,
                    "email": "s@example.com",
                    "name": "Signer",
                    "role": "SIGNER",
                    "token": "tok-rec",
                },
            )
        if request.method == "POST" and request.url.path.endswith("/send"):
            return httpx.Response(
                200,
                json={"status": "SENT", "recipients": [{"token": "tok-rec"}]},
            )
        return httpx.Response(404, json={"message": "unexpected"})

    with _client(handler) as client:
        result = client.create_document(
            PDF, "Тест", "s@example.com", recipient_name="Signer"
        )

    assert len(recipient_requests) == 1
    recipient_body = recipient_requests[0].read().decode("utf-8")
    assert '"email":"s@example.com"' in recipient_body
    assert '"name":"Signer"' in recipient_body
    assert '"role":"SIGNER"' in recipient_body
    assert result["document_id"] == 50
    assert result["signing_url"] == f"{BASE}/sign/tok-rec"


def test_legacy_instance_still_uploads_via_presigned_url():
    """A 404 from /upload falls back to the legacy presigned PUT URL."""

    put_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/documents":
            return httpx.Response(
                201,
                json={
                    "id": 7,
                    "uploadUrl": f"{BASE}/storage/upload/abc",
                    "recipients": [{"email": "s@example.com"}],
                },
            )
        if request.method == "PUT":
            put_requests.append(request)
            assert request.headers["Content-Type"] == "application/pdf"
            assert request.content == PDF
            return httpx.Response(200)
        if request.method == "POST" and request.url.path.endswith("/send"):
            return httpx.Response(
                200,
                json={
                    "status": "SENT",
                    "recipients": [{"email": "s@example.com", "token": "tok-2"}],
                },
            )
        return httpx.Response(404, json={"message": "unexpected"})

    with _client(handler) as client:
        result = client.create_document(PDF, "Тест", "s@example.com")

    assert len(put_requests) == 1
    assert result["document_id"] == 7
    assert result["signing_url"] == f"{BASE}/sign/tok-2"


def test_relative_upload_url_is_resolved_against_origin_not_api_root():
    """A relative presigned upload URL must not be nested under /api/v1 twice."""

    upload_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/documents":
            return httpx.Response(
                201,
                json={
                    "document": {"id": 11},
                    "upload": {"url": "/storage/upload/abc"},
                },
            )
        if request.method == "PUT":
            upload_requests.append(request)
            return httpx.Response(200)
        if request.method == "POST" and request.url.path.endswith("/send"):
            return httpx.Response(
                200,
                json={"status": "SENT", "recipients": [{"token": "tok-3"}]},
            )
        return httpx.Response(404, json={"message": "unexpected"})

    with _client(handler) as client:
        result = client.create_document(PDF, "Тест", "s@example.com")

    assert len(upload_requests) == 1
    assert str(upload_requests[0].url) == f"{BASE}/storage/upload/abc"
    assert result["document_id"] == 11
    assert result["signing_url"] == f"{BASE}/sign/tok-3"


def test_upload_404_surfaces_documenso_message():
    """A failing fallback upload carries Documenso's message and status."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/documents":
            return httpx.Response(
                201,
                json={"id": 9, "uploadUrl": f"{BASE}/storage/upload/gone"},
            )
        if request.method == "PUT":
            return httpx.Response(
                404,
                json={"message": "An error has occured while uploading the file"},
            )
        return httpx.Response(404, json={"message": "unexpected"})

    with _client(handler) as client:
        with pytest.raises(DocumensoAPIError) as exc_info:
            client.create_document(PDF, "Тест", "s@example.com")
    assert "uploading the file" in str(exc_info.value)
    assert exc_info.value.status_code == 404


def test_dashboard_signing_url_is_rejected():
    """A signingUrl pointing at the dashboard must never reach the iframe."""

    assert (
        _recipient_signing_url([{"signingUrl": f"{BASE}/documents/12"}], BASE)
        is None
    )


def test_normalize_base_url_strips_team_slugs_and_duplicate_paths():
    """Dashboard URLs with team slugs / repeated API paths reduce to the origin."""

    expected = "https://app.documenso.com"
    assert normalize_base_url("https://app.documenso.com/t/acme/api/v1/") == expected
    assert (
        normalize_base_url("https://app.documenso.com/api/v1/api/v1/documents")
        == expected
    )
    assert normalize_base_url("https://app.documenso.com/documents/42") == expected
