"""
Bluesky publisher — AT Protocol posting.

Posts fact-check results as threaded Bluesky posts:
  1. Root post: summary (speaker, date, verdict counts)
  2. Reply threads: one post per claim (truncated to 300 chars)
  3. Optional final post: link to full HTML report

Uses the AT Protocol API (api.bsky.app) with app-password authentication.
"""

from __future__ import annotations

import logging
import textwrap
from datetime import datetime, timezone
from typing import Optional

from truthbot.models import Report, Verdict, VerdictLabel

logger = logging.getLogger(__name__)

_BSKY_PDS_URL = "https://bsky.social"
_MAX_POST_CHARS = 300

_VERDICT_EMOJI = {
    VerdictLabel.TRUE: "✅",
    VerdictLabel.MOSTLY_TRUE: "🟢",
    VerdictLabel.MISLEADING: "⚠️",
    VerdictLabel.EXAGGERATED: "📊",
    VerdictLabel.FALSE: "❌",
    VerdictLabel.UNVERIFIABLE: "❓",
}


class BlueskyPublisher:
    """
    Post fact-check results to Bluesky as threaded posts.

    Parameters
    ----------
    handle:
        Bluesky handle (e.g. "yourname.bsky.social").
    app_password:
        Bluesky app password (not your main account password).
    """

    def __init__(
        self,
        handle: Optional[str] = None,
        app_password: Optional[str] = None,
    ) -> None:
        from truthbot.config import settings
        self._handle = handle or settings.bluesky_handle
        self._app_password = app_password or settings.bluesky_app_password
        self._session: Optional[dict] = None

    def is_configured(self) -> bool:
        """Return True if credentials are available."""
        return bool(self._handle and self._app_password)

    def post_report(self, report: Report) -> Optional[str]:
        """
        Post a fact-check report as a Bluesky thread.

        Creates a root summary post, then replies with each verdict.
        Returns the URL of the root post, or None on failure.

        Parameters
        ----------
        report:
            The completed fact-check report to publish.

        Returns
        -------
        Optional[str]
            URL of the root post on Bluesky, or None if not configured
            or posting failed.
        """
        if not self.is_configured():
            logger.info("Bluesky not configured — skipping post.")
            return None

        try:
            session = self._authenticate()
            root_ref = self._post_summary(session, report)
            self._post_verdicts(session, report, root_ref)
            url = self._post_url(root_ref, self._handle)
            logger.info("Bluesky thread posted: %s", url)
            return url
        except Exception as exc:
            logger.error("Bluesky posting failed: %s", exc)
            return None

    def format_summary_post(self, report: Report) -> str:
        """
        Format the root summary post text.

        Stays within the 300-character Bluesky limit.

        Parameters
        ----------
        report:
            The report to summarize.

        Returns
        -------
        str
            Post text, at most 300 characters.
        """
        speaker = report.transcript.speaker[:40]
        date_str = (
            report.transcript.date.strftime("%b %d, %Y")
            if report.transcript.date
            else ""
        )
        header = f"🔍 Fact check: {speaker}"
        if date_str:
            header += f" ({date_str})"

        summary = report.verdict_summary
        lines = [header, ""]
        for label, count in summary.items():
            if count > 0:
                emoji = _VERDICT_EMOJI.get(VerdictLabel(label), "•")
                lines.append(f"{emoji} {label}: {count}")

        text = "\n".join(lines)
        return text[:_MAX_POST_CHARS]

    def format_verdict_post(self, claim_text: str, verdict: Verdict) -> str:
        """
        Format a single verdict reply post.

        Parameters
        ----------
        claim_text:
            The claim text.
        verdict:
            The verdict for this claim.

        Returns
        -------
        str
            Post text, at most 300 characters.
        """
        emoji = _VERDICT_EMOJI.get(verdict.label, "•")
        label_line = f"{emoji} {verdict.label.value} ({verdict.confidence.value} confidence)"
        # Truncate claim to leave room for label + explanation
        max_claim = _MAX_POST_CHARS - len(label_line) - len(verdict.explanation[:80]) - 10
        claim_short = textwrap.shorten(claim_text, width=max(60, max_claim), placeholder="…")
        text = f'"{claim_short}"\n\n{label_line}\n{verdict.explanation[:80]}'
        return text[:_MAX_POST_CHARS]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _authenticate(self) -> dict:
        """Create a Bluesky session via the AT Protocol."""
        import httpx

        resp = httpx.post(
            f"{_BSKY_PDS_URL}/xrpc/com.atproto.server.createSession",
            json={"identifier": self._handle, "password": self._app_password},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    def _create_post(
        self,
        session: dict,
        text: str,
        reply_to: Optional[dict] = None,
    ) -> dict:
        """Create a single Bluesky post record."""
        import httpx

        record: dict = {
            "$type": "app.bsky.feed.post",
            "text": text[:_MAX_POST_CHARS],
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        if reply_to:
            record["reply"] = reply_to

        resp = httpx.post(
            f"{_BSKY_PDS_URL}/xrpc/com.atproto.repo.createRecord",
            headers={"Authorization": f"Bearer {session['accessJwt']}"},
            json={
                "repo": session["did"],
                "collection": "app.bsky.feed.post",
                "record": record,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    def _post_summary(self, session: dict, report: Report) -> dict:
        """Post the root summary and return its AT URI + CID."""
        text = self.format_summary_post(report)
        result = self._create_post(session, text)
        return {"uri": result["uri"], "cid": result["cid"]}

    def _post_verdicts(self, session: dict, report: Report, root_ref: dict) -> None:
        """Reply to the root post with each verdict."""
        reply_ref = root_ref.copy()
        for claim in report.claims[:10]:  # cap thread length
            verdict = report.verdict_for(claim.id)
            if not verdict:
                continue
            text = self.format_verdict_post(claim.text, verdict)
            reply = {"root": root_ref, "parent": reply_ref}
            result = self._create_post(session, text, reply_to=reply)
            reply_ref = {"uri": result["uri"], "cid": result["cid"]}

    @staticmethod
    def _post_url(ref: dict, handle: str) -> str:
        """Convert an AT URI to a bsky.app URL."""
        # AT URI: at://did:plc:xxx/app.bsky.feed.post/rkey
        parts = ref["uri"].split("/")
        rkey = parts[-1] if parts else ref["uri"]
        return f"https://bsky.app/profile/{handle}/post/{rkey}"
