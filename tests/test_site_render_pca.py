"""Reconciled-judge (PCA) claim-card rendering.

The PCA bridge emits one reconciled ``ModelVerdict`` (or none, for a split), so the
legacy "N of M agree" per-adapter vocabulary reads as a vacuous "1 of 1 agree" and
split claims render a blank model strip. These tests pin the reconciled-judge mode:
panel-vote summary + Layer A→panel→CRM-114 provenance strip, split claims included.
"""

from __future__ import annotations

from truthbot.verdict import bridge
from truthbot.publish import site


def _card(row, claim):
    b = bridge.bridge([row], [claim]).bundles[0]
    b.claim.is_checkable = True
    return site._claim_card(b, 0, 5, standalone=True)


def _claim(sid, text, source):
    return {"sid": sid, "text": text, "speaker": "X", "date_str": "2026-02-24",
            "layer_a": {"label": "check-worthy", "source": source}}


def test_resolved_pca_card_speaks_panel_vote_vocabulary_with_provenance():
    row = {"sid": "s:0", "status": "resolved", "verdict": "FALSE", "confidence": 0.8,
           "citations": [], "reasoning": "Contradicted by BLS.",
           "votes": {"MISLEADING": 2, "FALSE": 1}, "split": False, "escalated": True,
           "crm114": {"stage1": "MISLEADING", "final": "FALSE"}}
    html = _card(row, _claim("s:0", "Inflation is the highest ever.", "A2"))

    assert "Reconciled judgment" in html
    assert "2 of 3</span> seats agree" in html          # not "1 of 1 agree"
    assert "1 of 1" not in html
    # provenance chain surfaced (Layer A + tally + Severity Classifier override).
    # The two-stage FALSE-vs-MISLEADING discriminator is shown to readers as
    # "Severity Classifier"; the internal identifier remains "CRM-114".
    assert "Layer A: check-worthy (A2)" in html
    assert "PCA panel: Misleading ×2, False ×1" in html
    assert "Severity Classifier: MISLEADING→FALSE" in html
    # No reader-facing "CRM-114" anywhere in the rendered card (provenance strip,
    # tooltip, or the reasoning-body override annotation).
    assert "CRM-114" not in html


def test_split_pca_card_shows_tally_not_blank_strip():
    row = {"sid": "s:1", "status": "disagreement", "verdict": None, "confidence": None,
           "citations": [], "reasoning": "", "votes": {"TRUE": 1, "FALSE": 1},
           "split": True, "escalated": True}
    html = _card(row, _claim("s:1", "The border is fully secure.", "A1"))

    assert "Panel split" in html
    assert "False ×1, True ×1" in html
    assert "No single verdict" in html                  # the empty-grid placeholder
    assert "0 of 0" not in html                         # the old vacuous tally


def test_sources_consulted_rendered_when_pack_nonempty_but_nothing_cited():
    # A split (Unverifiable) claim that cited nothing, but a real pack WAS
    # retrieved. The card must surface "Sources consulted (N)" with the real
    # URLs and must NOT render a bare "No sources retrieved." empty-state.
    from truthbot.verdict.evidence_pack import EvidencePack, PackItem
    from truthbot.models import SourceTier

    pack = EvidencePack(
        sid="s:2", window=None,
        items=[
            PackItem(pack_id="E1", source_name="BLS",
                     source_url="https://bls.gov/data",
                     tier=SourceTier.GOVERNMENT, snippet="Unemployment 3.9%",
                     retrieved_at="2026-01-01T00:00:00+00:00", sha256="a"),
            PackItem(pack_id="E2", source_name="AP",
                     source_url="https://apnews.com/story",
                     tier=SourceTier.WIRE, snippet="AP reports",
                     retrieved_at="2026-01-01T00:00:00+00:00", sha256="b"),
        ],
    )
    row = {"sid": "s:2", "status": "disagreement", "verdict": None,
           "confidence": None, "citations": [], "reasoning": "",
           "votes": {"TRUE": 1, "FALSE": 1}, "split": True, "escalated": True}
    b = bridge.bridge([row], [_claim("s:2", "A claim.", "A1")], {"s:2": pack}).bundles[0]
    b.claim.is_checkable = True
    html = site._claim_card(b, 0, 5, standalone=True)

    assert "Sources consulted (2)" in html
    assert "bls.gov/data" in html
    assert "apnews.com/story" in html
    # nothing cited -> the combined-evidence empty state must NOT be a bare
    # "No sources retrieved." claim; it points at Sources consulted instead.
    assert "No sources retrieved." not in html


def test_per_seat_predictions_render_with_models_and_collapse_default():
    # P67 review round A (2026-07-19): when by_role was captured, the provenance
    # strip names what each seat predicted (with the seat's model when the report
    # roster is provided), and the Sources consulted block is COLLAPSED by default.
    from truthbot.verdict.evidence_pack import EvidencePack, PackItem
    from truthbot.models import SourceTier

    pack = EvidencePack(
        sid="s:4", window=None,
        items=[PackItem(pack_id="E1", source_name="BLS",
                        source_url="https://bls.gov/data",
                        tier=SourceTier.GOVERNMENT, snippet="Unemployment 3.9%",
                        retrieved_at="2026-01-01T00:00:00+00:00", sha256="a")],
    )
    row = {"sid": "s:4", "status": "resolved", "verdict": "MISLEADING",
           "confidence": 0.8, "citations": ["E1"], "reasoning": "r",
           "votes": {"MISLEADING": 2, "FALSE": 1},
           "by_role": {"proposer": ["MISLEADING"], "critic": ["FALSE"],
                       "arbiter": ["MISLEADING"]},
           "split": False, "escalated": True}
    b = bridge.bridge([row], [_claim("s:4", "A claim.", "A2")], {"s:4": pack}).bundles[0]
    b.claim.is_checkable = True
    roster = {"name": "dev", "seats": {"proposer": ["mistral"],
                                       "critic": ["dsv4-flash"],
                                       "arbiter": ["claude-haiku"]}}
    html = site._claim_card(b, 0, 5, standalone=True, panel_roster=roster)

    assert "proposer (mistral): Misleading" in html
    assert "critic (dsv4-flash): False" in html
    assert "arbiter (claude-haiku): Misleading" in html
    # collapsed by default — no `open` attribute on the sources details element
    assert '<details class="evidence-details" open>' not in html
    assert "Sources consulted (1)" in html
    # without a roster, seats still render by role name alone
    html2 = site._claim_card(b, 0, 5, standalone=True)
    assert "proposer: Misleading" in html2


def test_pca_chip_shows_fine_label_not_falsey_umbrella():
    # 2026-07-19 review: a Misleading panel verdict rendered under a "Falsey"
    # headline chip (the legacy Truthy-scale strict projection). PCA cards must
    # headline the panel's own 4-label verdict; umbrella buckets are legacy-only.
    row = {"sid": "s:6", "status": "resolved", "verdict": "MISLEADING",
           "confidence": 0.8, "citations": [], "reasoning": "r",
           "votes": {"MISLEADING": 2, "TRUE": 1},
           "by_role": {"proposer": ["MISLEADING"], "critic": ["TRUE"],
                       "arbiter": ["MISLEADING"]},
           "split": True, "escalated": True}
    html = _card(row, _claim("s:6", "New laws subvert elections.", "A2"))
    assert "Falsey" not in html
    assert 'data-coarse-lenient="Misleading"' in html
    assert 'data-coarse-strict="Misleading"' in html


def test_sources_consulted_shows_pack_ids():
    # Model reasoning cites E1/E2/…; the sources list must render those ids so
    # the citations are traceable (ids were captured but never displayed).
    from truthbot.verdict.evidence_pack import EvidencePack, PackItem
    from truthbot.models import SourceTier

    pack = EvidencePack(
        sid="s:7", window=None,
        items=[PackItem(pack_id="E1", source_name="BLS",
                        source_url="https://bls.gov/data",
                        tier=SourceTier.GOVERNMENT, snippet="s",
                        retrieved_at="2026-01-01T00:00:00+00:00", sha256="a"),
               PackItem(pack_id="E2", source_name="AP",
                        source_url="https://apnews.com/story",
                        tier=SourceTier.WIRE, snippet="s",
                        retrieved_at="2026-01-01T00:00:00+00:00", sha256="b")],
    )
    row = {"sid": "s:7", "status": "resolved", "verdict": "TRUE", "confidence": 0.9,
           "citations": ["E2"], "reasoning": "E2 confirms it.",
           "votes": {"TRUE": 2}, "split": False, "escalated": False}
    b = bridge.bridge([row], [_claim("s:7", "A claim.", "A2")], {"s:7": pack}).bundles[0]
    b.claim.is_checkable = True
    html = site._claim_card(b, 0, 5, standalone=True)
    assert '<span class="ev-id">[E1]</span>' in html
    assert '<span class="ev-id">[E2]</span>' in html


def test_reasoning_eids_link_to_sources_consulted_anchors():
    # P67 Round B follow-up: an E-id mentioned in reasoning becomes a jump link
    # to the matching Sources-consulted item; ids not in the pack stay plain.
    from truthbot.verdict.evidence_pack import EvidencePack, PackItem
    from truthbot.models import SourceTier

    pack = EvidencePack(
        sid="s:8", window=None,
        items=[PackItem(pack_id="E1", source_name="BLS",
                        source_url="https://bls.gov/data",
                        tier=SourceTier.GOVERNMENT, snippet="s",
                        retrieved_at="2026-01-01T00:00:00+00:00", sha256="a")],
    )
    row = {"sid": "s:8", "status": "resolved", "verdict": "TRUE", "confidence": 0.9,
           "citations": ["E1"], "reasoning": "E1 confirms it; E9 does not exist.",
           "votes": {"TRUE": 2}, "split": False, "escalated": False}
    b = bridge.bridge([row], [_claim("s:8", "A claim.", "A2")], {"s:8": pack}).bundles[0]
    b.claim.is_checkable = True
    html = site._claim_card(b, 0, 5, standalone=True)

    import re as _re
    m = _re.search(r'href="#(ev-[A-Za-z0-9_-]+)-E1"', html)
    assert m, "reasoning E1 must render as an anchor link"
    anchor = f'id="{m.group(1)}-E1"'
    assert anchor in html                      # the pack item carries the target id
    assert 'class="ev-ref"' in html
    assert 'href="#' + m.group(1) + '-E9"' not in html   # unknown id stays plain text


def test_tie_routed_card_copy_names_the_severity_classifier():
    # A DISAGREEMENT tie resolved by the stage-2 discriminator must not claim
    # "PCA panel resolved X" — the panel did not resolve; the classifier did.
    row = {"sid": "s:5", "status": "resolved", "verdict": "FALSE", "confidence": None,
           "citations": [], "reasoning": "",
           "votes": {"FALSE": 1, "MISLEADING": 1, "UNVERIFIABLE": 1},
           "by_role": {"proposer": ["FALSE"], "critic": ["MISLEADING"],
                       "arbiter": ["UNVERIFIABLE"]},
           "split": True, "escalated": True,
           "crm114": {"stage1": "DISAGREEMENT", "final": "FALSE"}}
    html = _card(row, _claim("s:5", "A tie claim.", "A2"))
    assert "Panel split with no plurality" in html
    assert "Severity Classifier resolved False" in html
    assert "PCA panel resolved" not in html
    assert "CRM-114" not in html          # reader-facing rename holds


def test_legacy_multi_adapter_card_unchanged():
    # >1 model verdict + empty provenance => classic "Model consensus" path.
    from datetime import datetime, timezone
    from truthbot.models import (
        Claim, ConsensusVerdict, Confidence, ModelVerdict, VerdictBundle, VerdictLabel,
    )
    claim = Claim(transcript_id="t", text="A claim.", speaker="X", is_checkable=True)
    mvs = [
        ModelVerdict(adapter_name=f"a{i}", model_id=f"m{i}", claim_id=claim.id,
                     label=VerdictLabel.TRUE, confidence=Confidence.HIGH, explanation="r")
        for i in range(3)
    ]
    consensus = ConsensusVerdict(
        claim_id=claim.id, model_verdicts=mvs, consensus_label=VerdictLabel.TRUE,
        consensus_verdict="True", confidence=Confidence.HIGH, agreement=True,
        consensus_strength="strong", explanation="x",
    )
    b = VerdictBundle(claim=claim, speaker="X", date_str="2026-02-24",
                      model_verdicts=mvs, consensus=consensus)
    html = site._claim_card(b, 0, 5, standalone=True)
    assert "Model consensus" in html
    assert "3 of 3" in html
    assert "Reconciled judgment" not in html
    assert "pca-provenance" not in html


# ── Statement Triage view ─────────────────────────────────────────────────────

from datetime import datetime


def _triage_report(characterization):
    return site.SiteReport(
        report_id="abcdef12-0000-0000-0000-000000000000",
        speaker="Donald Trump",
        role="President",
        date=datetime(2026, 2, 24),
        venue="U.S. Capitol",
        transcript_source_url="",
        bundles=[],
        characterization=characterization,
    )


_SAMPLE_CHAR = [
    {"sid": "trump:0", "speech": "trump", "idx": 0,
     "text": "Well, thank you very much, everybody.",
     "context": "", "label": "non-check-worthy", "source": "A1", "a1_score": 0.0},
    {"sid": "trump:1", "speech": "trump", "idx": 1,
     "text": "This is the greatest economy in the history of the world, believe me.",
     "context": "", "label": "opinion", "source": "A2", "a1_score": 0.42},
]


def test_statement_triage_page_lists_set_aside_sentences_and_stages():
    sr = _triage_report(_SAMPLE_CHAR)
    html = site._render_statement_triage(sr)
    assert html  # non-empty characterization -> a page is rendered
    # heading / title
    assert "Statement Triage" in html
    assert "<title>Statement Triage — Donald Trump" in html
    # both set-aside sentences appear verbatim (escaped)
    assert "Well, thank you very much, everybody." in html
    assert "greatest economy in the history of the world" in html
    # grouped by stage, with the A1 lexical prefilter and A2 classifier surfaced
    assert "Lexical prefilter (Stage A1)" in html
    assert "Check-worthiness classifier (Stage A2)" in html
    # per-sentence provenance: stage tag, label, and a1_score
    assert "a1_score: 0.42" in html
    assert "label: opinion" in html
    # breadcrumb links back to the report page
    assert "reports/2026-02-24-donald-trump-abcdef.html" in html


def test_statement_triage_empty_renders_nothing_legacy_clean():
    sr = _triage_report([])
    assert site._render_statement_triage(sr) == ""


def test_publish_emits_triage_page_only_when_characterization_present(tmp_path):
    pub = site.SitePublisher(site_root=tmp_path)
    # With characterization -> a <slug>-triage.html page + a cross-link on the report.
    sr = _triage_report(_SAMPLE_CHAR)
    pub.publish(sr)
    triage = tmp_path / "reports" / f"{sr.triage_slug}.html"
    assert triage.exists()
    assert "Statement Triage" in triage.read_text(encoding="utf-8")
    report = tmp_path / "reports" / f"{sr.report_slug}.html"
    assert "See what we set aside and why" in report.read_text(encoding="utf-8")

    # Without characterization (legacy) -> no triage page, no cross-link.
    sr2 = site.SiteReport(
        report_id="99999999-0000-0000-0000-000000000000",
        speaker="Jane Doe", role="", date=datetime(2025, 1, 1),
        venue="", transcript_source_url="", bundles=[],
    )
    pub.publish(sr2)
    assert not (tmp_path / "reports" / f"{sr2.triage_slug}.html").exists()
    assert "Statement Triage" not in (
        tmp_path / "reports" / f"{sr2.report_slug}.html"
    ).read_text(encoding="utf-8")


def test_offline_backfill_writes_triage_page_from_artifact(tmp_path):
    import json

    # Minimal existing site with a published report + reports.json index.
    (tmp_path / "reports").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "assets").mkdir(parents=True)
    slug = "2026-02-24-donald-trump-0c33d1"
    (tmp_path / "reports" / f"{slug}.html").write_text("<html>report</html>", encoding="utf-8")
    (tmp_path / "data" / "reports.json").write_text(
        json.dumps([{"id": "0c33d1c9", "date": "2026-02-24", "speaker": "Donald Trump",
                     "url": f"reports/{slug}.html"}]),
        encoding="utf-8",
    )

    artifact = {
        "run_id": "0e0d9336-aaaa-bbbb-cccc-dddddddddddd",
        "meta": {"speaker": "Donald Trump", "date": "2026-02-24", "venue": "U.S. Capitol"},
        "characterization": _SAMPLE_CHAR,
    }
    art_path = tmp_path / "artifact.json"
    art_path.write_text(json.dumps(artifact), encoding="utf-8")

    out = site.backfill_statement_triage(art_path, tmp_path)
    assert out is not None and out.exists()
    # Named to match the EXISTING report slug (auto-detected from reports.json),
    # not the artifact's own run_id.
    assert out.name == f"{slug}-triage.html"
    body = out.read_text(encoding="utf-8")
    assert "Statement Triage" in body
    assert "Well, thank you very much, everybody." in body
    assert "greatest economy in the history of the world" in body
    # breadcrumb points back at the existing published report page
    assert f"reports/{slug}.html" in body


def test_offline_backfill_no_characterization_returns_none(tmp_path):
    import json
    art_path = tmp_path / "empty.json"
    art_path.write_text(json.dumps(
        {"run_id": "x", "meta": {"speaker": "X", "date": "2026-01-01"}, "characterization": []}
    ), encoding="utf-8")
    assert site.backfill_statement_triage(art_path, tmp_path) is None


# ── PCA panel composition (per-run roster provenance) ──────────────────────────


def _roster_report(panel_roster):
    return site.SiteReport(
        report_id="abcdef12-0000-0000-0000-000000000000",
        speaker="Donald Trump",
        role="President",
        date=datetime(2026, 2, 24),
        venue="U.S. Capitol",
        transcript_source_url="",
        bundles=[],
        panel_roster=panel_roster,
    )


_DEV_ROSTER = {"name": "dev", "seats": {
    "proposer": ["mistral"], "critic": ["dsv4-flash"], "arbiter": ["claude-haiku"]}}


def test_panel_composition_helper_renders_roles_and_models():
    html = site._panel_composition_html(_roster_report(_DEV_ROSTER))
    assert html
    assert "PCA panel composition" in html
    # roster name surfaced
    assert "roster: dev" in html
    # each seat role + its model alias
    assert "Proposer" in html and "mistral" in html
    assert "Critic" in html and "dsv4-flash" in html
    assert "Arbiter" in html and "claude-haiku" in html
    # ordering: proposer before critic before arbiter
    assert html.index("Proposer") < html.index("Critic") < html.index("Arbiter")


def test_panel_composition_rendered_once_in_report_html():
    html = site._render_report(_roster_report(_DEV_ROSTER))
    assert html.count("PCA panel composition") == 1
    assert "claude-haiku" in html


def test_panel_composition_empty_renders_nothing_legacy_clean():
    # No roster at all → helper renders nothing and the report omits the block.
    assert site._panel_composition_html(_roster_report({})) == ""
    assert "PCA panel composition" not in site._render_report(_roster_report({}))
    # Roster name present but all seats empty → still nothing (nothing to show).
    empty_seats = {"name": "dev", "seats": {"proposer": [], "critic": [], "arbiter": []}}
    assert site._panel_composition_html(_roster_report(empty_seats)) == ""


def test_panel_composition_escapes_model_alias():
    hostile = {"name": "x", "seats": {"proposer": ["<script>evil</script>"]}}
    html = site._panel_composition_html(_roster_report(hostile))
    assert "<script>evil" not in html
    assert "&lt;script&gt;evil" in html
