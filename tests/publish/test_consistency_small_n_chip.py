"""The chip check must not read the neighbouring frame's percentage.

The small-sample guard replaces the "Truthy or better" percentage with a
caveat. The chip parser used to search the WHOLE page under re.S, so a frame
with no vp-stat-num silently matched the NEXT frame's number and reported the
False chip's percentage as the Truthy chip's -- a false violation that also
means the parser was never really frame-scoped.
"""
from truthbot.publish.consistency import check_report_page

REPORT = {"slug": "x", "claims_checked": 6}


def _page(truthy_body: str, false_body: str, ratio: str) -> str:
    return (
        f'<div class="vp-verdict">{ratio}</div>'
        '<div class="vp-headline-stats">'
        f'<div class="vp-headline-stat vp-stat-truthy" title="t">{truthy_body}</div>'
        f'<div class="vp-headline-stat vp-stat-false" title="f">{false_body}</div>'
        '</div>')


NUM = '<div class="vp-stat-body"><div class="vp-stat-num">%s</div></div>'
CAVEAT = ('<div class="vp-stat-body"><div class="vp-stat-lbl">'
          'Small sample — read the claims, not the score.</div></div>')


def _claims(true_n: int, false_n: int) -> list[dict]:
    return ([{"verdict": "True"}] * true_n) + ([{"verdict": "False"}] * false_n)


def test_small_sample_caveat_is_not_read_as_a_percentage():
    """6 of 6 true: the Truthy frame carries the caveat, not 0%."""
    page = _page(CAVEAT, NUM % "0%", "6 of 6 decided claims true-leaning")
    violations = check_report_page(page, REPORT, _claims(6, 0))
    assert not [v for v in violations if "vp-stat-truthy" in v], violations


def test_a_wrong_percentage_is_still_caught():
    """The fix must not blunt the check it is scoped inside."""
    page = _page(NUM % "50%", NUM % "0%", "6 of 6 decided claims true-leaning")
    violations = check_report_page(page, REPORT, _claims(6, 0))
    assert any("vp-stat-truthy" in v and "50%" in v for v in violations), violations


def test_a_genuinely_missing_frame_is_still_a_violation():
    """Absent AND no caveat = the markup really is broken."""
    page = _page('<div class="vp-stat-body"></div>', NUM % "0%",
                 "6 of 6 decided claims true-leaning")
    violations = check_report_page(page, REPORT, _claims(6, 0))
    assert any("vp-stat-truthy" in v for v in violations), violations
