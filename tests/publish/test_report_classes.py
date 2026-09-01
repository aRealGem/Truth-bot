"""Occasion classes and the class-partitioned rate statistic (M-6)."""
import pytest

from truthbot.publish import site
from truthbot.publish.site import (UNCLASSIFIED, _genre_rate_table,
                                   report_class, report_class_label,
                                   report_class_label_inline,
                                   report_class_order)

PRESIDENTIAL = ["clinton_1998", "gwbush_2006", "obama_2014", "biden_2022",
                "trump_2026"]
SENATE = ["budd_2025-04-02", "cruz_2026-06-24", "tillis_2025-01-23",
          "warren_2025-04-29"]


@pytest.fixture
def corpus(monkeypatch):
    """A rate table over both classes, with reason-coded rows for two speeches."""
    rates = {sid: {"checked": 100} for sid in PRESIDENTIAL + SENATE}
    monkeypatch.setattr(site, "_CORPUS_GENRE_RATES", rates)
    pills = {}
    # trump: 9 coded of 100 -> highest among presidential
    for i in range(9):
        pills[f"t{i}"] = {"sid": f"trump_2026:{i}"}
    # obama: 2 coded
    for i in range(2):
        pills[f"o{i}"] = {"sid": f"obama_2014:{i}"}
    # budd: 4 coded -> highest among senate, but LOWER than trump's 9
    for i in range(4):
        pills[f"b{i}"] = {"sid": f"budd_2025-04-02:{i}"}
    monkeypatch.setattr(site, "_REASON_PILLS", pills)
    return rates


def test_classes_are_authored_for_the_whole_corpus():
    for sid in PRESIDENTIAL:
        assert report_class(sid) == "presidential_address"
    for sid in SENATE:
        assert report_class(sid) == "senate_floor"


def test_an_unlisted_report_is_unclassified_not_guessed():
    assert report_class("some_2030") == UNCLASSIFIED
    assert report_class("") == UNCLASSIFIED


def test_labels_and_order_are_authored():
    assert report_class_label("presidential_address") == "Presidential addresses"
    assert report_class_label("senate_floor") == "Senate floor speeches"
    assert report_class_order() == ["presidential_address", "senate_floor"]


def test_inline_label_is_authored_not_lowercased():
    """'Senate' is a proper noun; mechanical .lower() would corrupt it."""
    assert report_class_label_inline("presidential_address") == "presidential addresses"
    assert report_class_label_inline("senate_floor") == "Senate floor speeches"


def test_table_is_partitioned_to_the_speech_class(corpus):
    pres = _genre_rate_table("trump_2026")
    assert set(pres) == set(PRESIDENTIAL)
    sen = _genre_rate_table("budd_2025-04-02")
    assert set(sen) == set(SENATE)


def test_no_cross_class_contamination_of_the_ranking(corpus):
    """Budd tops the senate class even though trump's rate is higher overall."""
    sen = _genre_rate_table("budd_2025-04-02")
    assert max(sen, key=lambda s: sen[s]["rate"]) == "budd_2025-04-02"
    assert "trump_2026" not in sen


def test_unclassified_is_excluded_from_every_rate_statistic(corpus, monkeypatch):
    rates = dict(corpus)
    rates["stranger_2030"] = {"checked": 100}
    monkeypatch.setattr(site, "_CORPUS_GENRE_RATES", rates)
    # It gets no table of its own ...
    assert _genre_rate_table("stranger_2030") == {}
    # ... and never lands in another class's denominator.
    assert "stranger_2030" not in _genre_rate_table("trump_2026")
    assert "stranger_2030" not in _genre_rate_table("budd_2025-04-02")


def test_unpartitioned_call_still_returns_the_whole_corpus(corpus):
    """No speech_id = no partition; the corpus-wide builder keeps working."""
    assert set(_genre_rate_table()) == set(PRESIDENTIAL + SENATE)


def test_n_floor_is_three():
    assert site._GENRE_NOTE_MIN_SPEECHES == 3


def test_a_class_below_the_floor_suppresses_the_note(monkeypatch):
    """Two speeches is not a field to rank within."""
    monkeypatch.setattr(site, "_CORPUS_GENRE_RATES",
                        {"budd_2025-04-02": {"checked": 10},
                         "cruz_2026-06-24": {"checked": 10}})
    monkeypatch.setattr(site, "_REASON_PILLS",
                        {"b0": {"sid": "budd_2025-04-02:0"}})
    table = _genre_rate_table("budd_2025-04-02")
    assert len(table) == 2
    assert len(table) < site._GENRE_NOTE_MIN_SPEECHES
