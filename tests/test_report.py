import json

from agentaudit.analysis import analyze
from agentaudit.report import forest_svg, rank_heatmap_svg, to_html, to_markdown, variance_svg
from agentaudit.simulate import SyntheticPopulation


def analysis() -> dict:  # type: ignore[type-arg]
    pop = SyntheticPopulation((0.0, 0.3, 1.2), names=("a<b>", "b", "c"))
    return json.loads(json.dumps(analyze(pop.sample(50, 4, 0), n_boot=200)))  # JSON round trip


def test_markdown_contains_all_sections() -> None:
    md = to_markdown(analysis())
    for needle in (
        "Scores with 95%",
        "Ranking uncertainty",
        "Variance decomposition",
        "Power planning",
    ):
        assert needle in md
    assert md.count("|") > 40


def test_html_is_self_contained_and_escapes() -> None:
    page = to_html(analysis(), "T <x>")
    assert page.startswith("<!doctype html>") and "<svg" in page
    assert "http://" not in page.replace("http://www.w3.org", "") and "<script" not in page
    assert "a&lt;b&gt;" in page and "T &lt;x&gt;" in page
    assert "<table>" in page


def test_svg_helpers_render() -> None:
    a = analysis()
    assert forest_svg(a["models"]).count("<circle") == 3
    assert rank_heatmap_svg(a["ranking"]).count("<rect") == 9
    assert variance_svg(a["variance"]["shares"]).count("<rect") >= 8
