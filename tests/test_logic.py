"""Checks that need no PyTorch and no model download:   python tests/test_logic.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import htr_translate as TR


def test_squash_repeats():
    assert TR.squash_repeats("A: A: A: A: A: hello") == "A: A: hello"
    assert TR.squash_repeats("x a b a b a b a b") == "x a b a b"
    assert TR.squash_repeats("a normal sentence stays the same") == "a normal sentence stays the same"
    assert TR.is_degenerate("(Sho) " * 40 + "The Enchanted took two years")
    assert not TR.is_degenerate("The Enchanted took about two years to complete and was well received.")


def test_reliability_flags_garbage():
    import htr_data as D
    texts = ["यामुळे आपल्या भावातील उद्योगाला अधिक चालना मिळण्याची शक्यता आहे."] * 20
    lm = D.CharNgramLM(order=5).fit(texts)
    base = sum(TR.bits_per_char(t, lm) for t in texts) / len(texts)
    assert TR.reliability(texts[0], lm, base)["reliability"] == "HIGH"
    assert TR.reliability("ा ा ा ा ा ि ा ा ु ा", lm, base)["reliability"] == "LOW"
    assert TR.reliability("", lm, base)["reliability"] == "LOW"


if __name__ == "__main__":
    test_squash_repeats(); test_reliability_flags_garbage(); print("all logic tests passed")
