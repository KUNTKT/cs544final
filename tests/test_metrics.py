from cot_factors.metrics import parse_yes_no


def test_parse_final_answer():
    t = "Step 1 ...\nFinal answer: Yes\n"
    assert parse_yes_no(t) is True
    t2 = "Some text\nFinal answer: No"
    assert parse_yes_no(t2) is False


def test_parse_fallback_last_yes():
    t = "unclear reasoning\nyes"
    assert parse_yes_no(t) is True
