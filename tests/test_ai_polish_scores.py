from app.ai_mode.polish import Line, better, quality, settled


def test_better_rejects_an_equal_rewrite():
    scores = {"natural": 2.0, "clear": 2.0, "faithful": 2.0}
    assert not better(scores, scores)


def test_quality_stays_on_one_scale():
    scores = {"natural": 3.0, "clear": 2.0, "faithful": 2.0}
    assert 0.0 <= quality(scores) <= 1.0


def test_settled_english_line_needs_a_first_person_pronoun():
    line = Line(page_index=0, id="1", source="I cannot stop", text="...", speaker="A")
    memory = {"address": [{"from": "A", "to": "B", "self": "ta", "other": "ngÆ°Æ¡i"}], "glossary": {}}
    address, _terms = settled(line, memory)
    assert address == ['self "ta"']


def test_settled_multiple_listeners_uses_named_addressee():
    line = Line(page_index=0, id='1', source='Nephis, are you ready?', text='...', speaker='Sunny')
    memory = {'address': [
        {'from': 'Sunny', 'to': 'Nephis', 'self': 'tôi', 'other': 'cô'},
        {'from': 'Sunny', 'to': 'Kai', 'self': 'tôi', 'other': 'cậu'},
    ]}
    address, _ = settled(line, memory)
    assert address == ['to Nephis: "cô"']
