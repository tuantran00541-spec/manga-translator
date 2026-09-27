from app.ai_mode.judge import _probability, evaluate_url, flagged, line_state


def test_probabilities_are_found_in_any_answer_shape():
    assert _probability({"answers": {"duplicate": {"type": "noul", "noul": 0.9}}}, "duplicate") == 0.9
    assert _probability({"results": [{"wrong_name": {"probability": 0.2}}]}, "wrong_name") == 0.2
    assert _probability({"duplicate": 0.4}, "duplicate") == 0.4
    assert _probability({"duplicate": {"noul": True}}, "duplicate") is None
    assert _probability({"answers": {"duplicate": {"probabilities": {"true": 0.8, "false": 0.2}}}}, "duplicate") == 0.8


def test_only_confident_doubts_send_a_line_back():
    assert flagged({"wrong_meaning": 0.2, "duplicate": 0.69}) == ""
    assert flagged({"wrong_meaning": 0.9, "wrong_name": 0.75}) == (
        "the meaning differs from the source or part of it is missing; a name or term does not match the glossary")


def test_the_judge_reads_the_line_its_neighbours_and_the_glossary():
    state = line_state({"source": "Master!", "translation": "Sư phụ!"}, "", "Con đây.",
                       {"names": [{"source": "Yanguo", "target": "Yanguo"}],
                        "address": [{"from": "disciples", "to": "master", "self": "bọn con", "other": "sư phụ"}]})
    assert "SOURCE: Master!" in state and "NEXT LINE: Con đây." in state and "PREVIOUS" not in state
    assert "Yanguo = Yanguo" in state and 'self "bọn con"' in state
    assert evaluate_url("https://gw.example/v1/chat/completions") == "https://gw.example/v1/evaluate"


def test_short_titles_need_a_surer_language_flag():
    assert flagged({"bad_language": 0.8}, "EPISODE 1") == ""
    assert flagged({"bad_language": 0.95}, "EPISODE 1") != ""
    assert flagged({"bad_language": 0.8}, "How much do you really know about the Nightmare Spell?") != ""
    assert flagged({"wrong_meaning": 0.8}, "SPACE THEATRE") != "", "meaning is judged as before"
