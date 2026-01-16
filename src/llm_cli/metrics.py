from evaluate import load


def compute_exact_match(ids, predictions, references):
    exact_match_metric = load("exact_match")
    score = exact_match_metric.compute(predictions=predictions, references=references, ignore_case=True)
    return score["exact_match"]