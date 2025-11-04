from rouge_score import rouge_scorer

def evaluate_rouge(candidate_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, candidate_summary)
    return scores

# Example usage
candidate = "The economy is improving with more jobs created monthly."
reference = "Job creation is increasing as the economy recovers."
print(evaluate_rouge(candidate, reference))

