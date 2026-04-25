"""Synthetic calibration data generator.

Generates question-answer pairs with ground-truth skill scores and writes
them to adaptive_interview_env/data/calibration_refs.json.

Usage:
    python training/generate_data.py --output adaptive_interview_env/data/calibration_refs.json
    python training/generate_data.py --num_samples 200 --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_interview_env.constants import SKILL_DIMENSIONS, DOMAINS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference rubric — used to assign ground-truth scores to synthetic answers
# ---------------------------------------------------------------------------

QUALITY_TEMPLATES = {
    "high": {
        "correctness": (0.85, 0.98),
        "edge_case_coverage": (0.80, 0.95),
        "complexity_analysis": (0.80, 0.95),
        "tradeoff_reasoning": (0.80, 0.95),
        "communication_clarity": (0.82, 0.96),
    },
    "medium": {
        "correctness": (0.50, 0.75),
        "edge_case_coverage": (0.40, 0.70),
        "complexity_analysis": (0.45, 0.72),
        "tradeoff_reasoning": (0.40, 0.68),
        "communication_clarity": (0.50, 0.75),
    },
    "low": {
        "correctness": (0.10, 0.45),
        "edge_case_coverage": (0.05, 0.40),
        "complexity_analysis": (0.08, 0.42),
        "tradeoff_reasoning": (0.05, 0.38),
        "communication_clarity": (0.10, 0.45),
    },
}

HIGH_QUALITY_ANSWERS = {
    "algorithms": [
        "Binary search works by repeatedly halving the search space. We compare the target to the middle element; if equal we return, if less we search the left half, otherwise the right half. Time complexity is O(log n), space O(1) iteratively. Edge cases: empty array returns -1, single element, target smaller than all elements, target larger than all elements.",
        "Merge sort divides the array in half recursively until single elements, then merges sorted halves. Time O(n log n) in all cases, space O(n) for the merge buffer. It's stable and predictable unlike quicksort. Tradeoff: uses more memory than in-place sorts.",
    ],
    "system_design": [
        "For a URL shortener I'd use a hash function (e.g. MD5 first 6 chars) to generate short codes, store mappings in a key-value store like Redis for fast reads, and a relational DB for persistence. Handle collisions by appending a counter. Scale reads with a CDN. Edge cases: same URL submitted twice (return existing code), expired URLs, malicious URLs.",
        "Rate limiting can be implemented with a token bucket algorithm. Each user gets a bucket of N tokens refilled at rate R. Each request consumes one token. Store buckets in Redis with TTL. Tradeoffs: token bucket allows bursts vs leaky bucket which smooths traffic. Distributed rate limiting needs atomic Redis operations to avoid race conditions.",
    ],
    "databases": [
        "INNER JOIN returns only rows with matching keys in both tables. LEFT JOIN returns all rows from the left table plus matched rows from the right, with NULLs for unmatched. To find customers with no orders: SELECT c.id FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.id IS NULL. Edge case: NULL customer_id in orders table.",
        "A B-tree index stores keys in sorted order in a balanced tree, enabling O(log n) lookups, range queries, and sorted scans. Write overhead: every insert/update/delete must update the index. Space overhead: roughly 10-30% of table size. Tradeoff: more indexes = faster reads, slower writes.",
    ],
    "concurrency": [
        "A race condition occurs when two threads access shared state concurrently and the outcome depends on scheduling. Example: two threads both read balance=100, both add 50, both write 150 — final balance should be 200. Fix with a mutex. Deadlock requires: mutual exclusion, hold-and-wait, no preemption, circular wait. Break any one condition to prevent it.",
        "A mutex allows only one thread at a time. A semaphore allows N threads. Use mutex for mutual exclusion of a critical section; use semaphore for resource counting (e.g. connection pool of size 10). Lock-free structures use CAS operations to avoid blocking entirely — higher throughput under contention but harder to implement correctly.",
    ],
}

LOW_QUALITY_ANSWERS = [
    "I would use a loop to solve this.",
    "It depends on the situation.",
    "You can use a database for that.",
    "I think it works by checking each element.",
    "Just use a hash map.",
    "It's O(n) I think.",
    "You need to handle errors.",
    "I would test it first.",
]

MEDIUM_QUALITY_ANSWERS = [
    "Binary search divides the array in half each time, so it's O(log n). You need the array to be sorted first.",
    "For a URL shortener you'd need a database to store the mappings and some way to generate short codes.",
    "A race condition is when two threads access the same variable at the same time. You fix it with locks.",
    "INNER JOIN only returns rows that match in both tables. LEFT JOIN keeps all rows from the left table.",
    "An index speeds up queries by creating a separate data structure. It makes reads faster but writes slower.",
]


def _random_score(quality: str, dim: str) -> float:
    lo, hi = QUALITY_TEMPLATES[quality][dim]
    return round(random.uniform(lo, hi), 3)


def _ground_truth_scores(quality: str) -> dict:
    return {dim: _random_score(quality, dim) for dim in SKILL_DIMENSIONS}


def _sample_question(domain: str) -> str:
    bank_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "adaptive_interview_env", "data", "question_bank.json",
    )
    try:
        with open(bank_path) as f:
            bank = json.load(f)
        dim = random.choice(SKILL_DIMENSIONS)
        pool = bank.get(domain, {}).get(dim, [])
        if pool:
            item = random.choice(pool)
            return item if isinstance(item, str) else item.get("question", str(item))
    except Exception:
        pass
    return f"Explain a key concept in {domain.replace('_', ' ')}."


def _sample_answer(domain: str, quality: str) -> str:
    if quality == "high":
        pool = HIGH_QUALITY_ANSWERS.get(domain, [])
        if pool:
            return random.choice(pool)
    if quality == "low":
        return random.choice(LOW_QUALITY_ANSWERS)
    return random.choice(MEDIUM_QUALITY_ANSWERS)


def generate(output_path: str, num_samples: int = 500, model_name: str = None):
    """Generate synthetic calibration data.

    If model_name is provided, uses the model to generate answers.
    Otherwise uses template-based answers.
    """
    records = []
    qualities = ["high", "medium", "low"]

    # If a model is provided, use it for answer generation
    model = None
    tokenizer = None
    if model_name:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info(f"Loading model {model_name} for answer generation...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
            model.eval()
            logger.info("Model loaded.")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using template answers.")

    per_quality = num_samples // len(qualities)
    remainder = num_samples - per_quality * len(qualities)

    for qi, quality in enumerate(qualities):
        count = per_quality + (1 if qi < remainder else 0)
        for _ in range(count):
            domain = random.choice(DOMAINS)
            question = _sample_question(domain)

            if model is not None and tokenizer is not None:
                try:
                    import torch
                    prompt = f"Answer this interview question briefly:\n{question}\nAnswer:"
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                                             temperature=0.7, pad_token_id=tokenizer.eos_token_id)
                    new_toks = out[0][inputs["input_ids"].shape[1]:]
                    answer = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
                except Exception:
                    answer = _sample_answer(domain, quality)
            else:
                answer = _sample_answer(domain, quality)

            scores = _ground_truth_scores(quality)
            records.append({
                "question": question,
                "answer": answer,
                "domain": domain,
                "quality": quality,
                "ground_truth_scores": scores,
            })

    random.shuffle(records)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    logger.info(f"Generated {len(records)} calibration records → {output_path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="adaptive_interview_env/data/calibration_refs.json")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--model", default=None, help="HF model name for answer generation")
    args = parser.parse_args()
    generate(args.output, args.num_samples, args.model)
