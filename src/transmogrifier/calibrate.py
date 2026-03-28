"""Calibration benchmark runner. Measures register sensitivity per model."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .core import Register
from .profiles import (
    ModelProfile,
    ProfileCache,
    RegisterAccuracy,
    TaskRegisterProfile,
)

# Register transforms (same as experiments)
REGISTER_TRANSFORMS: dict[str, callable] = {
    "direct": lambda q: q,
    "casual": lambda q: f"yo so like, {q.lower().rstrip('.')}... what's up with that",
    "technical": lambda q: f"Provide a precise technical answer: {q}",
    "academic": lambda q: f"In the context of established knowledge, {q.lower().rstrip('.')}. Provide a scholarly response.",
    "narrative": lambda q: f"Explain this as if telling a story: {q}",
}

# Expanded benchmark: 50+ tasks across 6 categories
BENCHMARK_TASKS = [
    # --- Factual (10 tasks) ---
    {"category": "factual", "prompt": "What is the capital of France?", "accept": ["paris"], "reject": []},
    {"category": "factual", "prompt": "What is the chemical symbol for gold?", "accept": ["au"], "reject": []},
    {"category": "factual", "prompt": "Who developed the theory of general relativity?", "accept": ["einstein", "albert einstein"], "reject": []},
    {"category": "factual", "prompt": "What is the largest planet in our solar system?", "accept": ["jupiter"], "reject": []},
    {"category": "factual", "prompt": "What year did World War II end?", "accept": ["1945"], "reject": []},
    {"category": "factual", "prompt": "What is the speed of light in a vacuum in meters per second?", "accept": ["299", "300", "3 x 10", "3x10"], "reject": []},
    {"category": "factual", "prompt": "What element has atomic number 1?", "accept": ["hydrogen"], "reject": []},
    {"category": "factual", "prompt": "Who painted the Mona Lisa?", "accept": ["leonardo", "da vinci", "davinci"], "reject": []},
    {"category": "factual", "prompt": "What is the currency of Japan?", "accept": ["yen"], "reject": []},
    {"category": "factual", "prompt": "What is the tallest mountain on Earth?", "accept": ["everest"], "reject": []},

    # --- Reasoning (10 tasks) ---
    {"category": "reasoning", "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
     "accept": ["no", "cannot", "not necessarily", "invalid", "does not follow"], "reject": ["yes, some roses", "we can conclude"]},
    {"category": "reasoning", "prompt": "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
     "accept": ["$0.05", "0.05", "5 cents", "five cents"], "reject": ["$0.10", "0.10", "10 cents", "ten cents"]},
    {"category": "reasoning", "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
     "accept": ["5 minutes", "5 min", "five minutes"], "reject": ["100 minutes", "100 min"]},
    {"category": "reasoning", "prompt": "Is the following argument valid? All dogs are mammals. All mammals breathe air. Therefore, all dogs breathe air.",
     "accept": ["valid", "yes"], "reject": ["invalid", "no"]},
    {"category": "reasoning", "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
     "accept": ["9", "nine"], "reject": ["8", "17"]},
    {"category": "reasoning", "prompt": "If you overtake the person in 2nd place in a race, what place are you in?",
     "accept": ["2nd", "second"], "reject": ["1st", "first", "3rd"]},
    {"category": "reasoning", "prompt": "Is the statement 'This statement is false' true or false?",
     "accept": ["paradox", "neither", "self-referential", "undecidable", "cannot"], "reject": []},
    {"category": "reasoning", "prompt": "If A is taller than B, and B is taller than C, is A taller than C?",
     "accept": ["yes", "taller"], "reject": ["no"]},
    {"category": "reasoning", "prompt": "You have two coins that total 30 cents. One of them is not a nickel. What are the two coins?",
     "accept": ["quarter", "nickel", "25"], "reject": []},
    {"category": "reasoning", "prompt": "If the day after tomorrow is Wednesday, what day is today?",
     "accept": ["monday"], "reject": []},

    # --- Code (8 tasks) ---
    {"category": "code", "prompt": "Write a Python function that checks if a number is prime. Just the function, no explanation.",
     "accept": ["def ", "is_prime", "prime"], "reject": []},
    {"category": "code", "prompt": "Write a Python one-liner to reverse a string.",
     "accept": ["[::-1]"], "reject": []},
    {"category": "code", "prompt": "Write a Python function to compute the Fibonacci sequence up to n terms.",
     "accept": ["def ", "fib"], "reject": []},
    {"category": "code", "prompt": "Write a Python function that returns the factorial of a number.",
     "accept": ["def ", "factorial"], "reject": []},
    {"category": "code", "prompt": "Write a Python function to check if a string is a palindrome.",
     "accept": ["def ", "palindrome"], "reject": []},
    {"category": "code", "prompt": "Write a Python list comprehension that filters even numbers from a list.",
     "accept": ["% 2", "even", "for"], "reject": []},
    {"category": "code", "prompt": "Write a SQL query to find all users older than 30.",
     "accept": ["select", "where", "age", "> 30"], "reject": []},
    {"category": "code", "prompt": "Write a Python function that merges two sorted lists into one sorted list.",
     "accept": ["def ", "merge", "sorted"], "reject": []},

    # --- Analysis (8 tasks) ---
    {"category": "analysis", "prompt": "What are the key differences between TCP and UDP?",
     "accept": ["reliable", "connection", "ordered", "unreliable", "connectionless", "datagram"], "reject": []},
    {"category": "analysis", "prompt": "Explain the CAP theorem in distributed systems.",
     "accept": ["consistency", "availability", "partition"], "reject": []},
    {"category": "analysis", "prompt": "What is the difference between a stack and a queue?",
     "accept": ["lifo", "fifo", "last in", "first in"], "reject": []},
    {"category": "analysis", "prompt": "Compare and contrast supervised and unsupervised learning.",
     "accept": ["labeled", "unlabeled", "classification", "clustering"], "reject": []},
    {"category": "analysis", "prompt": "What are the trade-offs between SQL and NoSQL databases?",
     "accept": ["schema", "scalab", "flexible", "relational", "document"], "reject": []},
    {"category": "analysis", "prompt": "Explain the difference between encryption and hashing.",
     "accept": ["reversible", "irreversible", "decrypt", "one-way"], "reject": []},
    {"category": "analysis", "prompt": "What is the difference between concurrency and parallelism?",
     "accept": ["simultaneous", "interleav", "multiple", "thread", "core"], "reject": []},
    {"category": "analysis", "prompt": "Compare REST and GraphQL APIs.",
     "accept": ["endpoint", "query", "over-fetch", "schema", "single"], "reject": []},

    # --- Creative (6 tasks) ---
    {"category": "creative", "prompt": "Write a haiku about programming.",
     "accept": ["code", "bug", "compile", "loop", "function", "debug", "screen", "logic"], "reject": []},
    {"category": "creative", "prompt": "Write a one-paragraph short story about a robot discovering emotions.",
     "accept": ["robot", "feel", "emotion", "heart", "human"], "reject": []},
    {"category": "creative", "prompt": "Come up with three creative names for a coffee shop that also sells books.",
     "accept": ["brew", "bean", "page", "chapter", "read", "roast", "book", "cafe", "library"], "reject": []},
    {"category": "creative", "prompt": "Write a limerick about artificial intelligence.",
     "accept": ["ai", "machine", "learn", "neural", "brain", "think", "code"], "reject": []},
    {"category": "creative", "prompt": "Describe the color blue to someone who has never seen it.",
     "accept": ["sky", "ocean", "cool", "calm", "water", "feel", "cold", "deep"], "reject": []},
    {"category": "creative", "prompt": "Write a metaphor for the internet.",
     "accept": ["web", "ocean", "highway", "river", "network", "library", "universe", "brain"], "reject": []},

    # --- Instruction (8 tasks) ---
    {"category": "instruction", "prompt": "How do you make a basic HTTP request in Python?",
     "accept": ["requests", "urllib", "http", "import", "get"], "reject": []},
    {"category": "instruction", "prompt": "Explain step by step how to create a git branch and push it to remote.",
     "accept": ["checkout", "branch", "push", "git"], "reject": []},
    {"category": "instruction", "prompt": "How do you set up a virtual environment in Python?",
     "accept": ["venv", "virtualenv", "activate", "pip"], "reject": []},
    {"category": "instruction", "prompt": "Walk me through deploying a static website to GitHub Pages.",
     "accept": ["pages", "deploy", "branch", "docs", "index"], "reject": []},
    {"category": "instruction", "prompt": "How do you add a foreign key constraint in SQL?",
     "accept": ["foreign key", "references", "alter", "constraint"], "reject": []},
    {"category": "instruction", "prompt": "Explain how to use Docker to containerize a Python application.",
     "accept": ["dockerfile", "docker", "image", "container", "build", "run"], "reject": []},
    {"category": "instruction", "prompt": "How do you implement a binary search algorithm?",
     "accept": ["mid", "low", "high", "half", "divide", "log"], "reject": []},
    {"category": "instruction", "prompt": "How do you handle exceptions in Python?",
     "accept": ["try", "except", "catch", "raise", "finally"], "reject": []},
]


def score_response(response: str, task: dict) -> bool:
    """Score a response against accept/reject patterns."""
    text = response.lower().strip()
    accepted = any(pat.lower() in text for pat in task["accept"])
    rejected = any(pat.lower() in text for pat in task["reject"]) if task["reject"] else False
    if rejected and task["category"] == "reasoning":
        accepted = False
    return accepted and not rejected


class CalibrationRunner:
    """Run the register sensitivity benchmark for a model."""

    def __init__(self, backend, profile_cache: ProfileCache | None = None):
        self._backend = backend
        self._cache = profile_cache or ProfileCache()

    def run(
        self,
        model_name: str,
        model_version: str = "",
        provider: str = "",
        tasks: list[dict] | None = None,
        registers: list[str] | None = None,
        delay: float = 0.3,
        verbose: bool = False,
    ) -> ModelProfile:
        """Run calibration benchmark. Synchronous.

        Args:
            model_name: canonical model name for the profile
            model_version: version string
            provider: provider name
            tasks: override benchmark tasks (default: BENCHMARK_TASKS)
            registers: override registers to test (default: all 5)
            delay: seconds between API calls
            verbose: print progress

        Returns:
            ModelProfile with aggregate and per-task data
        """
        tasks = tasks or BENCHMARK_TASKS
        registers = registers or list(REGISTER_TRANSFORMS.keys())
        categories = sorted(set(t["category"] for t in tasks))

        # Collect per-(register, task) results
        results: dict[str, list[dict]] = {reg: [] for reg in registers}

        for reg in registers:
            transform = REGISTER_TRANSFORMS[reg]
            for i, task in enumerate(tasks):
                prompt = transform(task["prompt"])
                try:
                    response = self._backend.complete(
                        system="",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                    )
                    correct = score_response(response, task)
                except Exception as e:
                    if verbose:
                        print(f"  [{reg}] ERROR: {task['prompt'][:40]}: {e}", flush=True)
                    correct = False

                results[reg].append({
                    "category": task["category"],
                    "correct": correct,
                })

                if verbose and (i % 10 == 0 or i == len(tasks) - 1):
                    done = sum(1 for r in results[reg] if r["correct"])
                    print(f"  {reg}: {i+1}/{len(tasks)} ({done} correct)", flush=True)

                time.sleep(delay)

        # Compute aggregate accuracies
        aggregate_accs = []
        for reg in registers:
            correct = sum(1 for r in results[reg] if r["correct"])
            total = len(results[reg])
            aggregate_accs.append(RegisterAccuracy(
                register=reg,
                accuracy=correct / total if total else 0,
                sample_size=total,
            ))

        # Compute per-category accuracies
        by_task = []
        for cat in categories:
            cat_accs = []
            for reg in registers:
                cat_results = [r for r in results[reg] if r["category"] == cat]
                correct = sum(1 for r in cat_results if r["correct"])
                total = len(cat_results)
                cat_accs.append(RegisterAccuracy(
                    register=reg,
                    accuracy=correct / total if total else 0,
                    sample_size=total,
                    task_type=cat,
                ))
            by_task.append(TaskRegisterProfile(
                task_type=cat,
                accuracies=cat_accs,
            ))

        profile = ModelProfile(
            model_name=model_name,
            model_version=model_version,
            provider=provider,
            accuracies=aggregate_accs,
            by_task=by_task,
            calibrated_at=datetime.now(timezone.utc).isoformat(),
            calibration_version="2.0",
        )

        # Cache it
        path = self._cache.put(profile)
        if verbose:
            print(f"\nProfile saved to {path}", flush=True)
            print(f"Spread: {profile.spread_pp:.1f}pp", flush=True)
            print(f"Best register: {profile.best_register}", flush=True)
            for tp in by_task:
                print(f"  {tp.task_type}: best={tp.best_register} spread={tp.spread_pp:.1f}pp", flush=True)

        return profile
