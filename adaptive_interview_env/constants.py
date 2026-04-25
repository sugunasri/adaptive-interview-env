SKILL_DIMENSIONS = [
    "correctness",
    "edge_case_coverage",
    "complexity_analysis",
    "tradeoff_reasoning",
    "communication_clarity",
]

# V1 domains
DOMAINS_V1 = ["algorithms", "system_design", "databases", "concurrency"]

# V2: expanded to 8 domains
DOMAINS = [
    "algorithms",
    "system_design",
    "databases",
    "concurrency",
    "machine_learning",
    "distributed_systems",
    "security",
    "object_oriented_design",
]

# V2: difficulty levels and score thresholds
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
DIFFICULTY_THRESHOLDS = {"medium": 0.6, "hard": 0.8}

# V2: student ability levels
ABILITY_LEVELS = ["weak", "average", "strong"]

MAX_STEPS = 20
EMA_DECAY_DEFAULT = 0.8
