"""
Custom stop words for clustering
"""

ACADEMIC_FRAMING = {
    "using",
    "use",
    "used",
    "based",
    "based on",
    "approach",
    "approaches",
    "method",
    "methods",
    "methodology",
    "study",
    "studies",
    "analysis",
    "analyses",
    "investigation",
    "investigations",
    "evaluation",
    "evaluations",
    "assessment",
    "assessments",
    "framework",
    "model",
    "models",
    "review",
    "overview",
    "towards",
    "via",
}

EFFECT_RESULT = {
    "effect",
    "effects",
    "impact",
    "impacts",
    "influence",
    "influences",
    "role",
    "roles",
    "relationship",
    "relationships",
    "association",
    "associations",
    "comparison",
    "comparisons",
}

DATA_MODELING = {
    "data",
    "dataset",
    "datasets",
    "measurement",
    "measurements",
    "simulation",
    "simulations",
    "experiment",
    "experiments",
    "experimental",
    "numerical",
    "computational",
    "modeling",
    "modelling",
}

APPLICATION_ORIENTED = {
    "application",
    "applications",
    "implementation",
    "implementations",
    "design",
    "designs",
    "development",
    "developments",
    "performance",
    "evaluation",
}

SUSTAINABILITY_POLICY = {
    "sustainable",
    "sustainability",
    "energy",
    "environmental",
    "environment",
    "green",
    "climate",
    "policy",
    "policies",
    "strategy",
    "strategies",
    "management",
}

SCOPE_SCALE = {
    "case",
    "cases",
    "system",
    "systems",
    "process",
    "processes",
}

DOMAIN_STOP_WORDS = (
    ACADEMIC_FRAMING
    | EFFECT_RESULT
    | DATA_MODELING
    | APPLICATION_ORIENTED
    | SUSTAINABILITY_POLICY
    | SCOPE_SCALE
)
