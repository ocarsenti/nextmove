"""
Behavioral questionnaire → 6-axis decision profile + archetype distribution.

V4 — 22 questions:
- 6 continuous axes × 3 cross-validating questions = 18
- Archetype × 4 behavioral vote questions = 4
- Display order interleaves axes to prevent pattern detection

Scoring:
- Continuous axes: each answer maps to 0.0 / 0.5 / 1.0.
  Per-axis score = mean, scaled to 1-5 integer.
- Archetype: 4 votes distributed across 5 archetypes.
  Output = distribution (each vote = 25 points, total = 100).
  NOT a single winner — the blend IS the signal.

Tie-breaking (deterministic, no randomness):
  When archetypes share the same vote count, we break ties using the
  user's continuous axis scores. Each archetype has a signature axis
  affinity — the weighted sum of the user's normalized axis scores for
  axes that characterize that archetype. If still tied, alphabetical
  order is the final fallback (Builder < Expert < Explorer < Leader < Operator).
"""
from collections import defaultdict


QUESTIONS = {
    # --- AXE 1: CHARGE OPÉRATIONNELLE (3 questions) ---
    "q1": {
        "text": "Tu choisis entre deux rythmes de travail :",
        "options": [
            {"label": "Semaine très intense (55–60h) puis périodes plus calmes", "value": "A"},
            {"label": "Charge stable (~40h/semaine) toute l'année", "value": "B"},
            {"label": "Charge fluctuante et imprévisible selon les projets", "value": "C"},
        ],
        "axis": "operational_load",
        "scoring": {"A": 1.0, "B": 0.0, "C": 0.5},
    },
    "q2": {
        "text": "On te propose un poste équivalent mais avec 25% de charge administrative en plus :",
        "options": [
            {"label": "J'accepte sans condition", "value": "A"},
            {"label": "J'accepte si c'est temporaire ou compensé", "value": "B"},
            {"label": "Je refuse si ça devient la norme", "value": "C"},
        ],
        "axis": "operational_load",
        "scoring": {"A": 1.0, "B": 0.5, "C": 0.0},
    },
    "q3": {
        "text": "Tu gères plusieurs sujets en parallèle :",
        "options": [
            {"label": "Tout gérer en parallèle même si c'est intense", "value": "A"},
            {"label": "Limiter à 2–3 sujets actifs", "value": "B"},
            {"label": "N'avoir qu'un sujet principal à la fois", "value": "C"},
        ],
        "axis": "operational_load",
        "scoring": {"A": 1.0, "B": 0.5, "C": 0.0},
    },

    # --- AXE 2: INCERTITUDE ENVIRONNEMENTALE (3 questions) ---
    "q4": {
        "text": "Deux postes identiques sauf la définition du scope :",
        "options": [
            {"label": "Scope clair dès le début", "value": "A"},
            {"label": "Scope partiellement défini", "value": "B"},
            {"label": "Scope à construire en avançant", "value": "C"},
        ],
        "axis": "uncertainty",
        "scoring": {"A": 0.0, "B": 0.5, "C": 1.0},
    },
    "q9": {
        "text": "Quand le périmètre de ton poste change souvent :",
        "options": [
            {"label": "Je m'adapte facilement", "value": "A"},
            {"label": "J'ai besoin de cycles stables entre les changements", "value": "B"},
            {"label": "Je préfère des cadres très stables", "value": "C"},
        ],
        "axis": "uncertainty",
        "scoring": {"A": 1.0, "B": 0.5, "C": 0.0},
    },
    "q15": {
        "text": "Quand la stratégie de ton équipe change souvent :",
        "options": [
            {"label": "Je m'adapte sans difficulté", "value": "A"},
            {"label": "Je préfère des cycles de planification plus longs", "value": "B"},
            {"label": "J'ai besoin d'une stabilité forte sur 12+ mois", "value": "C"},
        ],
        "axis": "uncertainty",
        "scoring": {"A": 1.0, "B": 0.5, "C": 0.0},
    },

    # --- AXE 3: AUTONOMIE DÉCISIONNELLE (3 questions) ---
    "q5": {
        "text": "Dans une décision critique de projet :",
        "options": [
            {"label": "Je décide seul(e) et j'assume", "value": "A"},
            {"label": "Je décide après validation rapide", "value": "B"},
            {"label": "Les décisions doivent être collectives", "value": "C"},
        ],
        "axis": "autonomy",
        "scoring": {"A": 1.0, "B": 0.5, "C": 0.0},
    },
    "q7": {
        "text": "Face à une situation urgente mais avec information incomplète :",
        "options": [
            {"label": "J'agis immédiatement avec ce que j'ai", "value": "A"},
            {"label": "J'attends des informations complémentaires", "value": "B"},
            {"label": "Je demande un arbitrage externe", "value": "C"},
        ],
        "axis": "autonomy",
        "scoring": {"A": 1.0, "B": 0.5, "C": 0.0},
    },
    "q16": {
        "text": "Quand un process bloque ton exécution :",
        "options": [
            {"label": "Je contourne pour avancer", "value": "A"},
            {"label": "Je remonte le blocage et j'attends l'ajustement", "value": "B"},
            {"label": "Je respecte strictement même si ça ralentit", "value": "C"},
        ],
        "axis": "autonomy",
        "scoring": {"A": 1.0, "B": 0.5, "C": 0.0},
    },

    # --- AXE 4: VÉLOCITÉ D'APPRENTISSAGE (3 questions) ---
    "q8": {
        "text": "Quand tu arrives dans une nouvelle mission :",
        "options": [
            {"label": "J'agis et j'ajuste en continu", "value": "A"},
            {"label": "J'observe puis j'exécute", "value": "B"},
            {"label": "J'analyse en profondeur avant d'agir", "value": "C"},
        ],
        "axis": "learning",
        "scoring": {"A": 1.0, "B": 0.5, "C": 0.0},
    },
    "q10": {
        "text": "Quand un nouveau domaine technique apparaît dans ton équipe :",
        "options": [
            {"label": "Autonomie rapide en moins de 2 semaines", "value": "A"},
            {"label": "Montée en compétence progressive sur 1 à 3 mois", "value": "B"},
            {"label": "Je préfère éviter les changements fréquents de domaine", "value": "C"},
        ],
        "axis": "learning",
        "scoring": {"A": 1.0, "B": 0.5, "C": 0.0},
    },
    "q11": {
        "text": "Quand tu dois monter en compétence rapidement :",
        "options": [
            {"label": "Immersion intense immédiate", "value": "A"},
            {"label": "Apprentissage progressif et structuré", "value": "B"},
            {"label": "Rythme stable sans pression d'apprentissage rapide", "value": "C"},
        ],
        "axis": "learning",
        "scoring": {"A": 1.0, "B": 0.5, "C": 0.0},
    },

    # --- AXE 5: OPTIONALITÉ STRATÉGIQUE (3 questions) ---
    "q6": {
        "text": "Quand les objectifs de ta mission sont flous :",
        "options": [
            {"label": "Je structure immédiatement un plan d'action", "value": "A"},
            {"label": "J'attends un minimum de direction avant d'agir", "value": "B"},
            {"label": "J'explore plusieurs directions en parallèle", "value": "C"},
        ],
        "axis": "optionality",
        "scoring": {"A": 0.5, "B": 0.0, "C": 1.0},
    },
    "q13": {
        "text": "Tu préfères une orientation de carrière :",
        "options": [
            {"label": "Progression linéaire dans ton domaine", "value": "A"},
            {"label": "Poste qui ouvre plusieurs trajectoires possibles", "value": "B"},
            {"label": "Spécialisation forte et durable", "value": "C"},
        ],
        "axis": "optionality",
        "scoring": {"A": 0.5, "B": 1.0, "C": 0.0},
    },
    "q14": {
        "text": "Tu choisis entre :",
        "options": [
            {"label": "Continuer dans ton domaine actuel", "value": "A"},
            {"label": "Évolution vers un domaine proche", "value": "B"},
            {"label": "Changement de domaine significatif", "value": "C"},
        ],
        "axis": "optionality",
        "scoring": {"A": 0.0, "B": 0.5, "C": 1.0},
    },

    # --- AXE 6: ARCHÉTYPE DE RÔLE (4 questions comportementales) ---
    "q12": {
        "text": "Le type de travail qui t'attire le plus :",
        "options": [
            {"label": "Construire un produit ou une solution de bout en bout", "value": "Builder"},
            {"label": "Optimiser des systèmes existants et les faire tourner", "value": "Operator"},
            {"label": "Résoudre un problème technique en profondeur", "value": "Expert"},
            {"label": "Coordonner une équipe vers un objectif commun", "value": "Leader"},
            {"label": "Explorer des terrains nouveaux et tester des idées", "value": "Explorer"},
        ],
        "axis": "archetype",
        "scoring": "vote",
    },
    "q20": {
        "text": "Quand un projet échoue, ton réflexe :",
        "options": [
            {"label": "Je reconstruis rapidement ce qui ne marche pas", "value": "Builder"},
            {"label": "Je restructure le plan et les priorités", "value": "Operator"},
            {"label": "J'analyse en profondeur la cause racine", "value": "Expert"},
            {"label": "Je rassemble l'équipe et je réaligne tout le monde", "value": "Leader"},
            {"label": "Je propose une approche radicalement différente", "value": "Explorer"},
        ],
        "axis": "archetype",
        "scoring": "vote",
    },
    "q21": {
        "text": "Ton équipe lance un nouveau projet. Tu te retrouves naturellement à :",
        "options": [
            {"label": "Construire le prototype ou la solution", "value": "Builder"},
            {"label": "Organiser l'exécution et les dépendances", "value": "Operator"},
            {"label": "Analyser et résoudre les problèmes complexes", "value": "Expert"},
            {"label": "Aligner les parties prenantes et donner la direction", "value": "Leader"},
            {"label": "Chercher un angle ou un marché que personne n'a vu", "value": "Explorer"},
        ],
        "axis": "archetype",
        "scoring": "vote",
    },
    "q22": {
        "text": "Face à deux jobs équivalents, tu choisis celui qui maximise :",
        "options": [
            {"label": "L'impact concret et tangible", "value": "Builder"},
            {"label": "La maîtrise opérationnelle et la stabilité", "value": "Operator"},
            {"label": "L'expertise et la profondeur technique", "value": "Expert"},
            {"label": "L'influence et le leadership", "value": "Leader"},
            {"label": "L'exploration et les terrains nouveaux", "value": "Explorer"},
        ],
        "axis": "archetype",
        "scoring": "vote",
    },

    # --- AXE 7: QUALITÉ RELATIONNELLE (3 questions) ---
    "q17": {
        "text": "Dans une équipe, tu préfères :",
        "options": [
            {"label": "Travail individuel avec syncs hebdomadaires", "value": "A"},
            {"label": "Mix autonomie et collaboration quotidienne", "value": "B"},
            {"label": "Collaboration continue et soutenue", "value": "C"},
        ],
        "axis": "relational",
        "scoring": {"A": 0.0, "B": 0.5, "C": 1.0},
    },
    "q18": {
        "text": "Ton quotidien de travail idéal :",
        "options": [
            {"label": "Autonomie forte avec points ponctuels", "value": "A"},
            {"label": "Collaboration régulière dans la journée", "value": "B"},
            {"label": "Travail très collectif en permanence", "value": "C"},
        ],
        "axis": "relational",
        "scoring": {"A": 0.0, "B": 0.5, "C": 1.0},
    },
    "q19": {
        "text": "Tu privilégies :",
        "options": [
            {"label": "Le projet avant tout, même si l'équipe est moyenne", "value": "A"},
            {"label": "Équilibre projet et équipe", "value": "B"},
            {"label": "L'équipe avant tout, même si le projet est moins intéressant", "value": "C"},
        ],
        "axis": "relational",
        "scoring": {"A": 0.0, "B": 0.5, "C": 1.0},
    },
}

DISPLAY_ORDER = [
    "q1",   # operational_load
    "q5",   # autonomy
    "q10",  # learning
    "q17",  # relational
    "q4",   # uncertainty
    "q13",  # optionality
    "q12",  # archétype
    "q2",   # operational_load
    "q7",   # autonomy
    "q9",   # uncertainty
    "q20",  # archétype
    "q8",   # learning
    "q18",  # relational
    "q6",   # optionality
    "q3",   # operational_load
    "q16",  # autonomy
    "q15",  # uncertainty
    "q11",  # learning
    "q14",  # optionality
    "q21",  # archétype
    "q19",  # relational
    "q22",  # archétype
]

AXIS_TO_PROFILE = {
    "operational_load": "operational_load",
    "uncertainty": "uncertainty",
    "autonomy": "autonomy",
    "learning": "learning",
    "optionality": "optionality",
    "relational": "relational",
}

ALL_ARCHETYPES = ["Builder", "Expert", "Operator", "Leader", "Explorer"]

# Each archetype's affinity to the 6 continuous axes.
# Weights reflect which behavioral axes most characterize that archetype.
# Used ONLY for deterministic tie-breaking when vote counts are equal.
ARCHETYPE_AXIS_AFFINITY = {
    "Builder":  {"autonomy": 0.3, "learning": 0.3, "operational_load": 0.2, "optionality": 0.1, "uncertainty": 0.1, "relational": 0.0},
    "Expert":   {"learning": 0.4, "autonomy": 0.2, "uncertainty": 0.0, "operational_load": 0.2, "optionality": 0.1, "relational": 0.1},
    "Operator": {"operational_load": 0.4, "relational": 0.2, "uncertainty": 0.0, "autonomy": 0.1, "learning": 0.2, "optionality": 0.1},
    "Leader":   {"relational": 0.4, "autonomy": 0.2, "uncertainty": 0.1, "operational_load": 0.1, "learning": 0.1, "optionality": 0.1},
    "Explorer": {"optionality": 0.3, "uncertainty": 0.3, "learning": 0.2, "autonomy": 0.1, "operational_load": 0.0, "relational": 0.1},
}

ALPHABETICAL_ORDER = {name: i for i, name in enumerate(sorted(ALL_ARCHETYPES))}


def score_to_scale5(score: float) -> int:
    return max(1, min(5, round(score * 4) + 1))


def _compute_axis_affinity(archetype: str, axis_raw_scores: dict[str, float]) -> float:
    """Weighted sum of normalized axis scores for a given archetype."""
    weights = ARCHETYPE_AXIS_AFFINITY[archetype]
    return sum(weights[axis] * axis_raw_scores.get(axis, 0.5) for axis in weights)


def _rank_archetypes(
    archetype_votes: dict[str, int],
    axis_raw_scores: dict[str, float],
) -> tuple[list[dict], str]:
    """
    Deterministic ranking of archetypes.
    Returns (sorted_list, tie_break_reason).

    Sorting key (descending priority):
      1. Vote count (higher wins)
      2. Axis affinity score (higher wins) — uses the user's own axis
         answers to differentiate archetypes that got equal votes
      3. Alphabetical name (A < Z) — guaranteed unique, final fallback
    """
    entries = []
    for name in ALL_ARCHETYPES:
        votes = archetype_votes.get(name, 0)
        affinity = _compute_axis_affinity(name, axis_raw_scores)
        entries.append({
            "name": name,
            "votes": votes,
            "affinity": round(affinity, 6),
            "alpha": ALPHABETICAL_ORDER[name],
        })

    entries.sort(key=lambda e: (-e["votes"], -e["affinity"], e["alpha"]))

    winner = entries[0]
    runner_up = entries[1]
    if winner["votes"] > runner_up["votes"]:
        reason = "clear_winner"
    elif abs(winner["affinity"] - runner_up["affinity"]) > 1e-9:
        reason = "axis_affinity"
    else:
        reason = "alphabetical"

    return entries, reason


def compute_profile_vector(answers: dict[str, str]) -> dict:
    axis_scores: dict[str, list[float]] = defaultdict(list)
    archetype_votes: dict[str, int] = defaultdict(int)

    for qid, answer in answers.items():
        q = QUESTIONS.get(qid)
        if not q:
            continue
        if q["axis"] == "archetype":
            archetype_votes[answer] += 1
        else:
            score_map = q["scoring"]
            if answer in score_map:
                axis_scores[q["axis"]].append(score_map[answer])

    result = {}
    axis_raw_means: dict[str, float] = {}
    for axis_key, profile_field in AXIS_TO_PROFILE.items():
        scores = axis_scores.get(axis_key, [])
        raw = sum(scores) / len(scores) if scores else 0.5
        axis_raw_means[axis_key] = raw
        result[profile_field] = score_to_scale5(raw)

    ranked, tie_break_reason = _rank_archetypes(archetype_votes, axis_raw_means)

    archetype_distribution = {}
    for entry in ranked:
        archetype_distribution[entry["name"]] = entry["votes"] * 25
    result["archetype"] = archetype_distribution
    result["archetype_ranking"] = [
        {"name": e["name"], "votes": e["votes"], "affinity": e["affinity"]}
        for e in ranked
    ]
    result["winner_archetype"] = ranked[0]["name"]
    result["tie_break_reason"] = tie_break_reason

    return result


def get_ordered_questions() -> list[dict]:
    result = []
    for qid in DISPLAY_ORDER:
        q = QUESTIONS[qid]
        result.append({"id": qid, **q})
    return result
