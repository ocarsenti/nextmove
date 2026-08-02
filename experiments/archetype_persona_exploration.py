"""
Script exploratoire — NE FAIT PARTIE D'AUCUNE suite de tests (hors tests/,
donc jamais collecté par pytest/testpaths, voir pytest.ini).

Sert à tester rapidement le comportement de compute_archetype_distribution
(percentage ET display_percentage) sur des personas inventés, sans toucher
au produit ni au questionnaire réel. Modifier/ajouter des personas dans le
dict `personas` ci-dessous pour de nouveaux essais.

Usage : python3 experiments/archetype_persona_exploration.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ontology.models import AxisScore, AxisStatus, Profile
from ontology.registry import AxisRegistry
from engine.archetype import compute_archetype_distribution

# On charge le vrai registre d'axes du repo (pas un mock)
registry = AxisRegistry()

def make_profile(user_id: str, values: dict[str, float], confidence: float = 0.85, n_q: int = 3) -> Profile:
    """values: axis_id -> raw_value (0.0-1.0). Les axes non listés ne sont pas dans le profil (comme
    un questionnaire partiellement répondu / axe non actif)."""
    scores = {
        axis_id: AxisScore(
            axis_id=axis_id,
            raw_value=v,
            confidence=confidence,
            n_questions=n_q,
            variance=0.05,
            status=AxisStatus.ACTIVE,
        )
        for axis_id, v in values.items()
    }
    return Profile(user_id=user_id, axis_scores=scores)


# ---------------------------------------------------------------------
# 5 personas inventés — profils de réponses plausibles, pas de triche
# vers un archétype particulier (mélange volontaire sur certains axes)
# ---------------------------------------------------------------------

personas = {

    "P1_dev_produit_early_stage": dict(
        autonomy=0.85, direction=0.6, behavioral_stability=0.5,
        cognitive_flexibility=0.75, social_interaction=0.4, influence=0.5,
        situational_leadership=0.55, rigor=0.45, cognitive_structuring=0.35,
        ambiguity_tolerance=0.8, exploration=0.7, resilience=0.75,
        decision_speed=0.7, risk_appetite=0.65, persistence=0.8,
        value_orientation=0.7,
    ),

    "P2_analyste_reglementaire": dict(
        autonomy=0.55, direction=0.3, behavioral_stability=0.8,
        cognitive_flexibility=0.35, social_interaction=0.35, influence=0.25,
        situational_leadership=0.2, rigor=0.92, cognitive_structuring=0.88,
        ambiguity_tolerance=0.25, exploration=0.2, resilience=0.6,
        decision_speed=0.35, risk_appetite=0.15, persistence=0.85,
        value_orientation=0.5,
    ),

    "P3_manager_operationnel": dict(
        autonomy=0.5, direction=0.75, behavioral_stability=0.7,
        cognitive_flexibility=0.55, social_interaction=0.8, influence=0.78,
        situational_leadership=0.82, rigor=0.55, cognitive_structuring=0.5,
        ambiguity_tolerance=0.5, exploration=0.4, resilience=0.65,
        decision_speed=0.68, risk_appetite=0.45, persistence=0.6,
        value_orientation=0.6,
    ),

    "P4_consultant_junior_indecis": dict(
        # profil volontairement plat / mou sur presque tout -> cas limite
        autonomy=0.55, direction=0.5, behavioral_stability=0.5,
        cognitive_flexibility=0.55, social_interaction=0.5, influence=0.5,
        situational_leadership=0.48, rigor=0.5, cognitive_structuring=0.5,
        ambiguity_tolerance=0.5, exploration=0.52, resilience=0.5,
        decision_speed=0.5, risk_appetite=0.5, persistence=0.5,
        value_orientation=0.5,
    ),

    "P5_operateur_terrain_fiable": dict(
        autonomy=0.35, direction=0.25, behavioral_stability=0.88,
        cognitive_flexibility=0.3, social_interaction=0.65, influence=0.3,
        situational_leadership=0.35, rigor=0.75, cognitive_structuring=0.7,
        ambiguity_tolerance=0.2, exploration=0.15, resilience=0.8,
        decision_speed=0.4, risk_appetite=0.15, persistence=0.82,
        value_orientation=0.45,
    ),
}

print("=" * 100)
for pid, values in personas.items():
    profile = make_profile(pid, values)
    dist = compute_archetype_distribution(profile, registry)

    print(f"\n### {pid}")
    print(f"Résumé moteur : {dist.summary}")
    print(f"Display mode  : {dist.display_mode}")
    print(f"Display texte : {dist.display_summary}")
    print("Distribution complète :")
    for s in dist.scores:
        marker = " <== dominant" if s.name == dist.dominant else (" <== secondaire" if s.name == dist.secondary else "")
        print(f"  {s.name:10s} percentage={s.percentage:5.1f}%  display_percentage={s.display_percentage:5.1f}%  (confidence={s.confidence:.2f}){marker}")

    # Top 3 axes par raw_value (pour la ligne "environnement idéal")
    top_axes = sorted(values.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print("Top 3 axes (raw_value) :")
    for axis_id, v in top_axes:
        ax = registry.get(axis_id)
        label = ax.label if ax else axis_id
        print(f"  {label:30s} = {v:.2f}")

print("\n" + "=" * 100)
