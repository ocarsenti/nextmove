"""Context engine — detects context from user input and modulates axis weights."""

from __future__ import annotations

from ontology.models import Axis, AxisStatus

KNOWN_CONTEXTS: dict[str, dict] = {
    "startup": {
        "label": "Startup / Scale-up",
        "description": "Environnement early-stage à forte incertitude, rôles fluides, décisions rapides.",
        "keywords": ["startup", "scale-up", "scaleup", "seed", "serie a", "séries a", "fondateur", "founder", "cto", "ceo early"],
    },
    "corporate": {
        "label": "Grande entreprise / Corporate",
        "description": "Organisation structurée, hiérarchie établie, processus formalisés.",
        "keywords": ["corporate", "grande entreprise", "cac 40", "fortune 500", "multinational", "grand groupe"],
    },
    "consulting": {
        "label": "Conseil / Consulting",
        "description": "Missions courtes, clients multiples, livraison de recommandations, influence sans autorité.",
        "keywords": ["consulting", "conseil", "mckinsey", "bcg", "bain", "cabinet", "mission client"],
    },
    "research": {
        "label": "Recherche",
        "description": "Production de connaissances, cycles longs, rigueur méthodologique, incertitude épistémique.",
        "keywords": ["recherche", "research", "labo", "laboratoire", "phd", "doctorat", "académique", "academic", "cnrs", "inserm"],
    },
    "clinical_team": {
        "label": "Équipe clinique / Santé",
        "description": "Coordination interpersonnelle critique, protocoles stricts, sécurité patient prioritaire.",
        "keywords": ["clinique", "clinical", "hôpital", "hospital", "médecin", "infirmier", "soignant", "chirurgie", "urgences"],
    },
    "public_sector": {
        "label": "Secteur public",
        "description": "Processus formels, accountability collective, aversion au risque institutionnelle.",
        "keywords": ["public", "fonction publique", "état", "ministère", "collectivité", "administration", "fonctionnaire"],
    },
}


class ContextEngine:
    """Detects context and applies context-specific axis modulations."""

    def detect_context(self, text: str) -> str | None:
        """Detect context from free text (job description, note). Returns context_id or None."""
        text_lower = text.lower()
        for ctx_id, ctx in KNOWN_CONTEXTS.items():
            if any(kw in text_lower for kw in ctx["keywords"]):
                return ctx_id
        return None

    def apply_context(
        self, axes: list[Axis], context_id: str | None
    ) -> dict[str, tuple[float, AxisStatus]]:
        """
        Returns {axis_id: (effective_weight, effective_status)} after applying context rules.
        Clamps effective weight to [0.1, 2.0].
        """
        result: dict[str, tuple[float, AxisStatus]] = {}

        for ax in axes:
            weight = ax.weight
            status = ax.status

            if context_id:
                for rule in ax.rules:
                    if rule.requires_data:
                        continue  # V6+ only
                    cond = rule.condition
                    if cond.signal.value == "context_match" and cond.context == context_id:
                        eff = rule.effect
                        if eff.action.value == "reweight" and eff.weight_modifier is not None:
                            weight = max(0.1, min(2.0, weight + eff.weight_modifier))
                        elif eff.action.value == "mask":
                            status = AxisStatus.MASKED

            result[ax.id] = (weight, status)

        return result

    def context_label(self, context_id: str | None) -> str:
        if context_id and context_id in KNOWN_CONTEXTS:
            return KNOWN_CONTEXTS[context_id]["label"]
        return "Générique"
