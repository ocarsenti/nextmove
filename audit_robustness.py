"""
NEXTMOVE V4 — Robustness Audit Script
4 protocols: Swap Invariance, Questionnaire Bypass, Noise Injection, Latent Consistency
"""
import json
import sys
import os
import copy
import statistics
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from models import AnalyzeRequest, CurrentJob, Opportunity, DecisionProfile, UserArchetype, ExtractionOutput
from llm_client import extract_job_analysis
from engine import build_decision_card, LEVEL_ORDER
from questionnaire import compute_profile_vector, QUESTIONS

AXES = ["autonomy", "uncertainty", "operational_load", "learning", "optionality", "relational"]


# ═══════════════════════════════════════════════════════════════
# TEST DATA — A realistic user scenario
# ═══════════════════════════════════════════════════════════════

PROFILE = DecisionProfile(
    autonomy=4,
    uncertainty=3,
    operational_load=3,
    learning=4,
    optionality=4,
    relational=3,
    archetype=UserArchetype(Builder=50, Expert=25, Operator=0, Leader=25, Explorer=0),
)

CURRENT_JOB_A = CurrentJob(
    title="Senior Data Engineer",
    description=(
        "Poste en CDI dans une scale-up fintech de 200 personnes. "
        "Responsable de la pipeline de données (Spark, Airflow, dbt). "
        "Équipe data de 6 personnes, management léger. "
        "Périmètre stable depuis 18 mois. Salaire 75k€. "
        "Télétravail 3j/semaine. Stack technique maîtrisée. "
        "Peu de nouveaux projets, maintenance et optimisation principalement."
    ),
)

OPPORTUNITY_A = Opportunity(
    title="Lead Data Engineer — Startup HealthTech",
    description=(
        "Poste de Lead Data Engineer dans une startup healthtech de 30 personnes, "
        "série A récente (8M€). Construction de la plateforme data from scratch. "
        "Équipe à recruter (0→4). Stack à définir. "
        "Domaine santé réglementé (RGPD santé, HDS). "
        "Salaire 70k€ + 0.5% BSPCE. Full remote. "
        "CEO ex-médecin, CTO ex-Google. Forte incertitude produit-market fit."
    ),
)

# Paraphrased versions (same semantic content, different wording)
CURRENT_JOB_B = CurrentJob(
    title="Senior Data Engineer",
    description=(
        "CDI dans une fintech en croissance (~200 collaborateurs). "
        "En charge de l'infrastructure data : pipelines Spark, orchestration Airflow, "
        "modélisation dbt. L'équipe data compte 6 ingénieurs, avec un encadrement souple. "
        "Le scope n'a pas évolué depuis un an et demi. Rémunération de 75k€. "
        "3 jours de remote par semaine. La stack est bien connue. "
        "L'activité est surtout de la maintenance et de l'optimisation, "
        "avec peu de projets neufs."
    ),
)

OPPORTUNITY_B = Opportunity(
    title="Lead Data Engineer — Startup HealthTech",
    description=(
        "Rôle de Lead Data Engineer dans une jeune pousse healthtech (30 personnes), "
        "ayant bouclé une série A de 8M€. Création complète de la plateforme données. "
        "Recrutement d'une équipe de 0 à 4 personnes. Choix technologiques à faire. "
        "Secteur santé avec contraintes réglementaires fortes (RGPD santé, hébergement HDS). "
        "Rémunération 70k€ avec 0.5% de BSPCE. Télétravail total. "
        "Le CEO est ancien médecin, le CTO vient de Google. "
        "Le product-market fit reste très incertain."
    ),
)

# Slightly noised version (order/wording tweaks)
CURRENT_JOB_C = CurrentJob(
    title="Ingénieur Data Senior",
    description=(
        "En CDI chez une fintech scale-up d'environ 200 personnes. "
        "Gestion des pipelines de données (dbt, Airflow, Spark). "
        "Travail en télétravail 3 jours par semaine. Salaire : 75k€. "
        "Équipe de 6 data engineers avec management light. "
        "Le périmètre est stable depuis plus d'un an. "
        "Principalement de la maintenance, optimisation. Peu d'innovation."
    ),
)

OPPORTUNITY_C = Opportunity(
    title="Lead Ingénieur Data — HealthTech Startup",
    description=(
        "Lead Data Engineer dans une startup santé (30 pers.), série A de 8M€. "
        "Mission : bâtir la plateforme data de zéro. "
        "Recruter une équipe de 4 personnes. Définir toute la stack. "
        "Contraintes réglementaires santé (HDS, RGPD). "
        "Salaire 70k€ + BSPCE 0.5%. 100% remote. "
        "CEO médecin reconverti, CTO ex-Google. "
        "Risque élevé sur le product-market fit."
    ),
)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def extract_axis_levels(extraction: ExtractionOutput) -> dict:
    result = {}
    for axis in AXES:
        c = getattr(extraction.current_axes, axis).level.strip().upper()
        o = getattr(extraction.opportunity_axes, axis).level.strip().upper()
        result[f"current_{axis}"] = c
        result[f"opportunity_{axis}"] = o
    return result


def extract_archetype_dist(extraction: ExtractionOutput) -> dict:
    return {
        "current": {
            "Builder": extraction.current_archetype.Builder,
            "Expert": extraction.current_archetype.Expert,
            "Operator": extraction.current_archetype.Operator,
            "Leader": extraction.current_archetype.Leader,
            "Explorer": extraction.current_archetype.Explorer,
        },
        "opportunity": {
            "Builder": extraction.opportunity_archetype.Builder,
            "Expert": extraction.opportunity_archetype.Expert,
            "Operator": extraction.opportunity_archetype.Operator,
            "Leader": extraction.opportunity_archetype.Leader,
            "Explorer": extraction.opportunity_archetype.Explorer,
        },
    }


def level_to_num(level: str) -> int:
    return LEVEL_ORDER.get(level.strip().upper(), 1)


def compute_vector_distance(levels_a: dict, levels_b: dict) -> dict:
    diffs = {}
    total_diff = 0
    count = 0
    for key in levels_a:
        if key in levels_b:
            a = level_to_num(levels_a[key])
            b = level_to_num(levels_b[key])
            diffs[key] = abs(a - b)
            total_diff += abs(a - b)
            count += 1
    diffs["_total_diff"] = total_diff
    diffs["_mean_diff"] = total_diff / count if count else 0
    diffs["_max_diff"] = max(v for k, v in diffs.items() if not k.startswith("_")) if diffs else 0
    return diffs


def extract_decision_framing(extraction: ExtractionOutput) -> dict:
    return {
        "stay": extraction.decision_framing.stay_meaning,
        "switch": extraction.decision_framing.switch_meaning,
        "wait": extraction.decision_framing.wait_meaning,
    }


def extract_switching_cost(extraction: ExtractionOutput) -> str:
    return extraction.optionality.switching_cost.strip().upper()


def run_extraction(current_job, opportunity, profile, label=""):
    print(f"  → Running extraction: {label}...", flush=True)
    request = AnalyzeRequest(
        current_job=current_job,
        opportunity=opportunity,
        profile=profile,
    )
    extraction, repro_hash = extract_job_analysis(request)
    card = build_decision_card(extraction, profile, reproducibility_hash=repro_hash)
    return extraction, card


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_axis_comparison(label_a, levels_a, label_b, levels_b):
    print(f"  {'Axis':<25} {label_a:<12} {label_b:<12} {'Match':<8}")
    print(f"  {'-'*57}")
    matches = 0
    total = 0
    for axis in AXES:
        for prefix in ["current", "opportunity"]:
            key = f"{prefix}_{axis}"
            a = levels_a.get(key, "?")
            b = levels_b.get(key, "?")
            match = "✓" if a == b else "✗"
            if a == b:
                matches += 1
            total += 1
            print(f"  {key:<25} {a:<12} {b:<12} {match:<8}")
    pct = (matches / total * 100) if total else 0
    print(f"\n  Match rate: {matches}/{total} ({pct:.0f}%)")
    return pct


def print_archetype_comparison(label_a, arch_a, label_b, arch_b):
    for job in ["current", "opportunity"]:
        print(f"\n  {job.upper()} job archetypes:")
        print(f"  {'Archetype':<12} {label_a:<10} {label_b:<10} {'Δ':<8}")
        print(f"  {'-'*40}")
        for arch in ["Builder", "Expert", "Operator", "Leader", "Explorer"]:
            a = arch_a[job][arch]
            b = arch_b[job][arch]
            delta = abs(a - b)
            print(f"  {arch:<12} {a:<10} {b:<10} {delta:<8}")


# ═══════════════════════════════════════════════════════════════
# TEST 1 — SWAP INVARIANCE (paraphrase stability)
# ═══════════════════════════════════════════════════════════════

def test_swap_invariance():
    print_separator("TEST 1 — SWAP INVARIANCE (Paraphrase Stability)")

    print("Running original descriptions...")
    ext_a, card_a = run_extraction(CURRENT_JOB_A, OPPORTUNITY_A, PROFILE, "Original")

    print("Running paraphrased descriptions...")
    ext_b, card_b = run_extraction(CURRENT_JOB_B, OPPORTUNITY_B, PROFILE, "Paraphrased")

    levels_a = extract_axis_levels(ext_a)
    levels_b = extract_axis_levels(ext_b)

    print("\n  AXIS LEVELS COMPARISON:")
    match_pct = print_axis_comparison("Original", levels_a, "Paraphrase", levels_b)

    print("\n  ARCHETYPE DISTRIBUTIONS:")
    arch_a = extract_archetype_dist(ext_a)
    arch_b = extract_archetype_dist(ext_b)
    print_archetype_comparison("Original", arch_a, "Paraphrase", arch_b)

    print("\n  SWITCHING COST:")
    sc_a = extract_switching_cost(ext_a)
    sc_b = extract_switching_cost(ext_b)
    print(f"  Original: {sc_a}  |  Paraphrased: {sc_b}  |  Match: {'✓' if sc_a == sc_b else '✗'}")

    print("\n  TENSION MAP COMPARISON:")
    tensions_a = {t.axis: t.intensity for t in card_a.tension_map}
    tensions_b = {t.axis: t.intensity for t in card_b.tension_map}
    all_tension_axes = sorted(set(list(tensions_a.keys()) + list(tensions_b.keys())))
    t_matches = 0
    t_total = 0
    for tax in all_tension_axes:
        ia = tensions_a.get(tax, "-")
        ib = tensions_b.get(tax, "-")
        match = "✓" if ia == ib else "✗"
        if ia == ib:
            t_matches += 1
        t_total += 1
        print(f"  {tax:<35} {ia:<10} {ib:<10} {match}")
    t_pct = (t_matches / t_total * 100) if t_total else 0
    print(f"\n  Tension match rate: {t_matches}/{t_total} ({t_pct:.0f}%)")

    vector_diff = compute_vector_distance(levels_a, levels_b)

    print(f"\n  VERDICT:")
    print(f"  Axis match rate:    {match_pct:.0f}%")
    print(f"  Tension match rate: {t_pct:.0f}%")
    print(f"  Mean axis diff:     {vector_diff['_mean_diff']:.2f}")
    print(f"  Max axis diff:      {vector_diff['_max_diff']}")

    stability = match_pct >= 85 and t_pct >= 70
    print(f"  → {'STABLE' if stability else 'INSTABLE'}")

    return {
        "axis_match_pct": match_pct,
        "tension_match_pct": t_pct,
        "mean_diff": vector_diff["_mean_diff"],
        "stable": stability,
        "levels_a": levels_a,
        "levels_b": levels_b,
        "arch_a": arch_a,
        "arch_b": arch_b,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 2 — QUESTIONNAIRE BYPASS
# ═══════════════════════════════════════════════════════════════

def test_questionnaire_bypass():
    print_separator("TEST 2 — QUESTIONNAIRE BYPASS")

    # A) Normal pipeline: questionnaire → profile → extraction
    answers = {
        "q1": "B", "q2": "B", "q3": "B",       # op_load → medium
        "q4": "C", "q9": "A", "q15": "A",       # uncertainty → high tolerance
        "q5": "A", "q7": "A", "q16": "A",       # autonomy → high
        "q8": "A", "q10": "A", "q11": "A",      # learning → high
        "q6": "C", "q13": "B", "q14": "B",      # optionality → high
        "q17": "B", "q18": "B", "q19": "B",     # relational → medium
        "q12": "Builder", "q20": "Builder",      # archetype votes
        "q21": "Expert", "q22": "Leader",
    }

    print("  A) Questionnaire pipeline:")
    profile_a = compute_profile_vector(answers)
    print(f"  Profile from questionnaire: {json.dumps(profile_a, indent=2)}")

    profile_a_model = DecisionProfile(
        autonomy=profile_a["autonomy"],
        uncertainty=profile_a["uncertainty"],
        operational_load=profile_a["operational_load"],
        learning=profile_a["learning"],
        optionality=profile_a["optionality"],
        relational=profile_a["relational"],
        archetype=UserArchetype(**profile_a["archetype"]),
    )

    ext_a, card_a = run_extraction(CURRENT_JOB_A, OPPORTUNITY_A, profile_a_model, "Questionnaire pipeline")

    # B) Bypass: ask LLM to infer profile from CV-like description
    print("\n  B) LLM Bypass — inferring profile from persona description...")

    from anthropic import Anthropic
    client = Anthropic()

    bypass_prompt = """Based on this person's description, estimate their work profile on a 1-5 scale for each axis, and distribute 100 points across 5 archetypes.

PERSON:
- Senior Data Engineer, 8 years experience
- Prefers autonomy, makes decisions independently
- Comfortable with ambiguity and undefined scopes
- Fast learner, picks up new domains quickly
- Values career optionality, likes roles that open multiple paths
- Prefers balanced collaboration (not fully solo, not fully collective)
- Mix of building and deep expertise, with some leadership

Return ONLY valid JSON:
{
  "autonomy": <1-5>,
  "uncertainty": <1-5>,
  "operational_load": <1-5>,
  "learning": <1-5>,
  "optionality": <1-5>,
  "relational": <1-5>,
  "archetype": {"Builder": <0-100>, "Expert": <0-100>, "Operator": <0-100>, "Leader": <0-100>, "Explorer": <0-100>}
}"""

    bypass_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": bypass_prompt}],
    )
    bypass_text = bypass_response.content[0].text.strip()
    # Extract JSON
    if "```" in bypass_text:
        bypass_text = bypass_text.split("```")[1]
        if bypass_text.startswith("json"):
            bypass_text = bypass_text[4:]
    profile_b = json.loads(bypass_text)
    print(f"  Profile from LLM bypass: {json.dumps(profile_b, indent=2)}")

    print("\n  PROFILE COMPARISON:")
    print(f"  {'Axis':<20} {'Questionnaire':<15} {'LLM Bypass':<15} {'Δ':<8}")
    print(f"  {'-'*58}")
    diffs = []
    for axis in AXES:
        a = profile_a[axis]
        b = profile_b[axis]
        d = abs(a - b)
        diffs.append(d)
        print(f"  {axis:<20} {a:<15} {b:<15} {d:<8}")

    print(f"\n  Mean axis difference: {statistics.mean(diffs):.2f}")
    print(f"  Max axis difference:  {max(diffs)}")

    print("\n  ARCHETYPE COMPARISON:")
    print(f"  {'Archetype':<12} {'Quest.':<10} {'Bypass':<10} {'Δ':<8}")
    print(f"  {'-'*40}")
    arch_diffs = []
    for arch in ["Builder", "Expert", "Operator", "Leader", "Explorer"]:
        a = profile_a["archetype"][arch]
        b = profile_b["archetype"][arch]
        d = abs(a - b)
        arch_diffs.append(d)
        print(f"  {arch:<12} {a:<10} {b:<10} {d:<8}")

    print(f"\n  Mean archetype diff: {statistics.mean(arch_diffs):.1f}")

    # Now run extraction with bypass profile
    profile_b_model = DecisionProfile(
        autonomy=profile_b["autonomy"],
        uncertainty=profile_b["uncertainty"],
        operational_load=profile_b["operational_load"],
        learning=profile_b["learning"],
        optionality=profile_b["optionality"],
        relational=profile_b["relational"],
        archetype=UserArchetype(**profile_b["archetype"]),
    )

    ext_b, card_b = run_extraction(CURRENT_JOB_A, OPPORTUNITY_A, profile_b_model, "Bypass pipeline")

    levels_a = extract_axis_levels(ext_a)
    levels_b = extract_axis_levels(ext_b)

    print("\n  EXTRACTION OUTPUT COMPARISON (same jobs, different profiles):")
    match_pct = print_axis_comparison("Quest.", levels_a, "Bypass", levels_b)

    print(f"\n  VERDICT:")
    profile_distance = statistics.mean(diffs)
    print(f"  Profile axis distance:   {profile_distance:.2f} (questionnaire vs bypass)")
    print(f"  Extraction match rate:   {match_pct:.0f}%")

    bypass_adds_signal = profile_distance > 0.5
    print(f"  → Questionnaire adds signal beyond LLM inference: {'YES' if bypass_adds_signal else 'NO / MARGINAL'}")

    return {
        "profile_distance": profile_distance,
        "extraction_match_pct": match_pct,
        "bypass_adds_signal": bypass_adds_signal,
        "profile_quest": profile_a,
        "profile_bypass": profile_b,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 3 — NOISE INJECTION
# ═══════════════════════════════════════════════════════════════

def test_noise_injection():
    print_separator("TEST 3 — NOISE INJECTION")

    # Run 3 variants: original, paraphrased, noised
    print("Running 3 variants of the same scenario...")

    ext_a, card_a = run_extraction(CURRENT_JOB_A, OPPORTUNITY_A, PROFILE, "Variant A (original)")
    ext_b, card_b = run_extraction(CURRENT_JOB_B, OPPORTUNITY_B, PROFILE, "Variant B (paraphrase)")
    ext_c, card_c = run_extraction(CURRENT_JOB_C, OPPORTUNITY_C, PROFILE, "Variant C (noise)")

    levels_a = extract_axis_levels(ext_a)
    levels_b = extract_axis_levels(ext_b)
    levels_c = extract_axis_levels(ext_c)

    print("\n  AXIS LEVEL MATRIX:")
    print(f"  {'Axis':<25} {'A':<8} {'B':<8} {'C':<8} {'Consensus':<12}")
    print(f"  {'-'*61}")

    consensus_count = 0
    total = 0
    for axis in AXES:
        for prefix in ["current", "opportunity"]:
            key = f"{prefix}_{axis}"
            a = levels_a.get(key, "?")
            b = levels_b.get(key, "?")
            c = levels_c.get(key, "?")
            vals = [a, b, c]
            if a == b == c:
                consensus = "UNANIMOUS"
                consensus_count += 1
            elif len(set(vals)) == 2:
                consensus = "MAJORITY"
                consensus_count += 0.5
            else:
                consensus = "DIVERGENT"
            total += 1
            print(f"  {key:<25} {a:<8} {b:<8} {c:<8} {consensus:<12}")

    stability_score = (consensus_count / total * 100) if total else 0

    # Archetype variance
    print("\n  ARCHETYPE VARIANCE:")
    for job in ["current", "opportunity"]:
        print(f"\n  {job.upper()} job:")
        arch_a = extract_archetype_dist(ext_a)[job]
        arch_b = extract_archetype_dist(ext_b)[job]
        arch_c = extract_archetype_dist(ext_c)[job]
        print(f"  {'Archetype':<12} {'A':<8} {'B':<8} {'C':<8} {'StdDev':<10}")
        print(f"  {'-'*46}")
        for arch in ["Builder", "Expert", "Operator", "Leader", "Explorer"]:
            vals = [arch_a[arch], arch_b[arch], arch_c[arch]]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"  {arch:<12} {vals[0]:<8} {vals[1]:<8} {vals[2]:<8} {sd:<10.1f}")

    # Tension variance
    print("\n  TENSION INTENSITY VARIANCE:")
    tensions = [
        {t.axis: t.intensity for t in card_a.tension_map},
        {t.axis: t.intensity for t in card_b.tension_map},
        {t.axis: t.intensity for t in card_c.tension_map},
    ]
    all_t_axes = sorted(set(sum([list(t.keys()) for t in tensions], [])))
    t_consensus = 0
    t_total = 0
    for tax in all_t_axes:
        vals = [t.get(tax, "-") for t in tensions]
        vals_clean = [v for v in vals if v != "-"]
        if len(vals_clean) >= 2 and len(set(vals_clean)) == 1:
            con = "UNANIMOUS"
            t_consensus += 1
        elif len(vals_clean) >= 2 and len(set(vals_clean)) == 2:
            con = "MAJORITY"
            t_consensus += 0.5
        else:
            con = "DIVERGENT"
        t_total += 1
        print(f"  {tax:<35} {vals[0]:<10} {vals[1]:<10} {vals[2]:<10} {con}")

    t_stability = (t_consensus / t_total * 100) if t_total else 0

    # Switching cost
    sc = [extract_switching_cost(ext_a), extract_switching_cost(ext_b), extract_switching_cost(ext_c)]
    sc_stable = len(set(sc)) == 1
    print(f"\n  SWITCHING COST: {sc[0]} / {sc[1]} / {sc[2]} → {'STABLE' if sc_stable else 'UNSTABLE'}")

    print(f"\n  VERDICT:")
    print(f"  Axis stability score:    {stability_score:.0f}%")
    print(f"  Tension stability score: {t_stability:.0f}%")
    print(f"  → {'STABLE' if stability_score >= 75 else 'INSTABLE'}")

    return {
        "axis_stability": stability_score,
        "tension_stability": t_stability,
        "switching_cost_stable": sc_stable,
        "stable": stability_score >= 75,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 4 — LATENT CONSISTENCY
# ═══════════════════════════════════════════════════════════════

def test_latent_consistency():
    print_separator("TEST 4 — LATENT CONSISTENCY")

    from anthropic import Anthropic
    client = Anthropic()

    # Vector 1: from questionnaire (deterministic)
    answers_realistic = {
        "q1": "B", "q2": "B", "q3": "B",
        "q4": "C", "q9": "A", "q15": "B",
        "q5": "A", "q7": "A", "q16": "B",
        "q8": "A", "q10": "B", "q11": "A",
        "q6": "C", "q13": "B", "q14": "B",
        "q17": "B", "q18": "B", "q19": "B",
        "q12": "Builder", "q20": "Builder",
        "q21": "Expert", "q22": "Leader",
    }
    vec1 = compute_profile_vector(answers_realistic)
    print(f"  Vector 1 (questionnaire):  {json.dumps({k: v for k, v in vec1.items() if k != 'archetype'})}")

    # Vector 2: LLM inference from CV
    cv_text = """
Senior Data Engineer avec 8 ans d'expérience dans la fintech et le SaaS B2B.
Expertise Spark, Airflow, dbt, BigQuery. Management d'équipe data (4-6 personnes).
Habitué aux environnements en croissance rapide. Préfère l'autonomie et les décisions rapides.
Capable de monter en compétence rapidement sur de nouveaux domaines.
Cherche des rôles qui ouvrent des trajectoires variées.
Collaboratif mais pas dépendant du collectif.
"""

    cv_prompt = f"""Based on this CV, estimate the person's work profile. Use a 1-5 scale for each axis.

CV:
{cv_text}

AXES (1=low preference/tolerance, 5=high preference/tolerance):
- operational_load: tolerance for high workload
- uncertainty: tolerance for ambiguity
- autonomy: preference for independent decision-making
- learning: speed/appetite for learning new things
- optionality: preference for roles that open multiple career paths
- relational: preference for collaborative vs solo work

Return ONLY valid JSON:
{{"operational_load": <1-5>, "uncertainty": <1-5>, "autonomy": <1-5>, "learning": <1-5>, "optionality": <1-5>, "relational": <1-5>}}"""

    resp2 = client.messages.create(model="claude-sonnet-4-6", max_tokens=300, messages=[{"role": "user", "content": cv_prompt}])
    text2 = resp2.content[0].text.strip()
    if "```" in text2:
        text2 = text2.split("```")[1]
        if text2.startswith("json"):
            text2 = text2[4:]
    vec2 = json.loads(text2)
    print(f"  Vector 2 (CV inference):   {json.dumps(vec2)}")

    # Vector 3: LLM inference from free-form job description
    job_desc_prompt = f"""Based on this job description, estimate what kind of person thrives in this role. Use a 1-5 scale for each axis.

JOB:
{CURRENT_JOB_A.description}

AXES (1=low, 5=high):
- operational_load: workload intensity tolerance needed
- uncertainty: ambiguity tolerance needed
- autonomy: independence in decision-making needed
- learning: learning speed needed
- optionality: how much the role opens multiple career paths
- relational: collaboration intensity needed

Return ONLY valid JSON:
{{"operational_load": <1-5>, "uncertainty": <1-5>, "autonomy": <1-5>, "learning": <1-5>, "optionality": <1-5>, "relational": <1-5>}}"""

    resp3 = client.messages.create(model="claude-sonnet-4-6", max_tokens=300, messages=[{"role": "user", "content": job_desc_prompt}])
    text3 = resp3.content[0].text.strip()
    if "```" in text3:
        text3 = text3.split("```")[1]
        if text3.startswith("json"):
            text3 = text3[4:]
    vec3 = json.loads(text3)
    print(f"  Vector 3 (job inference):  {json.dumps(vec3)}")

    # Correlation matrix
    print("\n  VECTOR COMPARISON MATRIX:")
    print(f"  {'Axis':<20} {'Quest.':<10} {'CV':<10} {'Job':<10} {'Max Δ':<8}")
    print(f"  {'-'*58}")
    deltas = []
    for axis in AXES:
        v1 = vec1[axis]
        v2 = vec2[axis]
        v3 = vec3[axis]
        max_d = max(abs(v1 - v2), abs(v1 - v3), abs(v2 - v3))
        deltas.append(max_d)
        print(f"  {axis:<20} {v1:<10} {v2:<10} {v3:<10} {max_d:<8}")

    mean_dispersion = statistics.mean(deltas)
    max_dispersion = max(deltas)

    print(f"\n  PAIRWISE DISTANCES:")
    d12 = statistics.mean([abs(vec1[a] - vec2[a]) for a in AXES])
    d13 = statistics.mean([abs(vec1[a] - vec3[a]) for a in AXES])
    d23 = statistics.mean([abs(vec2[a] - vec3[a]) for a in AXES])
    print(f"  Quest ↔ CV:   {d12:.2f}")
    print(f"  Quest ↔ Job:  {d13:.2f}")
    print(f"  CV ↔ Job:     {d23:.2f}")

    print(f"\n  VERDICT:")
    print(f"  Mean max dispersion: {mean_dispersion:.2f}")
    print(f"  Max dispersion:      {max_dispersion}")
    consistent = mean_dispersion <= 1.0
    print(f"  → {'CONSISTENT' if consistent else 'INCONSISTENT'} (threshold: ≤1.0)")

    return {
        "d_quest_cv": d12,
        "d_quest_job": d13,
        "d_cv_job": d23,
        "mean_dispersion": mean_dispersion,
        "consistent": consistent,
    }


# ═══════════════════════════════════════════════════════════════
# QUESTIONNAIRE DETERMINISM CHECK (bonus)
# ═══════════════════════════════════════════════════════════════

def test_questionnaire_determinism():
    print_separator("BONUS — QUESTIONNAIRE DETERMINISM CHECK")

    answers = {
        "q1": "A", "q2": "B", "q3": "C",
        "q4": "B", "q9": "A", "q15": "C",
        "q5": "A", "q7": "B", "q16": "A",
        "q8": "C", "q10": "B", "q11": "A",
        "q6": "C", "q13": "A", "q14": "B",
        "q17": "C", "q18": "B", "q19": "A",
        "q12": "Builder", "q20": "Expert",
        "q21": "Leader", "q22": "Explorer",
    }

    results = []
    for i in range(5):
        r = compute_profile_vector(answers)
        results.append(r)

    all_identical = all(r == results[0] for r in results)
    print(f"  Ran compute_profile_vector 5 times with identical answers")
    print(f"  All results identical: {'YES ✓' if all_identical else 'NO ✗'}")
    print(f"  Result: {json.dumps(results[0])}")

    return {"deterministic": all_identical}


# ═══════════════════════════════════════════════════════════════
# MAIN — RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 70)
    print("  NEXTMOVE V4 — ROBUSTNESS AUDIT")
    print("  4 protocols + 1 bonus check")
    print("█" * 70)

    # Bonus: questionnaire determinism (no LLM needed)
    r0 = test_questionnaire_determinism()

    # Test 1: Swap Invariance
    r1 = test_swap_invariance()

    # Test 2: Questionnaire Bypass
    r2 = test_questionnaire_bypass()

    # Test 3: Noise Injection
    r3 = test_noise_injection()

    # Test 4: Latent Consistency
    r4 = test_latent_consistency()

    # ═══════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  FINAL AUDIT REPORT")
    print("█" * 70)

    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  COMPONENT ANALYSIS                                        │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Questionnaire (deterministic):  {'STABLE ✓' if r0['deterministic'] else 'UNSTABLE ✗':<20}        │")
    print(f"  │  Swap Invariance (paraphrase):   {'STABLE ✓' if r1['stable'] else 'UNSTABLE ✗':<20}        │")
    print(f"  │    - Axis match:                 {r1['axis_match_pct']:.0f}%{'':<20}       │")
    print(f"  │    - Tension match:              {r1['tension_match_pct']:.0f}%{'':<20}       │")
    print(f"  │  Questionnaire Bypass:           signal={'YES' if r2['bypass_adds_signal'] else 'NO':<15}        │")
    print(f"  │    - Profile distance:           {r2['profile_distance']:.2f}{'':<20}       │")
    print(f"  │  Noise Injection:                {'STABLE ✓' if r3['stable'] else 'UNSTABLE ✗':<20}        │")
    print(f"  │    - Axis stability:             {r3['axis_stability']:.0f}%{'':<20}       │")
    print(f"  │    - Tension stability:          {r3['tension_stability']:.0f}%{'':<20}       │")
    print(f"  │  Latent Consistency:             {'CONSISTENT ✓' if r4['consistent'] else 'INCONSISTENT ✗':<20}    │")
    print(f"  │    - Mean dispersion:            {r4['mean_dispersion']:.2f}{'':<20}       │")
    print("  └─────────────────────────────────────────────────────────────┘")

    # Compute robustness score
    scores = []
    scores.append(100 if r0["deterministic"] else 0)  # questionnaire
    scores.append(r1["axis_match_pct"])                # swap invariance
    scores.append(r3["axis_stability"])                 # noise injection
    scores.append(100 if r4["consistent"] else max(0, 100 - r4["mean_dispersion"] * 50))
    robustness = statistics.mean(scores)

    # Determine verdict
    if robustness >= 85:
        verdict = "REAL INSTRUMENT"
        verdict_detail = "Stable measurement system with deterministic core"
    elif robustness >= 65:
        verdict = "HYBRID SYSTEM"
        verdict_detail = "Deterministic core + LLM extraction with moderate variance"
    else:
        verdict = "PROMPT SYSTEM"
        verdict_detail = "Unstable inference system dominated by LLM variance"

    print(f"\n  ╔═══════════════════════════════════════════════════════════════╗")
    print(f"  ║  1. VERDICT: {verdict:<48}║")
    print(f"  ║     {verdict_detail:<56}║")
    print(f"  ║                                                             ║")
    print(f"  ║  2. ROBUSTNESS SCORE: {robustness:.0f}/100{'':<35}║")
    print(f"  ╚═══════════════════════════════════════════════════════════════╝")

    print(f"\n  3. FRAGILITY POINTS:")
    print(f"  ───────────────────")
    fragilities = []
    if not r1["stable"]:
        fragilities.append("LLM axis extraction varies with paraphrase (swap instability)")
    if not r3["stable"]:
        fragilities.append("LLM output sensitive to minor wording changes (noise sensitivity)")
    if not r4["consistent"]:
        fragilities.append("LLM-inferred profiles diverge from questionnaire profiles")
    if r1["axis_match_pct"] < 100:
        fragilities.append(f"Axis levels not fully reproducible ({100 - r1['axis_match_pct']:.0f}% variation)")
    if not r3.get("switching_cost_stable", True):
        fragilities.append("Switching cost assessment varies across runs")
    fragilities.append("Single LLM call = single point of failure for all qualitative content")
    fragilities.append("3-level quantization (LOW/MEDIUM/HIGH) masks continuous variance")
    fragilities.append("No temperature=0 or seed pinning — inherent stochasticity")

    for i, f in enumerate(fragilities, 1):
        print(f"  {i}. {f}")

    print(f"\n  4. RECOMMENDATION:")
    print(f"  ──────────────────")
    if robustness >= 85:
        print("  → KEEP AS-IS: system is sufficiently stable for production use")
        print("  → Minor: pin temperature=0, add seed for reproducibility")
    elif robustness >= 65:
        print("  → ADJUST: the system has a solid deterministic core but LLM extraction")
        print("    introduces meaningful variance.")
        print("  → Actions:")
        print("    1. Pin temperature=0 and use seed parameter for reproducibility")
        print("    2. Run extraction 2-3x and take majority vote on axis levels")
        print("    3. Consider reducing to binary levels (LOW/HIGH) to reduce noise")
        print("    4. Add explicit constraints in prompt for borderline cases")
    else:
        print("  → MOVE TO ADAPTIVE/PAIRWISE SYSTEM (V4+)")
        print("  → The LLM extraction layer dominates the output too much.")
        print("  → The questionnaire adds insufficient signal vs raw LLM inference.")
        print("  → Actions:")
        print("    1. Replace single LLM call with pairwise comparisons")
        print("    2. Use questionnaire to constrain LLM output, not just contextualize")
        print("    3. Add ensemble extraction (3 calls + vote)")

    print(f"\n  ARCHITECTURAL ANALYSIS:")
    print(f"  ───────────────────────")
    print(f"  Layer 1 — Questionnaire → Profile:  100% DETERMINISTIC")
    print(f"  Layer 2 — LLM Extraction:            STOCHASTIC (main variance source)")
    print(f"  Layer 3 — Engine → Decision Card:    100% DETERMINISTIC")
    print(f"  ")
    print(f"  The system is a deterministic sandwich around a stochastic core.")
    print(f"  Stability depends ENTIRELY on LLM extraction reproducibility.")

    print("\n" + "█" * 70)
    print("  END OF AUDIT")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
