"""
Smart Crop Recommendation System - Concept Learning Lab
=======================================================
Pure Python implementations of classic Concept Learning algorithms:
1. Find-S (Maximally Specific Hypothesis)
2. Candidate Elimination (Version Space Boundaries S and G)

Includes a documented educational categorical dataset for Crop Suitability demonstration,
step-by-step trace generation, and handling of noise/inconsistency (empty version spaces).
"""

from typing import List, Dict, Tuple, Set, Optional, Any
import copy


# =====================================================================
# DEFAULT TEACHING DATASET: Wetland Rice Cultivation Suitability
# =====================================================================
# Note: This is an educational categorical demonstration dataset.
# Concept: "Suitable for Wetland Rice Cultivation"
DEFAULT_ATTRIBUTES = {
    "SoilType": ["Clayey", "Loamy", "Sandy"],
    "Climate": ["Tropical", "Temperate", "Arid"],
    "Moisture": ["High", "Medium", "Low"],
    "Rainfall": ["Heavy", "Moderate", "Light"]
}

DEFAULT_EXAMPLES = [
    {
        "id": 1,
        "features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Heavy"},
        "label": "Yes",
        "rationale": "Ideal conditions: water-retaining clay soil, tropical warmth, high moisture, heavy rain."
    },
    {
        "id": 2,
        "features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "High", "Rainfall": "Moderate"},
        "label": "Yes",
        "rationale": "Suitable: clay soil with moderate rainfall is still sufficient with high moisture."
    },
    {
        "id": 3,
        "features": {"SoilType": "Sandy", "Climate": "Arid", "Moisture": "Low", "Rainfall": "Light"},
        "label": "No",
        "rationale": "Unsuitable: porous sandy soil, arid climate, low moisture cause total crop failure."
    },
    {
        "id": 4,
        "features": {"SoilType": "Clayey", "Climate": "Tropical", "Moisture": "Medium", "Rainfall": "Heavy"},
        "label": "Yes",
        "rationale": "Suitable: heavy rainfall compensates for medium ambient humidity in tropical clay soil."
    },
    {
        "id": 5,
        "features": {"SoilType": "Loamy", "Climate": "Temperate", "Moisture": "Low", "Rainfall": "Light"},
        "label": "No",
        "rationale": "Unsuitable: insufficient moisture and light rainfall in temperate conditions."
    }
]


# =====================================================================
# HYPOTHESIS HELPER FUNCTIONS
# =====================================================================

def is_more_general_attr(h_attr: str, s_attr: str) -> bool:
    """Check if h_attr is more general than or equal to s_attr."""
    if h_attr == "?":
        return True
    if s_attr == "0":
        return True
    return h_attr == s_attr


def is_more_general_or_equal(h1: Tuple[str, ...], h2: Tuple[str, ...]) -> bool:
    """
    Check if hypothesis h1 is more general than or equal to hypothesis h2 (h1 >=_g h2).
    """
    if all(val == "0" for val in h2):
        return True
    if all(val == "0" for val in h1):
        return all(val == "0" for val in h2)
    return all(is_more_general_attr(a, b) for a, b in zip(h1, h2))


def matches_instance(h: Tuple[str, ...], instance: Tuple[str, ...]) -> bool:
    """
    Check if hypothesis h matches an instance vector (i.e. predicts positive).
    A hypothesis containing '0' matches no instance.
    """
    if any(val == "0" for val in h):
        return False
    return all(h_attr == "?" or h_attr == inst_val for h_attr, inst_val in zip(h, instance))


# =====================================================================
# 1. FIND-S ALGORITHM
# =====================================================================

class FindS:
    """
    Find-S Algorithm: Learns the maximally specific hypothesis consistent with positive examples.
    Note: Find-S completely ignores negative training examples.
    """

    def __init__(self, attribute_names: List[str]):
        self.attribute_names = attribute_names
        self.n_attrs = len(attribute_names)
        self.hypothesis: Tuple[str, ...] = tuple("0" for _ in range(self.n_attrs))
        self.history: List[Dict[str, Any]] = []

    def fit(self, examples: List[Dict[str, Any]]) -> Tuple[str, ...]:
        """
        Train Find-S on a list of examples.
        Each example is a dict: {"features": {...}, "label": "Yes"/"No"}
        """
        self.hypothesis = tuple("0" for _ in range(self.n_attrs))
        self.history = []

        # Record initial state (Step 0)
        self.history.append({
            "step": 0,
            "example": None,
            "label": None,
            "hypothesis": self.hypothesis,
            "action": "Initialized to most specific hypothesis h0 = <0, 0, ...>"
        })

        for idx, ex in enumerate(examples, start=1):
            features_dict = ex["features"]
            label = str(ex["label"]).strip().capitalize()
            instance = tuple(features_dict[attr] for attr in self.attribute_names)

            if label in ["Yes", "+", "1", "True"]:
                # Positive example: generalize hypothesis
                old_h = self.hypothesis
                new_h = []
                for h_val, inst_val in zip(self.hypothesis, instance):
                    if h_val == "0":
                        new_h.append(inst_val)
                    elif h_val == inst_val:
                        new_h.append(h_val)
                    else:
                        new_h.append("?")
                self.hypothesis = tuple(new_h)
                action = f"Positive example: generalized from {old_h} to {self.hypothesis}"
            else:
                # Negative example: Find-S ignores it
                action = "Negative example: ignored by Find-S algorithm."

            self.history.append({
                "step": idx,
                "example": features_dict,
                "label": label,
                "hypothesis": self.hypothesis,
                "action": action
            })

        return self.hypothesis

    def predict(self, features: Dict[str, str]) -> str:
        """Predict whether an instance matches the learned hypothesis."""
        instance = tuple(features[attr] for attr in self.attribute_names)
        return "Yes" if matches_instance(self.hypothesis, instance) else "No"


# =====================================================================
# 2. CANDIDATE ELIMINATION ALGORITHM
# =====================================================================

class CandidateElimination:
    """
    Candidate Elimination Algorithm:
    Maintains the complete Version Space bounded by:
    - S: Set of maximally specific hypotheses
    - G: Set of maximally general hypotheses
    Assumes a consistent, noise-free concept-learning dataset.
    """

    def __init__(self, attribute_domains: Dict[str, List[str]]):
        self.attribute_domains = attribute_domains
        self.attribute_names = list(attribute_domains.keys())
        self.n_attrs = len(self.attribute_names)
        
        # Initial boundaries
        self.S: Set[Tuple[str, ...]] = {tuple("0" for _ in range(self.n_attrs))}
        self.G: Set[Tuple[str, ...]] = {tuple("?" for _ in range(self.n_attrs))}
        self.history: List[Dict[str, Any]] = []
        self.is_version_space_empty: bool = False

    def fit(self, examples: List[Dict[str, Any]]) -> Tuple[Set[Tuple[str, ...]], Set[Tuple[str, ...]]]:
        """
        Train Candidate Elimination on a list of examples.
        Records step-by-step history and handles version space collapse.
        """
        self.S = {tuple("0" for _ in range(self.n_attrs))}
        self.G = {tuple("?" for _ in range(self.n_attrs))}
        self.history = []
        self.is_version_space_empty = False

        self.history.append({
            "step": 0,
            "example": None,
            "label": None,
            "S": copy.deepcopy(self.S),
            "G": copy.deepcopy(self.G),
            "action": "Initialized S0 = {<0, 0, ...>} and G0 = {<?, ?, ...>}",
            "version_space_empty": False
        })

        for idx, ex in enumerate(examples, start=1):
            if self.is_version_space_empty:
                # Already collapsed due to inconsistency in earlier data
                self.history.append({
                    "step": idx,
                    "example": ex["features"],
                    "label": ex["label"],
                    "S": set(),
                    "G": set(),
                    "action": "Version space is already empty (inconsistent data previously detected).",
                    "version_space_empty": True
                })
                continue

            features_dict = ex["features"]
            label = str(ex["label"]).strip().capitalize()
            instance = tuple(features_dict[attr] for attr in self.attribute_names)

            if label in ["Yes", "+", "1", "True"]:
                # Positive Example
                # 1. Remove from G any hypothesis inconsistent with instance (i.e. does not match)
                self.G = {g for g in self.G if matches_instance(g, instance)}

                # 2. For each s in S that does not match instance:
                new_S = set()
                for s in self.S:
                    if matches_instance(s, instance):
                        new_S.add(s)
                    else:
                        # Generalize s minimally to cover instance
                        candidate_h = []
                        for s_val, inst_val in zip(s, instance):
                            if s_val == "0":
                                candidate_h.append(inst_val)
                            elif s_val == inst_val:
                                candidate_h.append(s_val)
                            else:
                                candidate_h.append("?")
                        cand = tuple(candidate_h)
                        # Retain cand only if it is more specific than or equal to some g in G
                        if any(is_more_general_or_equal(g, cand) for g in self.G):
                            new_S.add(cand)

                # Remove any hypothesis from new_S that is more general than another in new_S
                self.S = {
                    s1 for s1 in new_S
                    if not any(is_more_general_or_equal(s1, s2) and s1 != s2 for s2 in new_S)
                }

                action = f"Positive Example: Generalize S to cover {instance}; prune G members not covering it."

            else:
                # Negative Example
                # 1. Remove from S any hypothesis that matches instance (inconsistent with negative)
                self.S = {s for s in self.S if not matches_instance(s, instance)}

                # 2. For each g in G that matches instance:
                new_G = set()
                for g in self.G:
                    if not matches_instance(g, instance):
                        new_G.add(g)
                    else:
                        # Specialize g minimally so it does not match instance
                        for i in range(self.n_attrs):
                            if g[i] == "?":
                                attr_name = self.attribute_names[i]
                                for possible_val in self.attribute_domains[attr_name]:
                                    if possible_val != instance[i]:
                                        cand_list = list(g)
                                        cand_list[i] = possible_val
                                        cand = tuple(cand_list)
                                        # Retain cand only if it is more general than some s in S
                                        if any(is_more_general_or_equal(cand, s) for s in self.S):
                                            new_G.add(cand)

                # Remove any hypothesis from new_G that is more specific than another in new_G
                self.G = {
                    g1 for g1 in new_G
                    if not any(is_more_general_or_equal(g2, g1) and g1 != g2 for g2 in new_G)
                }

                action = f"Negative Example: Specialize G to exclude {instance}; prune S members covering it."

            # Check if version space collapsed
            if len(self.S) == 0 or len(self.G) == 0:
                self.is_version_space_empty = True
            else:
                # Also verify that at least one (s, g) pair satisfies s <=_g g
                valid_pair = any(is_more_general_or_equal(g, s) for s in self.S for g in self.G)
                if not valid_pair:
                    self.is_version_space_empty = True

            self.history.append({
                "step": idx,
                "example": features_dict,
                "label": label,
                "S": copy.deepcopy(self.S) if not self.is_version_space_empty else set(),
                "G": copy.deepcopy(self.G) if not self.is_version_space_empty else set(),
                "action": action if not self.is_version_space_empty else "Version Space collapsed! No consistent hypothesis exists.",
                "version_space_empty": self.is_version_space_empty
            })

        return self.S, self.G

    def predict(self, features: Dict[str, str]) -> str:
        """
        Predict label for an instance:
        - "Yes": If all hypotheses in Version Space match instance.
        - "No": If no hypothesis in Version Space matches instance.
        - "Uncertain": If some hypotheses match and some do not.
        - "Undefined": If Version Space is empty.
        """
        if self.is_version_space_empty or len(self.S) == 0 or len(self.G) == 0:
            return "Undefined (Empty Version Space)"

        instance = tuple(features[attr] for attr in self.attribute_names)

        # In Version Space theory (Mitchell, Ch. 2):
        # 1. An instance is unanimously positive iff every s in S matches it
        #    (since if s matches x, any h >=_g s also matches x).
        unanimously_positive = len(self.S) > 0 and all(matches_instance(s, instance) for s in self.S)

        # 2. An instance is unanimously negative iff NO g in G matches it
        #    (since if no g matches x, no h <=_g g can match x).
        unanimously_negative = not any(matches_instance(g, instance) for g in self.G)

        if unanimously_positive:
            return "Yes (Unanimously Positive)"
        elif unanimously_negative:
            return "No (Unanimously Negative)"
        else:
            return "Uncertain (Boundary Disagreement)"


# =====================================================================
# CLI DEMO / VERIFICATION RUNNER
# =====================================================================

def run_demo():
    print("=" * 70)
    print("CONCEPT LEARNING DEMO: Find-S & Candidate Elimination")
    print("Concept: 'Suitable for Wetland Rice Cultivation'")
    print("=" * 70)

    # 1. Run Find-S
    print("\n--- [1] FIND-S TRACE ---")
    find_s = FindS(list(DEFAULT_ATTRIBUTES.keys()))
    final_h = find_s.fit(DEFAULT_EXAMPLES)
    for step in find_s.history:
        print(f"Step {step['step']}: {step['action']}")
        print(f"       Hypothesis: {step['hypothesis']}")
    print(f"\nFinal Find-S Hypothesis: {final_h}")

    # 2. Run Candidate Elimination
    print("\n--- [2] CANDIDATE ELIMINATION TRACE ---")
    ce = CandidateElimination(DEFAULT_ATTRIBUTES)
    final_s, final_g = ce.fit(DEFAULT_EXAMPLES)
    for step in ce.history:
        print(f"Step {step['step']}: Label: {step['label']} | {step['action']}")
        print(f"       S: {sorted(list(step['S']))}")
        print(f"       G: {sorted(list(step['G']))}")
    print(f"\nFinal S Boundary: {sorted(list(final_s))}")
    print(f"Final G Boundary: {sorted(list(final_g))}")
    print(f"Version Space Empty? {ce.is_version_space_empty}")


if __name__ == "__main__":
    run_demo()
