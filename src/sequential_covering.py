"""
Smart Crop Recommendation System - Sequential Covering (Rule Learning)
======================================================================
Module 2: Classical Rule Learning implementation from scratch.
Implements the Sequential Covering (Learn-One-Rule) algorithm to induce
human-interpretable IF-THEN rules from categorical agricultural data.

Features:
- Documented categorical dataset derived from Crop_recommendation.csv
- Greedy specialization heuristic based on Rule Precision and Coverage
- Single target class (one-vs-rest) and multi-class rule induction
- Detailed execution trace, training metrics (coverage & precision),
  and graceful handling of unmatched instances and empty datasets.
"""

from typing import List, Dict, Tuple, Set, Optional, Any
import copy


# =====================================================================
# 1. CATEGORICAL EDUCATIONAL DATASET & DERIVATION DOCUMENTATION
# =====================================================================
"""
DERIVATION DOCUMENTATION:
------------------------
To enable fast, beginner-friendly rule induction, continuous numeric features
from Crop_recommendation.csv were discretized into categorical bins based on
standard agronomic thresholds:

1. MoistureLevel (from relative humidity %):
   - 'High'    : humidity > 75%   (tropical / wetland air)
   - 'Medium'  : 50% <= humidity <= 75% (semi-humid transition)
   - 'Low'     : humidity < 50%   (arid / dry atmosphere)

2. RainfallLevel (from precipitation mm):
   - 'High'    : rainfall > 150 mm  (flood / monsoonal rain)
   - 'Moderate': 70 mm <= rainfall <= 150 mm (temperate irrigation)
   - 'Low'     : rainfall < 70 mm   (drought / dry season)

3. NutrientProfile (from N and K in kg/ha):
   - 'NitrogenRich' : N >= 80 kg/ha (heavy vegetative foliage demand)
   - 'PotassiumRich': K >= 100 kg/ha (fruit / orchard sugar demand)
   - 'Balanced'     : N < 80 and K < 100 (standard soil ratio)

4. TemperatureLevel (from ambient temperature °C):
   - 'Warm': temperature > 24.0°C (subtropical / tropical warmth)
   - 'Cool': temperature <= 24.0°C (temperate / montane climate)
"""

CATEGORICAL_ATTRIBUTES: Dict[str, List[str]] = {
    "NutrientProfile": ["NitrogenRich", "PotassiumRich", "Balanced"],
    "MoistureLevel": ["High", "Medium", "Low"],
    "RainfallLevel": ["High", "Moderate", "Low"],
    "TemperatureLevel": ["Warm", "Cool"]
}

CATEGORICAL_CROP_DATASET: List[Dict[str, Any]] = [
    # Rice instances (Wetland crop)
    {"id": 1, "NutrientProfile": "Balanced", "MoistureLevel": "High", "RainfallLevel": "High", "TemperatureLevel": "Warm", "CropCategory": "Rice"},
    {"id": 2, "NutrientProfile": "Balanced", "MoistureLevel": "High", "RainfallLevel": "High", "TemperatureLevel": "Cool", "CropCategory": "Rice"},
    {"id": 3, "NutrientProfile": "Balanced", "MoistureLevel": "High", "RainfallLevel": "Moderate", "TemperatureLevel": "Warm", "CropCategory": "Rice"},
    {"id": 4, "NutrientProfile": "NitrogenRich", "MoistureLevel": "High", "RainfallLevel": "High", "TemperatureLevel": "Warm", "CropCategory": "Rice"},

    # Apple instances (High-potassium temperate fruit)
    {"id": 5, "NutrientProfile": "PotassiumRich", "MoistureLevel": "High", "RainfallLevel": "Moderate", "TemperatureLevel": "Cool", "CropCategory": "Apple"},
    {"id": 6, "NutrientProfile": "PotassiumRich", "MoistureLevel": "Medium", "RainfallLevel": "Moderate", "TemperatureLevel": "Cool", "CropCategory": "Apple"},
    {"id": 7, "NutrientProfile": "PotassiumRich", "MoistureLevel": "High", "RainfallLevel": "Low", "TemperatureLevel": "Cool", "CropCategory": "Apple"},

    # Cotton instances (Heavy nitrogen feeder)
    {"id": 8, "NutrientProfile": "NitrogenRich", "MoistureLevel": "High", "RainfallLevel": "Moderate", "TemperatureLevel": "Warm", "CropCategory": "Cotton"},
    {"id": 9, "NutrientProfile": "NitrogenRich", "MoistureLevel": "Medium", "RainfallLevel": "Moderate", "TemperatureLevel": "Warm", "CropCategory": "Cotton"},
    {"id": 10, "NutrientProfile": "NitrogenRich", "MoistureLevel": "High", "RainfallLevel": "Low", "TemperatureLevel": "Warm", "CropCategory": "Cotton"},

    # Chickpea instances (Dry legume / pulse)
    {"id": 11, "NutrientProfile": "Balanced", "MoistureLevel": "Low", "RainfallLevel": "Moderate", "TemperatureLevel": "Warm", "CropCategory": "Chickpea"},
    {"id": 12, "NutrientProfile": "Balanced", "MoistureLevel": "Low", "RainfallLevel": "Low", "TemperatureLevel": "Warm", "CropCategory": "Chickpea"},
    {"id": 13, "NutrientProfile": "Balanced", "MoistureLevel": "Low", "RainfallLevel": "Moderate", "TemperatureLevel": "Cool", "CropCategory": "Chickpea"},

    # Coffee instances (Tropical plantation)
    {"id": 14, "NutrientProfile": "Balanced", "MoistureLevel": "Medium", "RainfallLevel": "High", "TemperatureLevel": "Warm", "CropCategory": "Coffee"},
    {"id": 15, "NutrientProfile": "Balanced", "MoistureLevel": "Medium", "RainfallLevel": "Moderate", "TemperatureLevel": "Warm", "CropCategory": "Coffee"},
    {"id": 16, "NutrientProfile": "NitrogenRich", "MoistureLevel": "Medium", "RainfallLevel": "High", "TemperatureLevel": "Warm", "CropCategory": "Coffee"}
]


# =====================================================================
# 2. RULE CLASS DEFINITION
# =====================================================================

class Rule:
    """
    Represents an IF-THEN propositional rule:
    IF (Attr1 = Val1 AND Attr2 = Val2 ...) THEN TargetClass = C
    """

    def __init__(self, conditions: Dict[str, str], target_class: str):
        self.conditions = conditions  # e.g. {"MoistureLevel": "High", "RainfallLevel": "High"}
        self.target_class = target_class
        self.pos_covered_count: int = 0
        self.neg_covered_count: int = 0
        self.covered_instance_ids: List[int] = []
        self.precision: float = 0.0
        self.coverage_pct: float = 0.0

    def matches(self, features: Dict[str, str]) -> bool:
        """Check if an instance satisfies all conjunction conditions of this rule."""
        for attr, val in self.conditions.items():
            if features.get(attr) != val:
                return False
        return True

    def to_string(self) -> str:
        """Return formatted IF-THEN rule string."""
        if not self.conditions:
            return f"IF True THEN Crop = {self.target_class}"
        cond_str = " AND ".join([f"{attr} = '{val}'" for attr, val in self.conditions.items()])
        return f"IF {cond_str} THEN Crop = {self.target_class}"

    def __repr__(self) -> str:
        return self.to_string()


# =====================================================================
# 3. SEQUENTIAL COVERING ALGORITHM
# =====================================================================

class SequentialCovering:
    """
    Sequential Covering Rule Inducer:
    Learns disjunctive sets of conjunctive IF-THEN rules.

    Procedure for a target class:
    1. Divide training data into Positives (target class) and Negatives (other classes).
    2. While Positives remain uncovered:
       a. Learn-One-Rule by greedily adding attribute=value tests that maximize precision.
       b. Add the best rule to the learned rule set.
       c. Remove all positive examples covered by this new rule.
    3. Return the ordered list of rules.
    """

    def __init__(
        self,
        attributes: Dict[str, List[str]],
        target_col: str = "CropCategory",
        min_precision: float = 1.0,
        max_conditions: int = 4
    ):
        self.attributes = attributes
        self.target_col = target_col
        self.min_precision = min_precision
        self.max_conditions = max_conditions
        self.rules: List[Rule] = []
        self.history: List[Dict[str, Any]] = []

    def learn_one_rule(
        self,
        positives: List[Dict[str, Any]],
        negatives: List[Dict[str, Any]],
        target_class: str
    ) -> Optional[Rule]:
        """
        Greedy top-down beam search (Learn-One-Rule).
        Starts with empty antecedent (IF True) and iteratively adds conditions
        that maximize rule precision p = pos / (pos + neg).
        """
        if not positives:
            return None

        current_conditions: Dict[str, str] = {}
        best_rule: Optional[Rule] = None

        while len(current_conditions) < self.max_conditions:
            candidate_specs: List[Tuple[float, int, Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]] = []

            for attr, possible_vals in self.attributes.items():
                if attr in current_conditions:
                    continue  # Attribute already in antecedent

                for val in possible_vals:
                    trial_cond = copy.deepcopy(current_conditions)
                    trial_cond[attr] = val

                    # Test on current positive and negative sets
                    pos_cov = [ex for ex in positives if all(ex.get(k) == v for k, v in trial_cond.items())]
                    neg_cov = [ex for ex in negatives if all(ex.get(k) == v for k, v in trial_cond.items())]

                    n_pos = len(pos_cov)
                    n_neg = len(neg_cov)

                    if n_pos == 0:
                        continue  # Must cover at least one positive example

                    precision = n_pos / (n_pos + n_neg)
                    # Candidate score: tuple (precision, n_pos) for sorting
                    candidate_specs.append((precision, n_pos, trial_cond, pos_cov, neg_cov))

            if not candidate_specs:
                break

            # Pick specialization with highest precision, breaking ties with higher positive coverage
            candidate_specs.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best_prec, best_n_pos, best_cond, best_pos_cov, best_neg_cov = candidate_specs[0]

            current_conditions = best_cond

            rule_candidate = Rule(current_conditions, target_class)
            rule_candidate.pos_covered_count = best_n_pos
            rule_candidate.neg_covered_count = len(best_neg_cov)
            rule_candidate.covered_instance_ids = [ex.get("id", -1) for ex in best_pos_cov]
            rule_candidate.precision = best_prec
            best_rule = rule_candidate

            # Stopping condition: rule is 100% pure (excludes all negatives)
            if len(best_neg_cov) == 0:
                break

        return best_rule

    def fit_target_class(self, dataset: List[Dict[str, Any]], target_class: str) -> List[Rule]:
        """
        Sequential covering for a single target class (One-vs-Rest).
        """
        self.rules = []
        self.history = []

        if not dataset:
            return []

        positives = [copy.deepcopy(ex) for ex in dataset if ex.get(self.target_col) == target_class]
        negatives = [copy.deepcopy(ex) for ex in dataset if ex.get(self.target_col) != target_class]

        initial_pos_total = len(positives)
        iteration = 1

        self.history.append({
            "iteration": 0,
            "action": f"Initialized dataset for target class '{target_class}': {len(positives)} positive, {len(negatives)} negative examples.",
            "pos_remaining": len(positives),
            "rule": None
        })

        while len(positives) > 0:
            rule = self.learn_one_rule(positives, negatives, target_class)
            if rule is None or rule.pos_covered_count == 0:
                break

            # Compute coverage percentage relative to initial positives
            rule.coverage_pct = (rule.pos_covered_count / initial_pos_total) * 100 if initial_pos_total > 0 else 0.0

            self.rules.append(rule)

            # Sequential Covering Step: Remove covered positive examples
            covered_ids = set(rule.covered_instance_ids)
            positives = [ex for ex in positives if ex.get("id") not in covered_ids]

            self.history.append({
                "iteration": iteration,
                "action": f"Learned Rule #{iteration}: Covers {rule.pos_covered_count} positive and {rule.neg_covered_count} negative examples (Precision: {rule.precision*100:.1f}%).",
                "pos_remaining": len(positives),
                "rule": rule
            })
            iteration += 1

        return self.rules

    def fit_all_classes(
        self,
        dataset: List[Dict[str, Any]],
        class_order: Optional[List[str]] = None,
        default_class: str = "Rice"
    ) -> List[Rule]:
        """
        Multi-class Sequential Covering:
        Learns rules sequentially across all classes, appending a fallback default rule.
        """
        if not dataset:
            return []

        all_classes = class_order or sorted(list(set(ex[self.target_col] for ex in dataset)))
        all_rules: List[Rule] = []

        for c in all_classes:
            c_rules = self.fit_target_class(dataset, c)
            all_rules.extend(c_rules)

        # Append default fallback rule
        default_rule = Rule({}, default_class)
        default_rule.precision = 0.0
        default_rule.pos_covered_count = 0
        all_rules.append(default_rule)

        self.rules = all_rules
        return self.rules

    def predict(
        self,
        features: Dict[str, str],
        rules: Optional[List[Rule]] = None,
        fallback_class: Optional[str] = "Unmatched"
    ) -> Tuple[str, Optional[Rule]]:
        """
        Classify an instance using the learned rule set.
        Evaluates rules in priority order.
        Returns (predicted_class, matching_rule).
        """
        rule_list = rules if rules is not None else self.rules
        if not rule_list:
            return (fallback_class or "Unmatched", None)

        for rule in rule_list:
            if rule.matches(features):
                return (rule.target_class, rule)

        return (fallback_class or "Unmatched", None)

    def evaluate_training_set(
        self,
        dataset: List[Dict[str, Any]],
        rules: Optional[List[Rule]] = None
    ) -> Dict[str, Any]:
        """
        Calculate training-set metrics: precision, coverage, and overall accuracy.
        Explicitly labeled as training metrics, not independent test performance.
        """
        rule_list = rules if rules is not None else self.rules
        if not dataset or not rule_list:
            return {"accuracy": 0.0, "total_samples": 0, "correct": 0, "unmatched": 0}

        correct = 0
        unmatched = 0

        for ex in dataset:
            pred_class, matching_rule = self.predict(ex, rule_list, fallback_class=None)
            if matching_rule is None:
                unmatched += 1
            elif pred_class == ex.get(self.target_col):
                correct += 1

        accuracy = (correct / len(dataset)) * 100 if len(dataset) > 0 else 0.0
        return {
            "total_samples": len(dataset),
            "correct": correct,
            "unmatched": unmatched,
            "accuracy": accuracy,
            "metric_type": "Training Set Metrics (Empirical Fit, not independent test performance)"
        }


# =====================================================================
# CLI DEMO RUNNER
# =====================================================================

def run_demo():
    print("=" * 70)
    print("SEQUENTIAL COVERING (RULE LEARNING) DEMO")
    print("=" * 70)
    dataset = CATEGORICAL_CROP_DATASET
    print(f"Loaded categorical training dataset with {len(dataset)} examples across 5 crops.")

    target = "Rice"
    print(f"\n--- [1] Learning Rules for Target Class: '{target}' ---")
    sc = SequentialCovering(CATEGORICAL_ATTRIBUTES)
    rules = sc.fit_target_class(dataset, target)

    for i, r in enumerate(rules, 1):
        print(f"  Rule {i}: {r.to_string()}")
        print(f"          Covered IDs: {r.covered_instance_ids} | Pos: {r.pos_covered_count}, Neg: {r.neg_covered_count} | Precision: {r.precision*100:.1f}%")

    metrics = sc.evaluate_training_set(dataset, rules)
    print(f"\nTraining Coverage & Accuracy for '{target}' rules: {metrics['accuracy']:.1f}% ({metrics['correct']}/{metrics['total_samples']})")

    print("\n--- [2] Multi-Class Rule Learning Across All Crops ---")
    sc_multi = SequentialCovering(CATEGORICAL_ATTRIBUTES)
    all_rules = sc_multi.fit_all_classes(dataset)
    for i, r in enumerate(all_rules, 1):
        print(f"  Rule {i}: {r.to_string()} (Precision: {r.precision*100:.1f}%)")

    # Unmatched instance test
    unseen_instance = {"NutrientProfile": "PotassiumRich", "MoistureLevel": "Low", "RainfallLevel": "Low", "TemperatureLevel": "Warm"}
    pred, matched_r = sc_multi.predict(unseen_instance, rules=all_rules[:-1], fallback_class="No Rule Matched")
    print(f"\nUnseen Instance Test: {unseen_instance}")
    print(f"Prediction (without fallback default): {pred} (Matched Rule: {matched_r})")


if __name__ == "__main__":
    run_demo()
