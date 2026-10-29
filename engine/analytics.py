"""
Analytics and Evaluation Metrics for Experiment 4: Personalized Recommendation System
"""

from typing import List, Dict, Any
from .models import UserProfile, RecommendationItem


class RecommendationAnalytics:
    """Calculates quantitative quality metrics on recommendations."""

    @staticmethod
    def evaluate(profile: UserProfile, items: List[RecommendationItem]) -> Dict[str, Any]:
        if not items:
            return {
                "coverage_pct": 0,
                "skill_alignment_pct": 0,
                "practical_alignment_pct": 0,
                "average_score": 0,
                "diversity_index": 0.0
            }

        # 1. Interest Coverage: what fraction of user interests are addressed
        user_interests = [i.lower() for i in profile.interests]
        covered_interests = set()

        for item in items:
            text = f"{item.title} {item.reason} {' '.join(item.tags)} {item.category}".lower()
            for interest in user_interests:
                if interest in text:
                    covered_interests.add(interest)

        coverage_pct = round((len(covered_interests) / max(len(user_interests), 1)) * 100)

        # 2. Skill Level Alignment
        level_target = profile.skill_level.lower()
        aligned_levels = sum(1 for item in items if item.level.lower() == level_target or "basics" in item.title.lower() or "fundamentals" in item.title.lower() or "intro" in item.title.lower())
        skill_alignment_pct = round((aligned_levels / len(items)) * 100)

        # 3. Practical Type Alignment
        practical_req = profile.preferred_type.lower() == "practical"
        practical_scores = [item.practical_ratio for item in items]
        avg_practical = sum(practical_scores) / len(items) if practical_scores else 0.8
        practical_alignment_pct = round(avg_practical * 100) if practical_req else 80

        # 4. Average Match Score
        avg_score = round(sum(item.score for item in items) / len(items), 1)

        # 5. Diversity Index: unique categories
        categories = set(item.category for item in items if item.category)
        diversity_index = round(len(categories) / len(items), 2)

        return {
            "coverage_pct": min(coverage_pct, 100),
            "skill_alignment_pct": min(skill_alignment_pct, 100),
            "practical_alignment_pct": practical_alignment_pct,
            "average_score": avg_score,
            "diversity_index": diversity_index,
            "items_count": len(items)
        }
