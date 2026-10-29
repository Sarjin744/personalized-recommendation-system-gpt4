"""
Comprehensive Test Suite for Experiment 4: Personalized Recommendation System
Tests data integrity, engine logic, prompt generation, ranking algorithms, feedback loops, and metrics.
"""

import sys
import os
import json
import unittest

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (
    UserProfile,
    RecommendationItem,
    RecommendationResult,
    PromptBuilder,
    OllamaClient,
    PersonalizedRecommender,
    FeedbackEngine,
    RecommendationAnalytics
)


class TestPersonalizedRecommendationSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.recommender = PersonalizedRecommender()
        cls.feedback_engine = FeedbackEngine(cls.recommender)

    def test_01_catalog_integrity(self):
        """Verify data/course_catalog.json is valid and contains required fields."""
        catalog_path = os.path.join(os.path.dirname(__file__), "data", "course_catalog.json")
        self.assertTrue(os.path.exists(catalog_path), "course_catalog.json must exist")
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        courses = data.get("courses", [])
        self.assertGreaterEqual(len(courses), 10, "Should have at least 10 courses in catalog")
        for c in courses:
            self.assertIn("id", c)
            self.assertIn("title", c)
            self.assertIn("category", c)
            self.assertIn("tags", c)
            self.assertIn("level", c)
            self.assertIn("description", c)
            self.assertIn("practical_ratio", c)

    def test_02_preset_profiles_integrity(self):
        """Verify preset_profiles.json contains benchmark profiles from lab manual."""
        preset_path = os.path.join(os.path.dirname(__file__), "data", "preset_profiles.json")
        self.assertTrue(os.path.exists(preset_path), "preset_profiles.json must exist")
        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profiles = {p["id"]: p for p in data.get("profiles", [])}
        self.assertIn("arun", profiles, "Arun benchmark profile must exist")
        self.assertIn("student_a", profiles, "Student A profile must exist")
        self.assertIn("student_b", profiles, "Student B profile must exist")

        showcases = data.get("domain_showcases", {})
        self.assertIn("netflix", showcases)
        self.assertIn("amazon", showcases)
        self.assertIn("spotify", showcases)

    def test_03_prompt_builder_structured_prompt(self):
        """Verify Section 5 prompt builder structure."""
        profile = UserProfile(
            name="Arun",
            age=20,
            field="Computer Science",
            interests=["AI", "Python"],
            skill_level="Beginner",
            preferred_type="Practical",
            goal="Learn AI",
            natural_query="I am a beginner who likes programming and wants to learn AI."
        )
        prompt = PromptBuilder.build_structured_prompt(profile, top_k=5)
        self.assertIn("Arun", prompt)
        self.assertIn("Computer Science", prompt)
        self.assertIn("AI, Python", prompt)
        self.assertIn("Beginner", prompt)
        self.assertIn("Practical", prompt)
        self.assertIn("Learn AI", prompt)
        self.assertIn("OUTPUT REQUIREMENTS", prompt)

    def test_04_benchmark_arun_section_7(self):
        """Verify Section 7 expected output table for Arun."""
        profile = UserProfile(
            name="Arun",
            age=20,
            field="Computer Science",
            interests=["AI", "Python"],
            skill_level="Beginner",
            preferred_type="Practical",
            goal="Learn AI"
        )
        result = self.recommender.recommend(profile, top_k=5)
        self.assertEqual(len(result.items), 5, "Must return exactly 5 recommendations")

        expected_titles = [
            "Python for AI",
            "Machine Learning Basics",
            "Generative AI with Python",
            "Data Science with Python",
            "Deep Learning Basics"
        ]
        actual_titles = [item.title for item in result.items]
        for exp in expected_titles:
            self.assertIn(exp, actual_titles, f"Expected course '{exp}' not found in recommendations")

        # Verify rank ordering
        self.assertEqual(result.items[0].title, "Python for AI")
        self.assertEqual(result.items[0].score, 95)
        self.assertEqual(result.items[1].title, "Machine Learning Basics")
        self.assertEqual(result.items[1].score, 92)

        # Verify scores are in descending order
        scores = [item.score for item in result.items]
        self.assertEqual(scores, sorted(scores, reverse=True), "Scores must be ranked descending")

    def test_05_student_a_vs_student_b_personalization(self):
        """Verify Section 2: Different user profiles yield distinct recommendations."""
        prof_a = UserProfile(
            name="Student A",
            interests=["Python", "AI", "Data Science"],
            skill_level="Beginner",
            field="Computer Science"
        )
        prof_b = UserProfile(
            name="Student B",
            interests=["Cybersecurity", "Networks", "Ethical Hacking"],
            skill_level="Beginner",
            field="Cybersecurity"
        )

        res_a = self.recommender.recommend(prof_a, top_k=3)
        res_b = self.recommender.recommend(prof_b, top_k=3)

        titles_a = set(item.title for item in res_a.items)
        titles_b = set(item.title for item in res_b.items)

        # Confirm non-overlapping specialized recommendations
        intersection = titles_a.intersection(titles_b)
        self.assertEqual(len(intersection), 0, f"Student A and B recommendations should be distinct, but overlapped on: {intersection}")

        # Check B contains security topics
        b_text = " ".join(titles_b).lower()
        self.assertTrue(any(k in b_text for k in ["security", "network", "hacking"]))

    def test_06_feedback_refinement_section_6(self):
        """Verify Section 6: Feedback 'I am more interested in AI than Web Development'."""
        profile = UserProfile(name="Arun", interests=["AI", "Python"])
        initial_items = [
            RecommendationItem(rank=1, title="Python Programming", reason="", score=90),
            RecommendationItem(rank=2, title="Machine Learning", reason="", score=88),
            RecommendationItem(rank=3, title="Web Development", reason="", score=80)
        ]

        feedback_text = "I am more interested in AI than Web Development."
        f_res = self.feedback_engine.refine(profile, initial_items, feedback_text)

        self.assertEqual(f_res["status"], "success")
        updated_titles = [item.title for item in f_res["updated_items"]]

        # Web development must NOT be in the updated top recommendations
        for t in updated_titles:
            self.assertNotIn("web development", t.lower(), f"Web Development should have been removed, found in: {t}")

        # Core AI topics must be present
        has_ai = any("intelligence" in t.lower() or "ai" in t.lower() for t in updated_titles)
        self.assertTrue(has_ai, "Updated recommendations must focus on AI")

        # Verify diff report
        diff = f_res["diff"]
        self.assertTrue(len(diff["removed"]) > 0)
        self.assertTrue(len(diff["newly_added"]) > 0)

    def test_07_traditional_vs_generative_comparison(self):
        """Verify Section 3 comparison function returns meaningful contrasts."""
        comp = self.recommender.compare_traditional_vs_generative(UserProfile())
        self.assertIn("traditional_recommendations", comp)
        self.assertIn("generative_recommendations", comp)
        self.assertIn("insights", comp)
        self.assertGreaterEqual(len(comp["insights"]), 3)

    def test_08_analytics_evaluation(self):
        """Verify analytics calculates sensible percentage bounds [0, 100]."""
        profile = UserProfile(interests=["AI", "Python"], skill_level="Beginner", preferred_type="Practical")
        res = self.recommender.recommend(profile, top_k=5)
        analytics = res.analytics

        self.assertGreaterEqual(analytics["coverage_pct"], 50)
        self.assertLessEqual(analytics["coverage_pct"], 100)
        self.assertGreaterEqual(analytics["skill_alignment_pct"], 50)
        self.assertLessEqual(analytics["skill_alignment_pct"], 100)
        self.assertGreaterEqual(analytics["average_score"], 70)
        self.assertLessEqual(analytics["average_score"], 100)

    def test_09_custom_profile_recommendation(self):
        """Verify general recommendation logic handles arbitrary queries cleanly."""
        custom_prof = UserProfile(
            name="Deep Learner",
            field="Robotics",
            interests=["Deep Learning", "Computer Vision"],
            skill_level="Intermediate",
            preferred_type="Practical",
            goal="Build Autonomous Drones",
            natural_query="I need practical experience in object detection and deep neural networks."
        )
        res = self.recommender.recommend(custom_prof, top_k=4)
        self.assertEqual(len(res.items), 4)
        top_item = res.items[0]
        self.assertIn(top_item.category, ["Deep Learning", "Artificial Intelligence"])
        self.assertGreaterEqual(top_item.score, 80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
