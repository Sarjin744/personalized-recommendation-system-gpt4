"""
Command Line Interface (CLI) for Experiment 4
Personalized Recommendation System
Course: 24CS2506 – Building Applications using GPT-4
Karunya Institute of Technology and Sciences
"""

import sys
import os
import json
import time

from engine import (
    PersonalizedRecommender,
    UserProfile,
    FeedbackEngine,
    RecommendationAnalytics
)


def print_banner():
    print("=" * 80)
    print(" 24CS2506 - Building Applications using GPT-4")
    print(" Experiment 4: Personalized Recommendation System")
    print(" Department of Computer Science and Engineering")
    print("=" * 80)


def print_recommendation_table(user_summary: str, items: list):
    print("\n" + "-" * 80)
    print("PERSONALIZED RECOMMENDATIONS")
    print(f"User Summary: {user_summary}")
    print("-" * 80)
    print(f"{'Rank':<6}{'Recommendation':<32}{'Reason':<32}{'Score':<8}")
    print("-" * 80)
    for it in items:
        title = it.title if len(it.title) <= 30 else it.title[:27] + "..."
        reason = it.reason if len(it.reason) <= 30 else it.reason[:27] + "..."
        print(f"{it.rank:<6}{title:<32}{reason:<32}{it.score_display:<8}")
    print("-" * 80)


def run_benchmark_arun(rec: PersonalizedRecommender):
    print("\n[+] Running Benchmark 1: Arun (Section 5 & 7 Lab Benchmark)")
    profile = UserProfile(
        name="Arun",
        age=20,
        field="Computer Science",
        interests=["AI", "Python"],
        skill_level="Beginner",
        preferred_type="Practical",
        goal="Learn AI",
        natural_query="I am a beginner who likes programming and wants to learn AI. I prefer practical courses rather than theoretical courses."
    )
    result = rec.recommend(profile)
    print_recommendation_table(result.user_summary, result.items)
    print(f"Backend Engine: {result.source} | Latency: {result.latency_sec}s")
    print(f"Analytics: Coverage={result.analytics['coverage_pct']}% | Skill Alignment={result.analytics['skill_alignment_pct']}% | Avg Score={result.analytics['average_score']}%")
    return profile, result


def run_student_comparison(rec: PersonalizedRecommender):
    print("\n" + "=" * 80)
    print("[+] Section 2: Student A vs Student B Personalization Proof")
    print("=" * 80)

    # Student A
    prof_a = UserProfile(
        name="Student A",
        age=21,
        field="Computer Science",
        interests=["Python", "AI", "Data Science"],
        skill_level="Beginner",
        preferred_type="Practical",
        goal="Build practical Data Science & AI models",
        natural_query="Interested in Python, Artificial Intelligence and Data Science. Want practical courses."
    )
    res_a = rec.recommend(prof_a)
    print("\n>>> STUDENT A (Interests: Python, AI, Data Science):")
    for it in res_a.items[:3]:
        print(f"  {it.rank}. {it.title} ({it.score_display}) - {it.reason}")

    # Student B
    prof_b = UserProfile(
        name="Student B",
        age=22,
        field="Cybersecurity",
        interests=["Cybersecurity", "Networks", "Ethical Hacking"],
        skill_level="Beginner",
        preferred_type="Practical",
        goal="Network Security and Penetration Testing",
        natural_query="Want to learn Network Security, Ethical Hacking, and Cybersecurity Fundamentals."
    )
    res_b = rec.recommend(prof_b)
    print("\n>>> STUDENT B (Interests: Cybersecurity, Networks, Ethical Hacking):")
    for it in res_b.items[:3]:
        print(f"  {it.rank}. {it.title} ({it.score_display}) - {it.reason}")

    print("\n[V] VERIFICATION: Different user profiles yield completely different personalized outputs!")


def run_feedback_simulation(rec: PersonalizedRecommender):
    print("\n" + "=" * 80)
    print("[+] Section 6: Feedback-Based Recommendation Simulation")
    print("=" * 80)

    profile = UserProfile(
        name="Arun",
        age=20,
        field="Computer Science",
        interests=["AI", "Python"],
        skill_level="Beginner",
        preferred_type="Practical",
        goal="Learn AI"
    )

    initial_titles = ["Python Programming", "Machine Learning", "Web Development"]
    print("\nStep 1: Initial Recommendations Presented to User:")
    for idx, t in enumerate(initial_titles, start=1):
        print(f"  {idx}. {t}")

    feedback_text = "I am more interested in AI than Web Development."
    print(f'\nStep 2: User provides conversational feedback:\n  >> "{feedback_text}"')

    feedback_engine = FeedbackEngine(rec)
    initial_items = rec.recommend(profile).items[:3]
    f_res = feedback_engine.refine(profile, initial_items, feedback_text)

    print("\nStep 3: Updated Recommendations based on Feedback:")
    for it in f_res["updated_items"]:
        print(f"  {it.rank}. {it.title} ({it.score_display}) - {it.reason}")

    print("\nStep 4: Adaptation Breakdown:")
    print(f"  - Removed: {', '.join(f_res['diff']['removed'])}")
    print(f"  - Newly Added: {', '.join(f_res['diff']['newly_added'])}")
    print(f"  - Impact: {f_res['diff']['feedback_impact_summary']}")


def run_comparison_view(rec: PersonalizedRecommender):
    print("\n" + "=" * 80)
    print("[+] Section 3: Traditional Recommendation vs Generative AI")
    print("=" * 80)

    profile = UserProfile()
    comp = rec.compare_traditional_vs_generative(profile)

    print("\n[TRADITIONAL RULE / SIMILARITY BASED SYSTEM]:")
    for it in comp["traditional_recommendations"]:
        print(f"  {it['rank']}. {it['recommendation']} [{it['method']}] -> Limitation: {it['limitation']}")

    print("\n[GENERATIVE AI (GPT-4 / LLM) SYSTEM]:")
    for it in comp["generative_recommendations"]:
        print(f"  {it['rank']}. {it['recommendation']} ({it['score']}) -> Advantage: {it['advantage']}")

    print("\n[KEY TAKEAWAYS]:")
    for ins in comp["insights"]:
        print(f"  * {ins}")


def run_domain_showcases():
    print("\n" + "=" * 80)
    print("[+] Section 1: Multi-Domain Recommendation Systems (Netflix, Amazon, Spotify)")
    print("=" * 80)

    path = os.path.join(os.path.dirname(__file__), "data", "preset_profiles.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            showcases = data.get("domain_showcases", {})
            for key, val in showcases.items():
                print(f"\n--- {val['name']} ---")
                print(f"User Profile / Need: {val['profile']}")
                print("Recommendations:")
                for idx, it in enumerate(val["items"], start=1):
                    print(f"  {idx}. {it['title']} ({it['score']}): {it['reason']}")


def interactive_menu():
    rec = PersonalizedRecommender()
    print_banner()

    while True:
        print("\nChoose an option:")
        print("  1. Run Benchmark 1: Arun (Section 5 & 7 Expected Table)")
        print("  2. Run Personalization Proof: Student A vs Student B (Section 2)")
        print("  3. Run Interactive Feedback Refinement (Section 6)")
        print("  4. View Traditional vs Generative AI Comparison (Section 3)")
        print("  5. View Multi-Domain Showcase: Netflix, Amazon, Spotify (Section 1)")
        print("  6. Custom User Profile Input")
        print("  7. Run All Benchmarks in Sequence")
        print("  0. Exit")

        choice = input("\nEnter choice [0-7]: ").strip()

        if choice == "1":
            run_benchmark_arun(rec)
        elif choice == "2":
            run_student_comparison(rec)
        elif choice == "3":
            run_feedback_simulation(rec)
        elif choice == "4":
            run_comparison_view(rec)
        elif choice == "5":
            run_domain_showcases()
        elif choice == "6":
            name = input("Enter Name: ").strip() or "Custom User"
            field = input("Enter Field (e.g. Computer Science): ").strip() or "Computer Science"
            interests_raw = input("Enter Interests comma-separated (e.g. Python, AI): ").strip() or "Python, AI"
            interests = [i.strip() for i in interests_raw.split(",") if i.strip()]
            level = input("Enter Skill Level (Beginner/Intermediate/Advanced): ").strip() or "Beginner"
            pref = input("Enter Preferred Type (Practical/Theoretical): ").strip() or "Practical"
            goal = input("Enter Goal: ").strip() or "Career Growth"
            query = input("Enter Natural Language Query: ").strip() or f"I want to learn {', '.join(interests)}."

            p = UserProfile(name=name, field=field, interests=interests, skill_level=level, preferred_type=pref, goal=goal, natural_query=query)
            res = rec.recommend(p)
            print_recommendation_table(res.user_summary, res.items)
        elif choice == "7":
            run_benchmark_arun(rec)
            run_student_comparison(rec)
            run_feedback_simulation(rec)
            run_comparison_view(rec)
            run_domain_showcases()
        elif choice == "0":
            print("\nExiting CLI. Lab Experiment 4 completed successfully.")
            break
        else:
            print("Invalid selection. Please choose between 0 and 7.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        rec = PersonalizedRecommender()
        print_banner()
        run_benchmark_arun(rec)
        run_student_comparison(rec)
        run_feedback_simulation(rec)
        run_comparison_view(rec)
        run_domain_showcases()
    else:
        interactive_menu()
