"""
Prompt Builder for Experiment 4: Personalized Recommendation System
Converts structured user profiles, natural language statements, and feedback into optimized LLM prompts.
"""

from typing import List, Dict, Any, Optional
from .models import UserProfile


class PromptBuilder:
    """Constructs engineered prompts for LLMs and Generative Recommendation Engines."""

    @staticmethod
    def build_system_prompt() -> str:
        """Returns the expert persona and constraints for the recommendation system."""
        return (
            "You are an expert Personalized Educational and Career Recommendation AI powered by GPT-4.\n"
            "Your role is to analyze a user's academic profile, natural language desires, skill level, and goals, "
            "and provide ranked, highly tailored course recommendations with transparent justifications and match scores.\n"
            "Key Directives:\n"
            "1. Ground recommendations strictly on the user's specific skill level and preferred learning style (e.g. Practical vs Theoretical).\n"
            "2. Generate clear, human-understandable reasons that explain why each specific item was chosen.\n"
            "3. Assign calibrated match scores (0% to 100%) indicating affinity.\n"
            "4. Respond with valid JSON adhering to the specified schema without Markdown fences when requested."
        )

    @staticmethod
    def build_structured_prompt(
        profile: UserProfile,
        top_k: int = 5,
        available_courses: Optional[List[Dict[str, Any]]] = None,
        feedback: Optional[str] = None
    ) -> str:
        """
        Converts inputs into a structured prompt as required by Section 5 of Lab Manual.
        """
        interests_str = ", ".join(profile.interests)

        prompt = (
            "### TASK: GENERATE PERSONALIZED RECOMMENDATIONS\n\n"
            "### 1. USER PROFILE (Structured Profile Table):\n"
            f"- Name: {profile.name}\n"
            f"- Age: {profile.age}\n"
            f"- Field / Background: {profile.field}\n"
            f"- Core Interests: {interests_str}\n"
            f"- Skill Level: {profile.skill_level}\n"
            f"- Preferred Learning Style: {profile.preferred_type}\n"
            f"- Academic / Career Goal: {profile.goal}\n\n"
        )

        if profile.natural_query:
            prompt += (
                "### 2. NATURAL LANGUAGE USER STATEMENT:\n"
                f'"{profile.natural_query}"\n\n'
            )

        if feedback:
            prompt += (
                "### 3. INTERACTIVE USER FEEDBACK (Iteration Update):\n"
                f'The user previously reviewed recommendations and gave this feedback:\n'
                f'"{feedback}"\n'
                f'Adjust and refine the recommendations to directly honor this feedback.\n\n'
            )

        if available_courses:
            prompt += "### 4. CATALOG CONTEXT (Available Courses):\n"
            for c in available_courses[:12]:
                tags = ", ".join(c.get("tags", []))
                prompt += f"- {c['title']} [{c.get('level', 'Beginner')}] ({c.get('type', 'Practical')}): {c.get('description', '')} (Tags: {tags})\n"
            prompt += "\n"

        prompt += (
            f"### 5. OUTPUT REQUIREMENTS:\n"
            f"Generate exactly {top_k} recommendations ranked from 1 to {top_k}.\n"
            "Provide the response in valid JSON with this exact schema:\n"
            "{\n"
            '  "user_summary": "Computer Science student | Interests: AI, Python | Level: Beginner",\n'
            '  "recommendations": [\n'
            "    {\n"
            '      "rank": 1,\n'
            '      "recommendation": "Python for AI",\n'
            '      "reason": "Matches Python and AI interests",\n'
            '      "score": 95\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        return prompt

    @staticmethod
    def build_feedback_refinement_prompt(
        profile: UserProfile,
        initial_items: List[str],
        feedback: str
    ) -> str:
        """Constructs prompt specifically for Section 6: Feedback-Based Recommendation."""
        initial_list_str = "\n".join(f"- {item}" for item in initial_items)
        return (
            "### FEEDBACK-BASED RECOMMENDATION REFINEMENT\n"
            f"User Profile: {profile.name} ({profile.field}, Level: {profile.skill_level})\n"
            f"Core Interests: {', '.join(profile.interests)}\n\n"
            "Initial Recommendations Given:\n"
            f"{initial_list_str}\n\n"
            f'User Feedback Received:\n"{feedback}"\n\n'
            "Task: Update the recommendations according to the user feedback. Remove or demote disfavored topics and surface higher-priority topics.\n"
            "Output JSON format:\n"
            "{\n"
            '  "updated_recommendations": [\n'
            '    {"rank": 1, "recommendation": "Artificial Intelligence", "reason": "Prioritized based on feedback", "score": 98}\n'
            "  ],\n"
            '  "feedback_analysis": "Explanation of changes made based on user input"\n'
            "}"
        )

    @staticmethod
    def build_traditional_rule_prompt(profile: UserProfile) -> str:
        """Simulates how a traditional rule/keyword matching system handles the profile."""
        return (
            f"Keyword Matching Rule: SELECT courses WHERE category IN ({profile.interests}) "
            f"AND level = '{profile.skill_level}' ORDER BY rating DESC;"
        )
