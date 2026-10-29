"""
Data Models for Experiment 4: Personalized Recommendation System
Course: 24CS2506 – Building Applications using GPT-4
"""

import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class UserProfile:
    """Represents a structured user profile as defined in Section 5 of Lab Manual."""
    name: str = "Arun"
    age: int = 20
    field: str = "Computer Science"
    interests: List[str] = dataclasses.field(default_factory=lambda: ["AI", "Python"])
    skill_level: str = "Beginner"  # Beginner, Intermediate, Advanced
    preferred_type: str = "Practical"  # Practical, Theoretical, Balanced
    goal: str = "Learn AI"
    natural_query: str = "I am a beginner who likes programming and wants to learn AI. I prefer practical courses rather than theoretical courses."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "age": self.age,
            "field": self.field,
            "interests": self.interests,
            "skill_level": self.skill_level,
            "preferred_type": self.preferred_type,
            "goal": self.goal,
            "natural_query": self.natural_query
        }


@dataclass
class RecommendationItem:
    """Individual item in recommendation list matching Section 7 format."""
    rank: int
    title: str
    reason: str
    score: int  # 0 to 100 percentage
    category: str = "General"
    tags: List[str] = dataclasses.field(default_factory=list)
    level: str = "Beginner"
    practical_ratio: float = 0.8
    syllabus: List[str] = dataclasses.field(default_factory=list)

    @property
    def score_display(self) -> str:
        return f"{self.score}%"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "recommendation": self.title,
            "reason": self.reason,
            "score": self.score_display,
            "score_numeric": self.score,
            "category": self.category,
            "level": self.level,
            "tags": self.tags,
            "syllabus": self.syllabus
        }


@dataclass
class RecommendationResult:
    """Encapsulates full recommendation output and execution metadata."""
    user_summary: str
    items: List[RecommendationItem]
    structured_prompt: str
    system_prompt: str
    source: str
    latency_sec: float
    analytics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    feedback_history: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_summary": self.user_summary,
            "items": [item.to_dict() for item in self.items],
            "structured_prompt": self.structured_prompt,
            "system_prompt": self.system_prompt,
            "source": self.source,
            "latency_sec": self.latency_sec,
            "analytics": self.analytics,
            "feedback_history": self.feedback_history
        }
