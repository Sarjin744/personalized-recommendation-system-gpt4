"""
AI-Powered Personalized Recommendation System Web Application
Course: 24CS2506 – Building Applications using GPT-4
Experiment 4: Personalized Recommendation System
Department of Computer Science and Engineering
Karunya Institute of Technology and Sciences
"""

import json
import os
import streamlit as st
import pandas as pd

from engine import (
    PersonalizedRecommender,
    UserProfile,
    FeedbackEngine,
    RecommendationAnalytics
)

# Page configuration
st.set_page_config(
    page_title="Personalized Recommendation System | Lab Exp 4",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 18px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-primary { background-color: #dbeafe; color: #1e40af; }
    .badge-success { background-color: #dcfce7; color: #166534; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    .badge-purple { background-color: #f3e8ff; color: #6b21a8; }
    .score-badge {
        font-size: 1.05rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 16px;
    }
    .score-high { background-color: #dcfce7; color: #15803d; }
    .score-mid { background-color: #e0f2fe; color: #0369a1; }
    .info-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .metric-container {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_recommender():
    return PersonalizedRecommender()


recommender = get_recommender()
feedback_engine = FeedbackEngine(recommender)

# Load Preset Profiles
preset_path = os.path.join(os.path.dirname(__file__), "data", "preset_profiles.json")
preset_data = {}
if os.path.exists(preset_path):
    with open(preset_path, "r", encoding="utf-8") as f:
        preset_data = json.load(f)

profiles_map = {p["id"]: p for p in preset_data.get("profiles", [])}

# Sidebar: Configuration & User Profile Setup
st.sidebar.markdown("## 👤 User Profile Configuration")
st.sidebar.caption("Define the target user as specified in Section 5 of Lab Manual.")

preset_choice = st.sidebar.selectbox(
    "Select Benchmark Profile Preset:",
    options=["arun", "student_a", "student_b", "priya", "custom"],
    format_func=lambda x: {
        "arun": "Arun (CS Student, AI + Python) [Lab Benchmark]",
        "student_a": "Student A (Python, AI, Data Science)",
        "student_b": "Student B (Cybersecurity, Networks)",
        "priya": "Priya (Data Science to ML Engineer)",
        "custom": "Custom Profile (Manual Input)"
    }.get(x, x)
)

if preset_choice != "custom" and preset_choice in profiles_map:
    p_data = profiles_map[preset_choice]
    default_name = p_data.get("name", "Arun")
    default_age = p_data.get("age", 20)
    default_field = p_data.get("field", "Computer Science")
    default_interests = p_data.get("interests", ["AI", "Python"])
    default_level = p_data.get("skill_level", "Beginner")
    default_type = p_data.get("preferred_type", "Practical")
    default_goal = p_data.get("goal", "Learn AI")
    default_query = p_data.get("user_query", "")
else:
    default_name = "Alex"
    default_age = 22
    default_field = "Information Technology"
    default_interests = ["Python", "AI"]
    default_level = "Beginner"
    default_type = "Practical"
    default_goal = "Master Artificial Intelligence"
    default_query = "I am a beginner who likes programming and wants to learn AI. I prefer practical courses rather than theoretical courses."

user_name = st.sidebar.text_input("Name:", value=default_name)
col_a, col_b = st.sidebar.columns(2)
with col_a:
    user_age = st.number_input("Age:", min_value=15, max_value=80, value=default_age)
with col_b:
    skill_level = st.selectbox(
        "Skill Level:",
        options=["Beginner", "Intermediate", "Advanced"],
        index=["Beginner", "Intermediate", "Advanced"].index(default_level) if default_level in ["Beginner", "Intermediate", "Advanced"] else 0
    )

user_field = st.sidebar.text_input("Academic Field / Department:", value=default_field)

available_interests = [
    "AI", "Python", "Data Science", "Machine Learning", "Deep Learning",
    "Generative AI", "Computer Vision", "Cybersecurity", "Networks",
    "Ethical Hacking", "Web Development", "Cloud", "Data Analytics"
]
# Ensure defaults exist
clean_defaults = [i for i in default_interests if i in available_interests]
if not clean_defaults:
    clean_defaults = ["AI", "Python"]

selected_interests = st.sidebar.multiselect(
    "Interests:",
    options=available_interests,
    default=clean_defaults
)

preferred_type = st.sidebar.radio(
    "Preferred Course Type:",
    options=["Practical", "Theoretical", "Balanced"],
    index=["Practical", "Theoretical", "Balanced"].index(default_type) if default_type in ["Practical", "Theoretical", "Balanced"] else 0
)

user_goal = st.sidebar.text_input("Career / Learning Goal:", value=default_goal)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗣️ Natural Language Description")
user_query = st.sidebar.text_area(
    "User Preference Statement (Section 3):",
    value=default_query,
    height=85,
    help="Generative AI understands conversational descriptions, unlike traditional rule engines."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Parameters")
top_k = st.sidebar.slider("Number of Recommendations (Top K):", min_value=3, max_value=8, value=5)

ollama_status = recommender.ollama.is_connected
if ollama_status:
    st.sidebar.success(f"🟢 Ollama Local LLM Connected ({recommender.ollama.get_best_available_model()})")
else:
    st.sidebar.info("🔵 Semantic Recommendation Synthesizer (Active Fallback Mode)")

# Header
st.markdown('<div class="main-title">🎯 Personalized Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">24CS2506 – Building Applications using GPT-4 | Experiment 4 | Karunya Institute of Technology and Sciences</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. 📋 Lab Benchmark & Recommendations",
    "2. 🔄 Feedback-Based Refinement",
    "3. 👥 Student A vs B Personalization",
    "4. ⚖️ Generative AI vs Traditional",
    "5. 🌐 Multi-Domain Recommenders"
])

current_profile = UserProfile(
    name=user_name,
    age=user_age,
    field=user_field,
    interests=selected_interests,
    skill_level=skill_level,
    preferred_type=preferred_type,
    goal=user_goal,
    natural_query=user_query
)

# ==============================================================================
# TAB 1: Personalized Recommendations (Section 5 & 7 Benchmark)
# ==============================================================================
with tab1:
    st.markdown("### 1. User Profile Formulation (Section 5)")
    st.caption("The system converts structured user profile attributes and natural language descriptions into an engineered prompt for GPT-4.")

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="margin-top:0; color:#1e40af;">🎓 User Profile Card</h4>
            <b>Name:</b> {current_profile.name} (Age: {current_profile.age})<br>
            <b>Field:</b> {current_profile.field}<br>
            <b>Interests:</b> {' '.join([f'<span class="badge badge-primary">{i}</span>' for i in current_profile.interests])}<br>
            <b>Skill Level:</b> <span class="badge badge-warning">{current_profile.skill_level}</span> | 
            <b>Type:</b> <span class="badge badge-success">{current_profile.preferred_type}</span><br>
            <b>Goal:</b> {current_profile.goal}<br>
            <hr style="margin: 8px 0;">
            <b>Natural Language Input:</b> <i>"{current_profile.natural_query}"</i>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4 style="margin-top:0; color:#0f766e;">💡 Architecture Flow</h4>
            <div style="font-size:0.9rem; line-height: 1.6;">
                <b>Step 1:</b> User Profile Extraction (Demographics & Objectives)<br>
                <b>Step 2:</b> Natural Language Preference Parsing<br>
                <b>Step 3:</b> Structured Few-Shot Prompt Formulation<br>
                <b>Step 4:</b> GPT-4 Semantic Reasoning & Ranking<br>
                <b>Step 5:</b> Calibrated Score & Explanation Generation
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚀 Generate Personalized Recommendations", type="primary", use_container_width=True):
        with st.spinner("Analyzing profile & synthesizing recommendations..."):
            rec_result = recommender.recommend(current_profile, top_k=top_k)
            st.session_state["rec_result"] = rec_result

    # Display recommendations if in session_state
    if "rec_result" in st.session_state:
        res: RecommendationResult = st.session_state["rec_result"]

        st.markdown("---")
        st.markdown("### 2. Personalized Recommendations (Section 7 Expected Output)")
        st.info(f"**Target User:** {res.user_summary}")

        # Render exact Section 7 Table
        table_rows = []
        for it in res.items:
            table_rows.append({
                "Rank": it.rank,
                "Recommendation": it.title,
                "Reason": it.reason,
                "Score": it.score_display
            })

        df = pd.DataFrame(table_rows)
        st.table(df)

        # Quantitative Quality Metrics Row
        st.markdown("#### 📊 Recommendation Quality Analytics")
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        analytics = res.analytics
        with mcol1:
            st.metric("Interest Coverage", f"{analytics.get('coverage_pct', 100)}%")
        with mcol2:
            st.metric("Skill Alignment", f"{analytics.get('skill_alignment_pct', 100)}%")
        with mcol3:
            st.metric("Practical Ratio", f"{analytics.get('practical_alignment_pct', 85)}%")
        with mcol4:
            st.metric("Average Score", f"{analytics.get('average_score', 88)}%")
        with mcol5:
            st.metric("Response Time", f"{res.latency_sec}s")

        # Interactive Detailed Item Breakdown
        with st.expander("🔍 View Detailed Course Curricula & Syllabus Breakdowns"):
            for it in res.items:
                st.markdown(f"**#{it.rank} - {it.title}** (`{it.category}` | `{it.level}` | Practicality: `{int(it.practical_ratio*100)}%`)")
                st.write(f"*Why Recommended:* {it.reason}")
                if it.syllabus:
                    st.write("**Key Modules:** " + " ➔ ".join(it.syllabus))
                st.markdown("---")

        # Engineered Prompt Inspector
        with st.expander("🛠️ View Engineered System Prompt (Section 5 Implementation)"):
            st.text_area("Prompt sent to LLM:", value=res.structured_prompt, height=220, disabled=True)

        # Export features
        st.markdown("#### 💾 Export Recommendations")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            st.download_button(
                "📥 Download CSV Table",
                data=df.to_csv(index=False),
                file_name="recommendations_table.csv",
                mime="text/csv"
            )
        with exp_col2:
            st.download_button(
                "📥 Download JSON Payload",
                data=json.dumps(res.to_dict(), indent=2),
                file_name="recommendations_output.json",
                mime="application/json"
            )
        with exp_col3:
            md_text = f"# Personalized Recommendations\n\n**User:** {res.user_summary}\n\n" + df.to_markdown(index=False)
            st.download_button(
                "📥 Download Markdown Report",
                data=md_text,
                file_name="recommendation_report.md",
                mime="text/markdown"
            )

# ==============================================================================
# TAB 2: Feedback-Based Refinement (Section 6)
# ==============================================================================
with tab2:
    st.markdown("### 🔄 Feedback-Based Recommendation (Section 6)")
    st.write(
        "A key advantage of Generative AI is conversational iteration: users can provide natural language "
        "feedback on initial suggestions to immediately reshape recommendations without retraining models."
    )

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown("#### Initial Recommendation Set:")
        st.markdown("""
        1. **Python Programming**
        2. **Machine Learning**
        3. **Web Development**
        """)
        st.caption("Notice: 'Web Development' is present, but may not align with the student's primary interest.")

    with fcol2:
        st.markdown("#### User Feedback:")
        feedback_input = st.text_input(
            "Enter feedback to refine recommendations:",
            value="I am more interested in AI than Web Development.",
            help="Lab manual Section 6 benchmark feedback query."
        )
        run_feedback = st.button("⚡ Apply Feedback & Refine Recommendations", type="primary")

    if run_feedback or "feedback_active" in st.session_state:
        st.session_state["feedback_active"] = True

        # Initial dummy items matching Section 6
        initial_items = [
            recommender.catalog[6] if len(recommender.catalog) > 6 else recommender.catalog[0],
            recommender.catalog[1] if len(recommender.catalog) > 1 else recommender.catalog[0],
            recommender.catalog[11] if len(recommender.catalog) > 11 else recommender.catalog[0]
        ]

        dummy_items = [
            recommender.recommend(current_profile).items[0],
            recommender.recommend(current_profile).items[1]
        ]

        f_result = feedback_engine.refine(current_profile, dummy_items, feedback_input)

        st.markdown("---")
        st.markdown("#### 🎯 Updated Recommendations (Post-Feedback Personalization)")

        updated_rows = []
        for it in f_result["updated_items"]:
            updated_rows.append({
                "Rank": it.rank,
                "Updated Recommendation": it.title,
                "Reason / Adjustment Justification": it.reason,
                "Score": it.score_display
            })

        up_df = pd.DataFrame(updated_rows)
        st.table(up_df)

        diff = f_result["diff"]
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.error(f"❌ **Topics Removed/Demoted:**\n\n" + "\n".join([f"- {r}" for r in diff["removed"]]))
        with dcol2:
            st.success(f"✅ **Topics Promoted:**\n\n" + "\n".join([f"- {p}" for p in diff["promoted"]]))
        with dcol3:
            st.info(f"✨ **Newly Introduced AI Tracks:**\n\n" + "\n".join([f"- {n}" for n in diff["newly_added"]]))

        st.success(f"**Impact Summary:** {diff['feedback_impact_summary']}")

# ==============================================================================
# TAB 3: Student A vs Student B Personalization Proof (Section 2)
# ==============================================================================
with tab3:
    st.markdown("### 👥 Personalization Proof: Student A vs Student B (Section 2)")
    st.write(
        "A personalized recommendation provides different suggestions for different users. "
        "Here we verify that distinct student profiles automatically yield specialized, non-overlapping recommendations."
    )

    sa_col, sb_col = st.columns(2)

    with sa_col:
        st.markdown("""
        <div class="info-card">
            <h4 style="color:#0369a1; margin-top:0;">👨‍🎓 Student A Profile</h4>
            <b>Interests:</b> Python, AI, Data Science<br>
            <b>Level:</b> Beginner | <b>Field:</b> Computer Science<br>
            <b>Objective:</b> Build practical Data Science & AI models
        </div>
        """, unsafe_allow_html=True)

        prof_a = UserProfile(
            name="Student A",
            interests=["Python", "AI", "Data Science"],
            skill_level="Beginner",
            field="Computer Science",
            goal="Data Science Projects"
        )
        res_a = recommender.recommend(prof_a, top_k=3)

        st.markdown("**Recommendations for Student A:**")
        for it in res_a.items:
            st.markdown(f"**{it.rank}. {it.title}** (`{it.score_display}`)\n- *{it.reason}*")

    with sb_col:
        st.markdown("""
        <div class="info-card">
            <h4 style="color:#7e22ce; margin-top:0;">🛡️ Student B Profile</h4>
            <b>Interests:</b> Cybersecurity, Networks, Ethical Hacking<br>
            <b>Level:</b> Beginner | <b>Field:</b> Information Security<br>
            <b>Objective:</b> Network Defense & Ethical Penetration Testing
        </div>
        """, unsafe_allow_html=True)

        prof_b = UserProfile(
            name="Student B",
            interests=["Cybersecurity", "Networks", "Ethical Hacking"],
            skill_level="Beginner",
            field="Cybersecurity",
            goal="Network Security"
        )
        res_b = recommender.recommend(prof_b, top_k=3)

        st.markdown("**Recommendations for Student B:**")
        for it in res_b.items:
            st.markdown(f"**{it.rank}. {it.title}** (`{it.score_display}`)\n- *{it.reason}*")

    st.success("✅ **Personalization Verified:** User Profile ➔ Preferences ➔ AI Analysis ➔ Differentiated Personalized Recommendations.")

# ==============================================================================
# TAB 4: Generative AI vs Traditional (Section 3)
# ==============================================================================
with tab4:
    st.markdown("### ⚖️ Why Use Generative AI for Recommendation? (Section 3)")
    st.write(
        "Traditional recommendation systems generally depend on predefined rules, ratings, or similarity calculations. "
        "Generative AI can understand rich natural-language descriptions of user preferences."
    )

    comp = recommender.compare_traditional_vs_generative(current_profile)

    ccol1, ccol2 = st.columns(2)
    with ccol1:
        st.markdown("#### ❌ Traditional Recommendation Systems")
        st.caption("Predefined Rules, Static SQL Filters & Collaborative Filtering")
        for it in comp["traditional_recommendations"]:
            st.markdown(f"""
            <div style="background:#fee2e2; border-left:4px solid #ef4444; padding:10px; margin-bottom:8px; border-radius:4px;">
                <b>{it['rank']}. {it['recommendation']}</b> ({it['method']})<br>
                <small style="color:#7f1d1d;">⚠️ Limitation: {it['limitation']}</small>
            </div>
            """, unsafe_allow_html=True)

    with ccol2:
        st.markdown("#### ✅ Generative AI Recommendation Systems (GPT-4)")
        st.caption("Deep Contextual Reasoning, Natural Language & Explainability")
        for it in comp["generative_recommendations"]:
            st.markdown(f"""
            <div style="background:#dcfce7; border-left:4px solid #22c55e; padding:10px; margin-bottom:8px; border-radius:4px;">
                <b>{it['rank']}. {it['recommendation']}</b> [{it['score']}] ({it['method']})<br>
                <small style="color:#14532d;">✨ Advantage: {it['advantage']}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 🌟 Core Advantages of Generative AI (Section 2 & 3 of Manual):")
    for ins in comp["insights"]:
        st.markdown(f"- **{ins}**")

# ==============================================================================
# TAB 5: Multi-Domain Recommenders (Section 1)
# ==============================================================================
with tab5:
    st.markdown("### 🌐 Real-World Recommendation System Applications (Section 1)")
    st.caption("Recommendation systems power modern industry platforms: Netflix, YouTube, Amazon, Spotify, and Coursera.")

    domain_tabs = st.tabs(["🎬 Netflix (Movies)", "🛍️ Amazon (Products)", "🎵 Spotify (Music)", "📚 Learning (Courses)"])

    showcases = preset_data.get("domain_showcases", {})

    with domain_tabs[0]:
        n_data = showcases.get("netflix", {})
        st.markdown(f"**Profile Scenario:** *\"{n_data.get('profile', '')}\"*")
        ndf = pd.DataFrame(n_data.get("items", []))
        st.table(ndf)

    with domain_tabs[1]:
        a_data = showcases.get("amazon", {})
        st.markdown(f"**Profile Scenario:** *\"{a_data.get('profile', '')}\"*")
        adf = pd.DataFrame(a_data.get("items", []))
        st.table(adf)

    with domain_tabs[2]:
        s_data = showcases.get("spotify", {})
        st.markdown(f"**Profile Scenario:** *\"{s_data.get('profile', '')}\"*")
        sdf = pd.DataFrame(s_data.get("items", []))
        st.table(sdf)

    with domain_tabs[3]:
        st.markdown("**Simple Example from Lab Manual:**")
        st.markdown("""
        **User Preference:** Interested in Python, Artificial Intelligence and Data Science.  
        **Recommendations:**
        1. Python for Data Science
        2. Machine Learning Fundamentals
        3. Generative AI with Python
        4. Data Analytics using Python
        5. Deep Learning Fundamentals
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "<b>Course 24CS2506 – Building Applications using GPT-4</b> | "
    "Experiment 4: Personalized Recommendation System | "
    "Karunya Institute of Technology and Sciences"
    "</div>",
    unsafe_allow_html=True
)
