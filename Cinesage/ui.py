import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

class Movie(BaseModel):
    title:Optional[str]
    release_year:Optional[int]
    genre:List[str]
    director:Optional[str]
    cast:List[str]
    rating:Optional[float]
    summary:str
# Load environment variables (Make sure MISTRAL_API_KEY is in your .env file)
load_dotenv()

# Set up the Streamlit page layout and title
st.set_page_config(
    page_title="CineSage | Movie Review Intelligence",
    page_icon="🎬",
    layout="wide"
)

# Initialize the Mistral model
# (Wrapped in st.cache_resource so it doesn't re-initialize on every user click)
@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2506")

model = get_model()

# Define the analysis prompt template
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Movie Review Intelligence Analyst working for a recommendation and insights platform.

Your responsibility is to carefully analyze movie reviews and extract structured, actionable intelligence that can help:
- recommendation systems
- audience profiling
- product teams
- content analysts
- marketing teams
- sentiment analysis pipelines

You must behave like a precise information extraction engine.

TASK:
Analyze the provided movie review and extract useful instructions, preferences, complaints, emotional signals, and recommendations.

IMPORTANT RULES:
- Extract only facts explicitly mentioned or strongly implied.
- Do NOT hallucinate information.
- Keep outputs concise and normalized.
- One insight per bullet/list item.
- Avoid duplicates.
- Use lowercase phrases where possible.
- If information is unavailable, return empty arrays or empty strings.
ANALYSIS GUIDELINES:
- "liked_elements" should contain praised aspects.
- "disliked_elements" should contain criticized aspects.
- "actionable_insights" should be useful for recommendation engines or producers.
- "recommendation_signals" should indicate who may enjoy the movie.
- "watch_intent" should estimate how strongly the reviewer recommends watching it."""),
    ("human", "Review to analyze:\n\n{paragraph}")
])

# --- UI Header ---
st.title("🎬 CineSage")
st.subheader("Senior Movie Review Intelligence Platform")
st.caption("Extract structured, actionable insights from raw user text to power recommendation engines and audience profiling.")

st.divider()

# --- Layout split into two columns ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📥 Input Review")
    # Text area for the user to input the movie review paragraph
    user_paragraph = st.text_area(
        label="Paste the movie review paragraph below:",
        height=300,
        placeholder="e.g., I absolutely loved the gritty cinematography and the synth-driven score! However, the third act felt completely rushed..."
    )
    
    # Analyze button
    analyze_btn = st.button("Analyze Review", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Extracted Intelligence")
    
    if analyze_btn:
        if not user_paragraph.strip():
            st.warning("Please enter a valid review paragraph before clicking analyze.")
        else:
            with st.spinner("Analyzing review semantics and extracting facts..."):
                try:
                    # Construct prompt and invoke Mistral
                    final_prompt = PROMPT_TEMPLATE.invoke({"paragraph": user_paragraph})
                    response = model.invoke(final_prompt)
                    
                    # Display the generated content nicely in an informational box
                    st.success("Analysis Complete!")
                    st.markdown(response.content)
                    
                except Exception as e:
                    st.error(f"An error occurred during extraction: {e}")
    else:
        # Placeholder text before analysis runs
        st.info("Paste a review in the left pane and click 'Analyze Review' to see structured outputs here.")