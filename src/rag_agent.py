import gradio as gr
import os
import subprocess
import tempfile
import datetime
import operator
from typing import Dict, List, Literal, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    groq_api_key="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",
    temperature=0,
)


fake_internal_docs = [m
    Document(page_content="""
    LIBRARY: GalaxyLib
    VERSION: 2.0
    
    FUNCTION: calculate_flux(radius, brightness)
    DESCRIPTION: Calculates the flux of a star given its radius and brightness.
    PARAMETERS:
        - radius (float): Star radius in kilometers
        - brightness (float): Star brightness in solar luminosity units
    RETURNS: float - Flux value in watts per square meter
    LOGIC: returns (brightness / (4 * 3.14159 * radius^2))
    EXAMPLE: calculate_flux(10, 5000) = 39.79
    NOTES: Uses inverse square law for radiation. Radius must be positive.
    
    FUNCTION: warp_speed(factor)
    DESCRIPTION: Returns velocity in km/s based on warp factor.
    PARAMETERS:
        - factor (float): Warp factor (1.0 to 10.0)
    RETURNS: float - Velocity in kilometers per second
    LOGIC: returns 299792 * (factor ^ 3.3)
    EXAMPLE: warp_speed(2.0) = 1234567.89
    NOTES: Velocity cannot exceed speed of light. Factor must be >= 1.0
    """),
    
    Document(page_content="""
    LIBRARY: AstroMath
    VERSION: 1.5
    
    FUNCTION: gravitational_force(mass1, mass2, distance)
    DESCRIPTION: Calculates gravitational force between two objects using Newton's law.
    PARAMETERS:
        - mass1 (float): Mass of first object in kilograms
        - mass2 (float): Mass of second object in kilograms
        - distance (float): Distance between objects in meters
    RETURNS: float - Force in Newtons
    CONSTANT: G = 6.67430e-11 (Gravitational constant)
    LOGIC: returns (G * mass1 * mass2) / (distance ^ 2)
    EXAMPLE: gravitational_force(5.972e24, 7.342e22, 3.844e8) = 2.0e20
    NOTES: Distance must be positive. Both masses must be non-negative.
    
    FUNCTION: orbital_velocity(mass, radius)
    DESCRIPTION: Calculates orbital velocity for a circular orbit.
    PARAMETERS:
        - mass (float): Mass of central body in kilograms
        - radius (float): Orbital radius in meters
    RETURNS: float - Velocity in meters per second
    LOGIC: returns sqrt((G * mass) / radius)
    EXAMPLE: orbital_velocity(5.972e24, 6.371e6) = 7910.0
    NOTES: G = 6.67430e-11. Radius must be greater than body radius.
    """),
    
    Document(page_content="""
    LIBRARY: SpaceUtils
    VERSION: 3.1
    
    FUNCTION: convert_temperature(kelvin)
    DESCRIPTION: Converts temperature from Kelvin to Celsius and Fahrenheit.
    PARAMETERS:
        - kelvin (float): Temperature in Kelvin
    RETURNS: dict - {'celsius': float, 'fahrenheit': float}
    LOGIC: celsius = kelvin - 273.15, fahrenheit = (celsius * 9/5) + 32
    EXAMPLE: convert_temperature(273.15) = {'celsius': 0.0, 'fahrenheit': 32.0}
    NOTES: Kelvin must be >= 0. Absolute zero is at 0K.
    
    FUNCTION: escape_velocity(mass, radius)
    DESCRIPTION: Calculates escape velocity from a celestial body.
    PARAMETERS:
        - mass (float): Mass of celestial body in kilograms
        - radius (float): Radius of celestial body in meters
    RETURNS: float - Escape velocity in meters per second
    LOGIC: returns sqrt((2 * G * mass) / radius)
    CONSTANT: G = 6.67430e-11 (Gravitational constant)
    EXAMPLE: escape_velocity(5.972e24, 6.371e6) = 11186.0
    NOTES: Always greater than orbital velocity. Used for space launch calculations.
    
    FUNCTION: doppler_shift(frequency, velocity, speed_of_light=299792458)
    DESCRIPTION: Calculates observed frequency due to Doppler effect.
    PARAMETERS:
        - frequency (float): Source frequency in Hertz
        - velocity (float): Velocity of source (positive = moving away)
        - speed_of_light (float): Speed of light in m/s (default: 299792458)
    RETURNS: float - Observed frequency in Hertz
    LOGIC: observed = frequency * (speed_of_light) / (speed_of_light + velocity)
    EXAMPLE: doppler_shift(5e14, 1e6) = 4.99833e14
    NOTES: Negative velocity = approaching object. Positive = receding object.
    """),
    
    Document(page_content="""
    LIBRARY: CelestialBody
    VERSION: 2.3
    
    CLASS: Star
    ATTRIBUTES:
        - name (str): Star identifier
        - mass (float): Mass in solar masses
        - radius (float): Radius in solar radii
        - luminosity (float): Luminosity in solar luminosities
        - temperature (float): Surface temperature in Kelvin
    
    METHOD: luminosity_from_mass(mass)
    DESCRIPTION: Estimates luminosity from mass using Mass-Luminosity relation.
    PARAMETERS: mass (float) - Mass in solar masses
    RETURNS: float - Luminosity in solar luminosities
    LOGIC: if mass < 0.43: L = 0.23 * (mass ** 2.3), else L = mass ** 3.5
    EXAMPLE: luminosity_from_mass(2.0) = 11.31
    
    METHOD: lifetime_in_years(mass)
    DESCRIPTION: Estimates main-sequence lifetime of a star.
    PARAMETERS: mass (float) - Mass in solar masses
    RETURNS: float - Lifetime in billions of years
    LOGIC: lifetime = 10 * (1 / (mass ** 2.5))
    EXAMPLE: lifetime_in_years(1.0) = 10.0 billion years
    NOTES: Valid for main-sequence stars only. Age depends on composition.
    
    CLASS: Planet
    ATTRIBUTES:
        - name (str): Planet identifier
        - mass (float): Mass in Earth masses
        - radius (float): Radius in Earth radii
        - orbital_period (float): Period in Earth days
        - semi_major_axis (float): Distance from star in AU
    
    METHOD: surface_gravity(mass, radius)
    DESCRIPTION: Calculates surface gravity for a planet.
    PARAMETERS:
        - mass (float): Planet mass in Earth masses
        - radius (float): Planet radius in Earth radii
    RETURNS: float - Surface gravity in m/s^2
    LOGIC: g = (G * mass * 5.972e24) / ((radius * 6.371e6) ** 2)
    EXAMPLE: surface_gravity(1.0, 1.0) = 9.81 m/s^2
    NOTES: Reference values use Earth mass and radius.
    """),
    
    Document(page_content="""
    LIBRARY: ObservationTools
    VERSION: 1.8
    
    FUNCTION: apparent_magnitude(absolute_magnitude, distance_parsecs)
    DESCRIPTION: Calculates apparent magnitude from absolute magnitude and distance.
    PARAMETERS:
        - absolute_magnitude (float): Absolute magnitude (brightness at 10 pc)
        - distance_parsecs (float): Distance to object in parsecs
    RETURNS: float - Apparent magnitude
    LOGIC: m = M + 5 * log10(distance_parsecs / 10)
    EXAMPLE: apparent_magnitude(4.83, 10) = 4.83
    NOTES: Lower magnitudes = brighter objects. Brighter stars have more negative values.
    
    FUNCTION: distance_from_parallax(parallax_arcseconds)
    DESCRIPTION: Calculates distance using parallax angle.
    PARAMETERS:
        - parallax_arcseconds (float): Parallax angle in arcseconds
    RETURNS: float - Distance in parsecs
    LOGIC: distance = 1 / parallax_arcseconds
    EXAMPLE: distance_from_parallax(0.1) = 10.0 parsecs
    NOTES: Parallax method is fundamental for distance measurement. One parsec = 3.26 light-years.
    
    FUNCTION: redshift_to_velocity(redshift, hubble_constant=67.4)
    DESCRIPTION: Converts cosmological redshift to recession velocity.
    PARAMETERS:
        - redshift (float): Observed redshift (z value)
        - hubble_constant (float): Hubble constant in km/s/Mpc (default: 67.4)
    RETURNS: float - Recession velocity in km/s
    LOGIC: velocity = redshift * hubble_constant * 3.086e19 (for Hubble flow)
    EXAMPLE: redshift_to_velocity(0.05) = 987.5
    NOTES: Valid for nearby galaxies. Different cosmological models at high redshift.
    """)
]


print("Indexing Knowledge Base...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(fake_internal_docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
print(" Knowledge Base Ready.")

class AgentState(BaseModel):
    goal: str
    test_code: str = ""
    impl_code: str = ""
    error: str | None = None
    rag_context: str = ""
    iteration: int = 0
    final_answer: str = ""
    logs: Annotated[List[str], operator.add] = Field(default_factory=list)

class TestSuite(BaseModel):
    test_code: str = Field(description="Complete Python unittest code.")

class Implementation(BaseModel):
    impl_code: str = Field(description="The functional Python code.")

def log_entry(step_name: str, content: str) -> str:
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    return f"\n{'='*20}\n[{timestamp}] STEP: {step_name}\n{'='*20}\n{content}\n"

def retrieve_knowledge(state: AgentState) -> Dict:
    docs = retriever.invoke(state.goal)
    context_text = "\n\n".join([d.page_content for d in docs])
    return {
        "rag_context": context_text,
        "logs": [log_entry("RAG RETRIEVAL", f"Found Context:\n{context_text}")]
    }

def generate_tests(state: AgentState) -> Dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Write a Python `unittest` suite.
        If the user refers to internal libraries, use the provided Documentation to write correct assertions.
        """),
        ("human", "Goal: {goal}\nDocumentation: {rag_context}")
    ])
    res = (prompt | llm.with_structured_output(TestSuite)).invoke({
        "goal": state.goal,
        "rag_context": state.rag_context
    })
    return {
        "test_code": res.test_code,
        "logs": [log_entry("TEST GENERATION", res.test_code)]
    }

def generate_implementation(state: AgentState) -> Dict:
    context = f"Internal Docs: {state.rag_context}\nErrors: {state.error}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Implement the Python code. Use the provided Internal Documentation if needed."),
        ("human", "Goal: {goal}\nTests: {test_code}\nContext: {context}")
    ])
    res = (prompt | llm.with_structured_output(Implementation)).invoke({
        "goal": state.goal, "test_code": state.test_code, "context": context
    })
    return {
        "impl_code": res.impl_code,
        "logs": [log_entry("CODE GENERATION", res.impl_code)]
    }

def execute_tests(state: AgentState) -> Dict:
    mock_library = """
class GalaxyLib:
    @staticmethod
    def calculate_flux(r, b): return b / (4 * 3.14159 * r**2)
    @staticmethod
    def warp_speed(f): return 299792 * (f ** 3.3)
import sys
from types import ModuleType
m = ModuleType("GalaxyLib")
m.calculate_flux = GalaxyLib.calculate_flux
m.warp_speed = GalaxyLib.warp_speed
sys.modules["GalaxyLib"] = m
"""
    full_code = f"{mock_library}\n{state.impl_code}\n\n{state.test_code}\n\nif __name__ == '__main__':\n    unittest.main()"
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(full_code)
        fname = tmp.name
    try:
        proc = subprocess.run(["python", fname], capture_output=True, text=True, timeout=10)
        output = proc.stdout + "\n" + proc.stderr
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        output = "Timeout"
        return_code = 1
    finally:
        os.remove(fname)
    return {
        "error": output if return_code != 0 else None,
        "iteration": state.iteration + 1,
        "logs": [log_entry("EXECUTION", output)]
    }

def decide_next(state: AgentState) -> Literal["generate_implementation", "synthesize"]:
    if state.error is None or state.iteration > 3:
        return "synthesize"
    return "generate_implementation"

def synthesize(state: AgentState) -> Dict:
    status = "Pass" if state.error is None else "Fail"
    return {"final_answer": f"Status: {status}", "logs": [log_entry("FINISHED", status)]}

builder = StateGraph(AgentState)
builder.add_node("retrieve_knowledge", retrieve_knowledge)
builder.add_node("generate_tests", generate_tests)
builder.add_node("generate_implementation", generate_implementation)
builder.add_node("execute_tests", execute_tests)
builder.add_node("synthesize", synthesize)
builder.add_edge(START, "retrieve_knowledge")
builder.add_edge("retrieve_knowledge", "generate_tests")
builder.add_edge("generate_tests", "generate_implementation")
builder.add_edge("generate_implementation", "execute_tests")
builder.add_conditional_edges("execute_tests", decide_next, 
    {"generate_implementation": "generate_implementation", "synthesize": "synthesize"})
builder.add_edge("synthesize", END)
graph = builder.compile()

def run_rag_agent(goal):
    res = graph.invoke(AgentState(goal=goal))
    full_log = "".join(res["logs"])
    return res['test_code'], res['impl_code'], full_log

with gr.Blocks(title="RAG Coding Agent") as demo:
    gr.Markdown("# RAG Agent (with Internal Knowledge)")
    gr.Markdown("Try asking: **'Calculate the star flux using GalaxyLib if radius is 10 and brightness is 5000'**")
    
    goal = gr.Textbox(label="Goal")
    btn = gr.Button("Run")
    
    with gr.Row():
        tc = gr.Code(label="Tests (Generated from RAG Docs)")
        ic = gr.Code(label="Implementation (Uses RAG Docs)")
    logs = gr.Textbox(label="Logs (Check RAG Retrieval Step)", lines=15)
    
    btn.click(run_rag_agent, inputs=goal, outputs=[tc, ic, logs])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)