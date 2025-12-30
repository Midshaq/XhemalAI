import logging
import os
from dotenv import load_dotenv

# LiveKit v1.0 Imports
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess, # Added for prewarming
    WorkerOptions,
    Worker,
    cli,
    llm,
    function_tool,
)
from livekit.plugins import google, silero

# LlamaIndex (RAG)
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()
logger = logging.getLogger("xhemal-agent")

# --- 1. SETTINGS & GLOBAL INITIALIZATION ---
# Moving these to the top level ensures they are ready when the process starts
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash")

SYSTEM_PROMPT = """
You are XhemalAI (pronounced 'Jem-all'), a high-level crypto cofounder and strategist. 
Your voice is deep, confident, and fast-paced. You are a 'Hype Man' with a brain.
"""

# Initialize RAG components globally to avoid re-initializing per call
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
vector_store = QdrantVectorStore(client=client, collection_name="xhemal_knowledge")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine(similarity_top_k=3)

@function_tool
def lookup_knowledge(query: str):
    """Search crypto whitepapers for technical details."""
    logger.info(f"RAG Search: {query}")
    return str(query_engine.query(query))

# --- 2. PREWARMING ---
# This loads the heavy VAD model into memory before the worker starts taking jobs
def prewarm(proc: JobProcess):
    logger.info("Prewarming worker: Loading Silero VAD...")
    proc.userdata["vad"] = silero.VAD.load()

# --- 3. THE AGENT ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    logger.info(f"Starting job for room: {ctx.room.name}")
    
    # Use the pre-loaded VAD from userdata instead of loading it now
    vad_instance = ctx.proc.userdata["vad"]
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    xhemal = Agent(
        instructions=SYSTEM_PROMPT,
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Puck"
        ),
        tools=[lookup_knowledge]
    )

    # AgentSession now uses the cached VAD instance
    session = AgentSession(vad=vad_instance)
    await session.start(agent=xhemal, room=ctx.room)
    
    # Immediate greeting
    await session.generate_reply(
        instructions="Yo, greet the user with hype! Say 'Yo, what's up? What are we talking about today?'"
    )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,      # Added prewarm to handle heavy loading
            agent_name="xhemal",
            load_threshold=0.95       # Increased threshold so it doesn't mark as full
        )
    )
