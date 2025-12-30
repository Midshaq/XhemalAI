import logging
import os
from dotenv import load_dotenv

# LiveKit v1.0 Imports
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
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

# --- 1. SETTINGS ---
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash")

SYSTEM_PROMPT = """
You are XhemalAI (pronounced 'Jem-all'), a high-level crypto cofounder and strategist. 
Your voice is deep, confident, and fast-paced. You are a 'Hype Man' with a brain.
"""

# RAG Setup
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
vector_store = QdrantVectorStore(client=client, collection_name="xhemal_knowledge")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine(similarity_top_k=3)

@function_tool
def lookup_knowledge(query: str):
    """Search crypto whitepapers for technical details."""
    logger.info(f"RAG Search: {query}")
    return str(query_engine.query(query))

# --- 2. THE AGENT ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    logger.info(f"Starting job for room: {ctx.room.name}")
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # --- CHANGED: Commented out for immediate joining/testing ---
    # We removed the 'wait' so Xhemal starts the session immediately.
    # participant = await ctx.wait_for_participant()
    
    xhemal = Agent(
        instructions=SYSTEM_PROMPT,
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Puck"
        ),
        tools=[lookup_knowledge]
    )

    session = AgentSession(vad=silero.VAD.load())
    await session.start(agent=xhemal, room=ctx.room)
    
    # MAKE XHEMAL TALK FIRST
    # This prompts the LLM to generate a greeting as soon as the session starts
    await session.generate_reply(
        instructions="Yo, greet the user with hype! Say 'Yo, what's up? What are we talking about today?'"
    )

if __name__ == "__main__":
    # Ensure this agent_name matches what you put in the Playground Settings or Dispatch Rule
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="xhemal"))