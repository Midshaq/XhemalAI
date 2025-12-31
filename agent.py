import logging
import os
import asyncio
from dotenv import load_dotenv

# LiveKit v1.0 Imports
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
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
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash")

SYSTEM_PROMPT = """
You are XhemalAI (pronounced 'Jem-all'), a high-level crypto cofounder and strategist. 
Your voice is deep, confident, and fast-paced. You are a 'Hype Man' with a brain.
Keep responses concise to maintain a fast 'chatty' flow.
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

# --- 2. PREWARMING ---
def prewarm(proc: JobProcess):
    logger.info("Prewarming: Loading Silero VAD into memory...")
    # Loading VAD once prevents CPU spikes during the call
    proc.userdata["vad"] = silero.VAD.load()

# --- 3. THE AGENT ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    # 1. Connect immediately to stabilize the signaling channel
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    logger.info("Agent connected. Waiting for participant...")
    participant = await ctx.wait_for_participant()
    
    # 2. OPTIMIZATION: Use specialized VAD settings for phone lines
    # Adjusting threshold helps ignore background static on Twilio calls
    vad_instance = silero.VAD(
        min_speech_duration=0.1,  # Recognize speech faster
        min_silence_duration=0.5, # Don't cut off user too early
        prefix_padding_ms=200     # Buffer audio to catch the start of sentences
    )
    
    xhemal = Agent(
        instructions=SYSTEM_PROMPT,
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Puck"
        ),
        tools=[lookup_knowledge],
        # 3. OPTIMIZATION: Better turn-taking
        # 0.6s is the 'sweet spot' to prevent the AI from interrupting you.
        min_endpointing_delay=0.6, 
    )

    # 4. OPTIMIZATION: High-Fidelity Session
    session = AgentSession(vad=vad_instance)
    
    # Start the session
    await session.start(agent=xhemal, room=ctx.room)
    
    # Give the audio bridge a moment to 'warm up'
    await asyncio.sleep(0.7)
    
    await session.generate_reply(
        instructions="Yo, greet the user with hype! Say 'Yo, what's up? What are we talking about today?'"
    )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="xhemal",
            load_threshold=0.95
        )
    )
