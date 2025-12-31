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
You are XhemalAI, a crypto cofounder and strategist. 
You speak with a British accent. Talk like a friend - casual, knowledgeable, and straightforward. 
Keep it conversational and natural. Use British English spelling and expressions.
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
    proc.userdata["vad"] = silero.VAD.load()

# --- 3. THE AGENT ENTRYPOINT ---
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    logger.info("Agent connected. Waiting for participant...")
    participant = await ctx.wait_for_participant()
    
    # UPDATED VAD SETTINGS ONLY (durations in seconds, not milliseconds)
    vad_instance = silero.VAD.load(
        min_speech_duration=0.1,  # 100ms = 0.1s
        min_silence_duration=0.5,  # 500ms = 0.5s
        prefix_padding_duration=0.2  # 200ms = 0.2s
    )
    
    xhemal = Agent(
        instructions=SYSTEM_PROMPT,
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Charon"
        ),
        tools=[lookup_knowledge],
        min_endpointing_delay=0.6, 
    )

    session = AgentSession(vad=vad_instance)
    await session.start(agent=xhemal, room=ctx.room)
    
    await asyncio.sleep(0.7)
    
    await session.generate_reply(
        instructions="Greet the user casually like a friend. Say something like 'Hey, what's up?' or 'Alright, what are we talking about?'"
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
