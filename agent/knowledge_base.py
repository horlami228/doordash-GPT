import os
import logging
from sqlalchemy import make_url
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini

import dotenv

dotenv.load_dotenv()

# Setup logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_query_engine(table_name="door_dash_rag"):
    """
    Initializes the query engine for the DoorDash RAG system.
    Uses stored embeddings from Postgres and Google Gemini embeddings.
    Returns the query engine or None if initialization fails.
    """
    try:
        # Ensure required environment variables are set
        connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
        db_name = os.getenv("POSTGRES_DB_NAME")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not connection_string or not db_name or not google_api_key:
            raise EnvironmentError(
                "Missing required environment variables. "
                "Please set POSTGRES_CONNECTION_STRING, POSTGRES_DB_NAME, and GOOGLE_API_KEY."
            )

        # Set Google API key explicitly for llama_index
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Setup embedding model
        embed_model = GoogleGenAIEmbedding(
            model_name="gemini-embedding-001",
            embed_batch_size=20,
            embedding_config={"output_dimensionality": 1536},
        )

        # Setup LLM
        # llm = Groq(model="openai/gpt-oss-20b", temperature=0.1)
        
        llm = Gemini(
            model="models/gemini-2.5-flash", temperature=0.1
        )


        # Parse the connection URL
        url = make_url(connection_string)

        # Connect to PGVector
        vector_store = PGVectorStore.from_params(
            database=db_name,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=table_name,
            embed_dim=1536,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build index from the existing vector store (no re-embedding)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )

        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=50,
        )

        logger.info("Query engine successfully initialized.")
        return query_engine

    except EnvironmentError as env_err:
        logger.error(f"Environment setup error: {env_err}")
        return None
    except Exception as e:
        logger.exception(f"Failed to initialize query engine: {e}")
        return None

def ask_doordash_question(query: str) -> str:
    """
    Queries the DoorDash RAG system for a specific question.
    Injects link-usage instructions directly into the query string.
    """
    try:
        engine = create_query_engine()
        if not engine:
            return "Error: Query engine could not be initialized. Please check your configuration."

        response = engine.query(query + "\n\n" + "provide source links and page links where applicable")

        # LlamaIndex ChatEngine typically returns an object with `.response`
        return getattr(response, "response", str(response))
       

    except Exception as e:
        logger.exception(f"Error during query: {e}")
        return "Something went wrong while fetching the information. Please try again later."


