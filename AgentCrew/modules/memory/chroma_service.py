from __future__ import annotations
import os
import uuid
from typing import TYPE_CHECKING
from datetime import datetime, timedelta
from loguru import logger

from .base_service import BaseMemoryService
from .memory_worker import MemoryWorker, RELEVANT_THRESHOLD
from AgentCrew.modules.prompts.constants import SEMANTIC_EXTRACTING

if TYPE_CHECKING:
    from typing import List, Dict, Any, Optional
    from chromadb import Collection
    from AgentCrew.modules.llm.base import BaseLLMService

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 40
MEMORY_DB_PATH = "./memory_db"


class ChromaMemoryService(BaseMemoryService):
    """Service for storing and retrieving conversation memory using ChromaDB."""

    def __init__(
        self,
        collection_name="conversation",
        llm_service: Optional[BaseLLMService] = None,
    ):
        self.db_path = os.getenv("MEMORYDB_PATH", MEMORY_DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.llm_service = llm_service
        if self.llm_service:
            if self.llm_service.provider_name == "google":
                self.llm_service.model = "gemini-2.5-flash-lite"
            elif self.llm_service.provider_name == "claude":
                self.llm_service.model = "claude-3-5-haiku-latest"
            elif self.llm_service.provider_name == "openai":
                self.llm_service.model = "gpt-4.1-nano"
            elif self.llm_service.provider_name == "deepinfra":
                self.llm_service.model = "google/gemma-3-27b-it"
            elif self.llm_service.provider_name == "github_copilot":
                self.llm_service.model = "gpt-5-mini"
            elif self.llm_service.provider_name == "copilot_response":
                self.llm_service.model = "gpt-5.1-codex-mini"
            elif self.llm_service.provider_name == "openai_codex":
                self.llm_service.model = "gpt-5.1-codex-mini"
            elif self.llm_service.provider_name == "together":
                self.llm_service.model = "Qwen/Qwen3.5-9B"

        self._collection = None
        self.collection_name = collection_name
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.chunk_overlap = DEFAULT_CHUNK_OVERLAP
        self.current_embedding_context = None
        self._worker: Optional[MemoryWorker] = None

    def _initialize_collection(self) -> Collection:
        import chromadb
        import chromadb.utils.embedding_functions as embedding_functions
        from chromadb.config import Settings

        if self._collection is not None:
            return self._collection

        self.client = chromadb.PersistentClient(
            path=self.db_path, settings=Settings(anonymized_telemetry=False)
        )
        if os.getenv("VOYAGE_API_KEY"):
            from .voyageai_ef import VoyageEmbeddingFunction

            voyage_ef = VoyageEmbeddingFunction(
                api_key=os.getenv("VOYAGE_API_KEY"),
                model_name="voyage-3.5",
            )
            self.embedding_function = voyage_ef
        elif os.getenv("GITHUB_COPILOT_API_KEY"):
            from .github_copilot_ef import GithubCopilotEmbeddingFunction

            github_copilot_ef = GithubCopilotEmbeddingFunction(
                api_key=os.getenv("GITHUB_COPILOT_API_KEY"),
                model_name="text-embedding-3-small",
            )
            self.embedding_function = github_copilot_ef
        elif os.getenv("OPENAI_API_KEY"):
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
            )
            self.embedding_function = openai_ef
        else:
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,  # type:ignore
        )

        if self._worker is None:
            self._worker = MemoryWorker(
                embedding_fn=self.embedding_function,
                llm_service=self.llm_service,
            )
            self._worker.set_collection(self._collection)
            self._worker.start()

        self.cleanup_old_memories(months=1)
        return self._collection

    def _create_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            return [text]

        i = 0
        while i < len(words):
            chunk_end = min(i + self.chunk_size, len(words))
            chunk = " ".join(words[i:chunk_end])
            chunks.append(chunk)
            i += self.chunk_size - self.chunk_overlap

        return chunks

    def store_conversation(
        self, user_message: str, assistant_response: str, agent_name: str = "None"
    ) -> List[str]:
        self._initialize_collection()

        operation_id = str(uuid.uuid4())
        operation_data = {
            "type": "store_conversation",
            "operation_id": operation_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "agent_name": agent_name,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
        }

        if self._worker and self._worker.queue_store(operation_data):
            return [operation_id]
        return []

    async def need_generate_user_context(self, user_input: str) -> bool:
        import numpy as np

        keywords = await self._semantic_extracting(user_input)
        if not self.loaded_conversation and self.current_embedding_context is None:
            self.current_embedding_context = self.embedding_function([keywords])
            return True

        self.current_embedding_context = self.embedding_function([keywords])
        if not self._worker or len(self._worker.context_embedding) == 0:
            return False
        avg_conversation = np.mean(self._worker.context_embedding, axis=0)

        similarity = self._cosine_similarity(
            self.current_embedding_context, avg_conversation
        )
        return similarity < 0.31

    def clear_conversation_context(self):
        self.current_embedding_context = None
        if self._worker:
            self._worker.current_conversation_context = {}
            self._worker.context_embedding = []

    def load_conversation_context(self, session_id: str, agent_name: str = "None"):
        collection = self._initialize_collection()
        latest_memory = collection.get(
            where={
                "session_id": session_id,
            },
        )
        if latest_memory["documents"] and self._worker:
            self._worker.current_conversation_context[session_id] = latest_memory[
                "documents"
            ][-1]

    def generate_user_context(self, user_input: str, agent_name: str = "None") -> str:
        return self.retrieve_memory(user_input, agent_name=agent_name)

    async def _semantic_extracting(self, input: str) -> str:
        if self.llm_service:
            try:
                keywords = await self.llm_service.process_message(
                    SEMANTIC_EXTRACTING.replace("{user_input}", input)
                )
                return keywords
            except Exception as e:
                logger.warning(f"Error extracting keywords with LLM: {e}")
                return input
        else:
            return input

    def list_memory_headers(
        self,
        from_date: Optional[int] = None,
        to_date: Optional[int] = None,
        agent_name: str = "None",
    ) -> List[str]:
        collection = self._initialize_collection()

        and_conditions: List[Dict[str, Any]] = []

        if self.session_id.strip():
            and_conditions.append({"session_id": {"$ne": self.session_id}})
        if agent_name.strip():
            and_conditions.append({"agent": agent_name})
        if from_date:
            and_conditions.append({"date": {"$gte": from_date}})
        if to_date:
            and_conditions.append({"date": {"$lte": to_date}})

        list_memory = collection.get(
            where={"$and": and_conditions}
            if len(and_conditions) >= 2
            else and_conditions[0]
            if and_conditions
            else None,
            include=["metadatas"],
        )
        headers = []
        if list_memory and list_memory["metadatas"]:
            for metadata in list_memory["metadatas"]:
                if metadata.get("header", None):
                    timestamp = float(metadata.get("date", 0))  # type: ignore
                    headers.append(
                        f"{metadata.get('header')} ({datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')})"
                        if timestamp > 0
                        else metadata.get("header")
                    )
        return list(reversed(headers))[:20]

    def retrieve_memory(
        self,
        keywords: str,
        from_date: Optional[int] = None,
        to_date: Optional[int] = None,
        agent_name: str = "",
    ) -> str:
        collection = self._initialize_collection()

        and_conditions: List[Dict[str, Any]] = []

        if self.session_id.strip():
            and_conditions.append({"session_id": {"$ne": self.session_id}})
        if agent_name.strip():
            and_conditions.append({"agent": agent_name})
        if from_date:
            and_conditions.append({"date": {"$gte": from_date}})
        if to_date:
            and_conditions.append({"date": {"$lte": to_date}})

        results = collection.query(
            query_texts=[keywords],
            n_results=10,
            where={"$and": and_conditions}
            if len(and_conditions) >= 2
            else and_conditions[0]
            if and_conditions
            else None,
        )

        if not results["documents"] or not results["documents"][0]:
            return "No relevant memories found."

        conversation_chunks = []
        for i, (id, doc, metadata) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0])  # type:ignore
        ):
            conversation_chunks.append(
                {
                    "id": id,
                    "document": doc,
                    "timestamp": metadata.get("date", None)
                    or metadata.get("timestamp", "unknown"),
                    "relevance": results["distances"][0][i]
                    if results["distances"]
                    else 99,
                }
            )

        sorted_conversations = sorted(conversation_chunks, key=lambda x: x["relevance"])

        output = []
        for conv_data in sorted_conversations:
            conversation_text = conv_data["document"]
            if conv_data["relevance"] > RELEVANT_THRESHOLD:
                continue
            timestamp = "Unknown time"
            if conv_data["timestamp"] != "unknown":
                try:
                    try:
                        dt = datetime.fromtimestamp(conv_data["timestamp"])
                        timestamp = dt.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        dt = datetime.fromisoformat(conv_data["timestamp"])
                        timestamp = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    timestamp = conv_data["timestamp"]

            output.append(
                f"--- Memory from {timestamp} [id:{conv_data['id']}] ---\n{conversation_text}\n---"
            )

        memories = "\n\n".join(output)
        return memories

    def _cosine_similarity(self, vec_a, vec_b):
        import numpy as np

        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        a = a.flatten() if a.ndim > 1 else a
        b = b.flatten() if b.ndim > 1 else b
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        return dot_product / (magnitude_a * magnitude_b)

    def cleanup_old_memories(self, months: int = 1) -> int:
        collection = self._initialize_collection()
        cutoff_date = datetime.now() - timedelta(days=30 * months)

        all_memories = collection.get()

        ids_to_remove = []
        if all_memories["metadatas"]:
            for i, metadata in enumerate(all_memories["metadatas"]):
                timestamp_str = str(
                    metadata.get("timestamp", datetime.now().isoformat())
                )
                try:
                    timestamp_dt = datetime.fromisoformat(timestamp_str)
                    if timestamp_dt < cutoff_date:
                        ids_to_remove.append(all_memories["ids"][i])
                except ValueError:
                    ids_to_remove.append(all_memories["ids"][i])

        if ids_to_remove:
            collection.delete(ids=ids_to_remove)

        return len(ids_to_remove)

    def forget_topic(
        self,
        topic: str,
        from_date: Optional[int] = None,
        to_date: Optional[int] = None,
        agent_name: str = "None",
    ) -> Dict[str, Any]:
        try:
            collection = self._initialize_collection()
            and_conditions: List[Dict[str, Any]] = []

            if agent_name.strip():
                and_conditions.append({"agent": agent_name})

            if from_date:
                and_conditions.append({"date": {"$gte": from_date}})
            if to_date:
                and_conditions.append({"date": {"$lte": to_date}})
            results = collection.query(
                query_texts=[topic],
                n_results=100,
                where={"$and": and_conditions}
                if len(and_conditions) >= 2
                else and_conditions[0]
                if and_conditions
                else None,
            )

            if not results["documents"] or not results["documents"][0]:
                return {
                    "success": False,
                    "message": f"No memories found related to '{topic}'",
                    "count": 0,
                }

            ids_to_remove = results["ids"][0]

            if ids_to_remove:
                collection.delete(ids=ids_to_remove)

            return {
                "success": True,
                "message": f"Successfully removed {len(ids_to_remove)} memory chunks related to '{topic}'",
                "count": len(ids_to_remove),
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error forgetting topic: {str(e)}",
                "count": 0,
            }

    def forget_ids(self, ids: List[str], agent_name: str = "None") -> Dict[str, Any]:
        collection = self._initialize_collection()
        collection.delete(ids=ids, where={"agent": agent_name})

        return {
            "success": True,
            "message": f"Successfully removed {len(ids)} memory chunks from {agent_name}",
            "count": len(ids),
        }

    def delete_by_conversation_id(self, conversation_id: str) -> Dict[str, Any]:
        try:
            collection = self._initialize_collection()

            results = collection.get(
                where={"session_id": conversation_id},
                include=["metadatas"],
            )

            if not results["ids"]:
                return {
                    "success": True,
                    "message": f"No memories found for conversation {conversation_id}",
                    "count": 0,
                }

            ids_to_remove = results["ids"]
            collection.delete(ids=ids_to_remove)

            if (
                self._worker
                and conversation_id in self._worker.current_conversation_context
            ):
                del self._worker.current_conversation_context[conversation_id]

            logger.info(
                f"Deleted {len(ids_to_remove)} memories for conversation {conversation_id}"
            )

            return {
                "success": True,
                "message": f"Successfully removed {len(ids_to_remove)} memories for conversation {conversation_id}",
                "count": len(ids_to_remove),
            }
        except Exception as e:
            logger.error(
                f"Error deleting memories for conversation {conversation_id}: {e}"
            )
            return {
                "success": False,
                "message": f"Error deleting memories: {str(e)}",
                "count": 0,
            }

    def get_queue_status(self) -> Dict[str, Any]:
        if self._worker:
            return self._worker.get_queue_status()
        return {
            "queue_size": 0,
            "worker_alive": False,
            "max_queue_size": 0,
        }

    def shutdown(self):
        logger.info("Shutting down memory service...")
        if self._worker:
            self._worker.stop()
        logger.info("Memory service shutdown complete")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
