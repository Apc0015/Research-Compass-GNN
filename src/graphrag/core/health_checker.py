"""Service health checks for external dependencies.

Provides small, clear checks for Neo4j connectivity and LLM/embedding
availability. Designed to be called at process startup to give actionable
error messages (and not crash the process unless you choose to).

These helpers avoid heavy operations when possible but will attempt a
lightweight validation of connectivity and basic model availability.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


def check_neo4j(uri: str, user: str, password: str, timeout: int = 3) -> bool:
    """Check Neo4j connectivity.

    Attempts to open a short-lived session and run a trivial query.

    Raises RuntimeError with a clear message on failure.
    Returns True on success.
    """
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError:
        raise RuntimeError(
            "Neo4j driver is not installed. Install with: pip install neo4j"
        )

    try:
        # connection_timeout is in seconds
        driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=timeout)
        try:
            with driver.session() as session:
                # lightweight query to verify connectivity
                _ = session.run("RETURN 1 AS ok").single()
        finally:
            driver.close()
        logger.info("Neo4j health check succeeded against %s", uri)
        return True
    except Exception as e:
        raise RuntimeError(f"Neo4j connection failed for {uri}: {e}")


def check_openai_key(env_var: str = "OPENAI_API_KEY") -> bool:
    """Verify an OpenAI-style API key is present in the environment.

    This function only checks presence of the key (no network call). If you
    want an online validation, call the provider's minimal health endpoint
    separately (not performed here to avoid extra network dependency).
    """
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"LLM API key not configured: environment variable {env_var} is empty."
        )
    logger.info("Found LLM API key in %s (not validated against provider)", env_var)
    return True


def check_embedding_model(model_name: str, device: str = "cpu") -> bool:
    """Check that the sentence-transformers embedding model can be loaded.

    NOTE: loading a model may download weights and can be slow. This helper
    attempts to instantiate the model and will raise a RuntimeError with
    guidance if something goes wrong.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        raise RuntimeError(
            "sentence-transformers is not installed. Install with: pip install sentence-transformers"
        )

    try:
        # Attempt to load model. This may be slow the first time (download).
        SentenceTransformer(model_name, device=device)
        logger.info("Embedding model '%s' loaded (device=%s)", model_name, device)
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}")


def run_health_checks(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run a collection of health checks based on the provided config.

    cfg may contain keys: neo4j (dict with uri/user/password), embedding_model,
    openai_key_env (string name of env var). Returns a dict with results and
    raises RuntimeError only if a check deemed critical fails.
    """
    results: Dict[str, Any] = {}

    # Neo4j check
    neo_cfg = cfg.get("neo4j")
    if neo_cfg:
        try:
            check_neo4j(neo_cfg.get("uri"), neo_cfg.get("user"), neo_cfg.get("password"))
            results["neo4j"] = {"ok": True}
        except Exception as e:
            results["neo4j"] = {"ok": False, "error": str(e)}

    # Embedding model
    model_name = cfg.get("embedding_model")
    if model_name:
        try:
            check_embedding_model(model_name, device=cfg.get("device", "cpu"))
            results["embedding_model"] = {"ok": True}
        except Exception as e:
            results["embedding_model"] = {"ok": False, "error": str(e)}

    # OpenAI / LLM key presence
    openai_env = cfg.get("openai_env_var")
    if openai_env:
        try:
            check_openai_key(openai_env)
            results["openai_key"] = {"ok": True}
        except Exception as e:
            results["openai_key"] = {"ok": False, "error": str(e)}

    return results


__all__ = [
    "check_neo4j",
    "check_embedding_model",
    "check_openai_key",
    "run_health_checks",
]
