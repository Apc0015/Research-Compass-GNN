#!/usr/bin/env python3
"""
Unified Recommendation Engine - Comprehensive paper recommendation system.

This module consolidates all recommendation functionality from:
- recommendation_engine.py (hybrid recommendations)
- personalized_recommendations.py (GNN-powered personalized recommendations)
- collaborative_filtering.py (multi-user collaborative filtering)

Provides a complete recommendation system with:
- Hybrid recommendations (content + citation + GNN signals)
- Personalized user profiles and recommendations
- Collaborative filtering across users
- Author recommendations
- Research direction suggestions
- Community detection
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import logging
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class UnifiedRecommendationEngine:
    """
    Comprehensive recommendation system combining all recommendation strategies.

    Features:
    - Hybrid recommendations combining multiple signals
    - GNN-powered personalized recommendations
    - Collaborative filtering
    - Author and research direction recommendations
    - Community detection

    Example:
        >>> engine = UnifiedRecommendationEngine(graph_manager, gnn_manager, vector_search)
        >>> engine.create_user_profile("user1", ["paper1"], ["paper1"], ["ML"])
        >>> recs = engine.recommend_papers("user1", n=10)
    """

    def __init__(self, graph_manager, embedder=None, gnn_manager=None, vector_search=None):
        """
        Initialize the unified recommendation engine.

        Args:
            graph_manager: GraphManager for graph operations
            embedder: Optional embedder for content-based recommendations
            gnn_manager: Optional GNNManager for GNN predictions
            vector_search: Optional VectorSearch for semantic similarity
        """
        self.graph = graph_manager
        self.embedder = embedder
        self.gnn = gnn_manager
        self.gnn_manager = gnn_manager  # Alias for compatibility
        self.vector_search = vector_search
        self.user_profiles = {}  # user_id -> profile data

        logger.info("Unified Recommendation Engine initialized")

    # ========== User Profile Management ==========

    def create_user_profile(
        self,
        user_id: str,
        read_papers: List[str],
        liked_papers: List[str],
        interests: List[str]
    ) -> Dict:
        """
        Create user embedding in graph space.

        Combines user's reading history, explicit likes, and stated interests
        to create a rich user profile for personalized recommendations.

        Args:
            user_id: Unique user identifier
            read_papers: Papers user has read
            liked_papers: Papers user explicitly liked (subset of read_papers)
            interests: Topic keywords (e.g., ["deep learning", "NLP"])

        Returns:
            {
                'user_embedding': np.ndarray,
                'interest_vector': np.ndarray,
                'profile_summary': str
            }
        """
        profile = {
            'user_embedding': None,
            'interest_vector': None,
            'profile_summary': ''
        }

        try:
            embeddings = []
            weights = []

            # Get embeddings for liked papers (higher weight)
            for paper_id in liked_papers:
                if self.gnn_manager and hasattr(self.gnn_manager, 'embedder'):
                    emb = self.gnn_manager.embedder.get_embedding(paper_id)
                    if emb is not None:
                        embeddings.append(emb)
                        weights.append(2.0)  # Double weight for liked papers

            # Get embeddings for read papers (normal weight)
            for paper_id in read_papers:
                if paper_id not in liked_papers:
                    if self.gnn_manager and hasattr(self.gnn_manager, 'embedder'):
                        emb = self.gnn_manager.embedder.get_embedding(paper_id)
                        if emb is not None:
                            embeddings.append(emb)
                            weights.append(1.0)

            # Create user embedding as weighted average
            if embeddings:
                embeddings_array = np.array(embeddings)
                weights_array = np.array(weights).reshape(-1, 1)

                user_embedding = np.average(embeddings_array, axis=0, weights=weights_array.flatten())
                profile['user_embedding'] = user_embedding
            else:
                # Fallback: random embedding
                logger.warning(f"No embeddings found for user {user_id}, using random")
                profile['user_embedding'] = np.random.randn(128)

            # Create interest vector from keywords
            interest_embeddings = []
            if self.vector_search:
                for interest in interests:
                    try:
                        # Get embedding for interest keyword
                        emb = self.vector_search.embed_texts([interest])[0]
                        interest_embeddings.append(emb)
                    except Exception as e:
                        logger.warning(f"Failed to embed interest '{interest}': {e}")

            if interest_embeddings:
                profile['interest_vector'] = np.mean(interest_embeddings, axis=0)

            # Create summary
            profile['profile_summary'] = (
                f"User {user_id}: {len(read_papers)} papers read, "
                f"{len(liked_papers)} liked, interests in {', '.join(interests)}"
            )

            # Store profile
            self.user_profiles[user_id] = {
                'read_papers': set(read_papers),
                'liked_papers': set(liked_papers),
                'interests': interests,
                'embedding': profile['user_embedding'],
                'interest_vector': profile['interest_vector']
            }

            logger.info(f"Created profile for user {user_id}")

        except Exception as e:
            logger.error(f"Error creating user profile: {e}")

        return profile

    # ========== Paper Recommendations ==========

    def recommend_papers(
        self,
        user_id: str,
        n: int = 10,
        diversity_weight: float = 0.3
    ) -> List[Dict]:
        """
        Recommend papers using hybrid approach (GNN + embeddings + user profile).

        Strategy:
        - Compute similarity between user embedding and paper embeddings
        - Use GNN link prediction to find relevant papers
        - Apply diversity penalty to avoid filter bubble
        - Consider recency and impact

        Args:
            user_id: User identifier
            n: Number of recommendations
            diversity_weight: Weight for diversity (0-1), higher = more diverse

        Returns:
            [
                {
                    'paper_id': str,
                    'title': str,
                    'relevance_score': float,
                    'explanation': str,
                    'recommendation_type': 'similar|trending|exploratory',
                    'predicted_interest': float
                },
                ...
            ]
        """
        recommendations = []

        try:
            # Get user profile
            if user_id not in self.user_profiles:
                logger.warning(f"User {user_id} has no profile")
                return recommendations

            profile = self.user_profiles[user_id]
            user_embedding = profile['embedding']
            read_papers = profile['read_papers']

            # Collect candidate papers
            candidates = {}  # paper_id -> score

            # 1. Embedding-based recommendations
            if self.gnn_manager and hasattr(self.gnn_manager, 'embedder'):
                try:
                    all_embeddings = self.gnn_manager.embedder.embeddings

                    for paper_id, paper_emb in all_embeddings.items():
                        if paper_id in read_papers:
                            continue  # Skip already read

                        # Cosine similarity
                        similarity = np.dot(user_embedding, paper_emb) / (
                            np.linalg.norm(user_embedding) * np.linalg.norm(paper_emb) + 1e-8
                        )

                        candidates[paper_id] = candidates.get(paper_id, 0.0) + 0.4 * similarity

                except Exception as e:
                    logger.warning(f"Embedding-based recommendation failed: {e}")

            # 2. GNN link prediction from liked papers
            if self.gnn_manager:
                for liked_paper in list(profile['liked_papers'])[:3]:  # Top 3 liked
                    try:
                        predictions = self.gnn_manager.predict_paper_citations(liked_paper, top_k=20)

                        for paper_id, score in predictions:
                            if paper_id not in read_papers:
                                candidates[paper_id] = candidates.get(paper_id, 0.0) + 0.3 * score

                    except Exception as e:
                        logger.warning(f"GNN prediction failed for {liked_paper}: {e}")

            # 3. Interest-based recommendations
            for interest in profile['interests']:
                try:
                    matches = self.graph.search_entities(interest, limit=10)

                    for match in matches:
                        paper_id = match.get('name') or match.get('id')
                        if paper_id and paper_id not in read_papers:
                            candidates[paper_id] = candidates.get(paper_id, 0.0) + 0.2

                except Exception as e:
                    logger.warning(f"Interest-based search failed for '{interest}': {e}")

            # 4. Apply diversity penalty
            if diversity_weight > 0:
                candidates = self._apply_diversity_penalty(
                    candidates,
                    profile,
                    diversity_weight
                )

            # Rank and format recommendations
            ranked_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

            for paper_id, score in ranked_candidates[:n]:
                # Determine recommendation type
                rec_type = self._classify_recommendation_type(paper_id, profile, score)

                # Get paper title (if available)
                try:
                    paper_info = self.graph.search_entities(paper_id, limit=1)
                    title = paper_info[0].get('name', paper_id) if paper_info else paper_id
                except:
                    title = paper_id

                # Generate explanation
                explanation = self._generate_explanation(paper_id, profile, score, rec_type)

                recommendations.append({
                    'paper_id': paper_id,
                    'title': title,
                    'relevance_score': min(1.0, score),
                    'explanation': explanation,
                    'recommendation_type': rec_type,
                    'predicted_interest': min(1.0, score)
                })

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")

        return recommendations

    def recommend_papers_for_user(
        self,
        read_papers: List[str],
        interests: List[str],
        top_k: int = 20
    ) -> List[Tuple[str, float, str]]:
        """
        Return list of (paper_id, score, reason) using hybrid approach.

        Legacy method for backwards compatibility.
        Combines embedding similarity, citation neighbors, and GNN predictions.

        Args:
            read_papers: Papers already read
            interests: Topic keywords
            top_k: Number of recommendations

        Returns:
            [(paper_id, score, reason), ...]
        """
        candidates: Dict[str, float] = {}
        reasons: Dict[str, str] = {}

        # 1) Content-based via embeddings
        if self.embedder:
            for pid in read_papers:
                try:
                    sims = self.embedder.find_similar_by_embedding(pid, top_k=top_k)
                    for cid, score in sims:
                        candidates[cid] = candidates.get(cid, 0.0) + 0.35 * score
                        reasons[cid] = 'content'
                except Exception:
                    logger.exception("Embedder failed for %s", pid)

        # 2) Citation-based: include cited and citing papers
        try:
            for pid in read_papers:
                neighbors = self.graph.query_neighbors(pid, max_depth=1)
                for n in neighbors:
                    nid = n.get('name')
                    candidates[nid] = candidates.get(nid, 0.0) + 0.25
                    reasons[nid] = 'citation'
        except Exception:
            logger.exception("Citation expansion failed")

        # 3) GNN-based link predictions
        if self.gnn:
            try:
                for pid in read_papers:
                    preds = self.gnn.predict_paper_citations(pid, top_k=top_k)
                    for cid, score in preds:
                        candidates[cid] = candidates.get(cid, 0.0) + 0.25 * score
                        reasons[cid] = 'gnn'
            except Exception:
                logger.exception("GNN predictions failed")

        # 4) Interest keywords boost
        if interests:
            for topic in interests:
                try:
                    hits = self.graph.search_entities(topic, limit=top_k)
                    for h in hits:
                        pid = h.get('name')
                        candidates[pid] = candidates.get(pid, 0.0) + 0.15
                        reasons[pid] = 'interest'
                except Exception:
                    logger.exception("Interest-based search failed for %s", topic)

        # Build ranked list
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        results = [(pid, float(score), reasons.get(pid, 'hybrid')) for pid, score in ranked[:top_k]]
        return results

    # ========== Author Recommendations ==========

    def recommend_authors_to_follow(self, user_id: str, n: int = 5) -> List[Dict]:
        """
        Recommend authors based on user interests.

        Finds authors who write about topics the user is interested in,
        weighted by author productivity and impact.

        Args:
            user_id: User identifier
            n: Number of author recommendations

        Returns:
            [
                {
                    'author_id': str,
                    'author_name': str,
                    'relevance_score': float,
                    'recent_papers': [paper_ids],
                    'shared_interests': [topics],
                    'explanation': str
                },
                ...
            ]
        """
        recommendations = []

        try:
            if user_id not in self.user_profiles:
                return recommendations

            profile = self.user_profiles[user_id]

            # Find authors related to user's interests
            author_scores = defaultdict(float)
            author_papers = defaultdict(list)

            for interest in profile['interests']:
                try:
                    # Search for papers on this topic
                    papers = self.graph.search_entities(interest, limit=20)

                    # For each paper, find its authors
                    for paper in papers:
                        paper_id = paper.get('name') or paper.get('id')

                        # Get authors (simplified - in real implementation, query AUTHORED_BY)
                        author_scores[f"author_{paper_id}"] += 0.5
                        author_papers[f"author_{paper_id}"].append(paper_id)

                except Exception as e:
                    logger.warning(f"Author search failed for '{interest}': {e}")

            # Rank authors
            ranked_authors = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)

            for author_id, score in ranked_authors[:n]:
                recommendations.append({
                    'author_id': author_id,
                    'author_name': author_id,  # Would be actual name from DB
                    'relevance_score': min(1.0, score),
                    'recent_papers': author_papers[author_id][:5],
                    'shared_interests': profile['interests'],
                    'explanation': f"Writes about {', '.join(profile['interests'][:2])}"
                })

        except Exception as e:
            logger.error(f"Error recommending authors: {e}")

        return recommendations

    def recommend_papers_by_author(self, author_id: str, top_k: int = 10):
        """
        Recommend papers by a specific author.

        Args:
            author_id: Author identifier
            top_k: Number of recommendations

        Returns:
            List of paper recommendations with metadata
        """
        recs = []
        try:
            # Find co-authors and top cited papers by same authors
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    q = """
                    MATCH (a:Author {id:$id})-[:AUTHORED_BY]-(p:Entity)
                    WITH p
                    OPTIONAL MATCH (p)<-[r:CITES]-()
                    RETURN p.id as id, p.name as name, count(r) as citations
                    ORDER BY citations DESC LIMIT $k
                    """
                    res = session.run(q, id=author_id, k=top_k)
                    for r in res:
                        recs.append({
                            'id': r['id'],
                            'title': r.get('name'),
                            'citations': int(r.get('citations', 0))
                        })
        except Exception:
            logger.exception("Failed to recommend by author")

        return recs

    # ========== Research Direction Recommendations ==========

    def recommend_research_directions(self, user_id: str, n: int = 5) -> List[Dict]:
        """
        Suggest unexplored research directions based on user profile.

        Identifies gaps and opportunities at the intersection of user's
        interests and emerging topics.

        Args:
            user_id: User identifier
            n: Number of research direction suggestions

        Returns:
            [
                {
                    'direction': str,
                    'description': str,
                    'gap_score': float,
                    'related_papers': [papers],
                    'potential_impact': float
                },
                ...
            ]
        """
        directions = []

        try:
            if user_id not in self.user_profiles:
                return directions

            profile = self.user_profiles[user_id]
            interests = profile['interests']

            # Look for combinations of interests (interdisciplinary opportunities)
            if len(interests) >= 2:
                for i, interest1 in enumerate(interests):
                    for interest2 in interests[i+1:]:
                        # Search for papers combining both interests
                        try:
                            papers1 = set(p.get('name') for p in self.graph.search_entities(interest1, limit=20))
                            papers2 = set(p.get('name') for p in self.graph.search_entities(interest2, limit=20))

                            overlap = papers1 & papers2

                            # If small overlap, it's a potential gap
                            if len(overlap) < 5:
                                gap_score = 1.0 - (len(overlap) / 10.0)

                                directions.append({
                                    'direction': f"{interest1} + {interest2}",
                                    'description': f"Combining {interest1} with {interest2}",
                                    'gap_score': gap_score,
                                    'related_papers': list(overlap),
                                    'potential_impact': gap_score * 0.8
                                })

                        except Exception as e:
                            logger.warning(f"Gap analysis failed for {interest1}/{interest2}: {e}")

            # Sort by gap score (higher = more unexplored)
            directions.sort(key=lambda x: x['gap_score'], reverse=True)

        except Exception as e:
            logger.error(f"Error recommending research directions: {e}")

        return directions[:n]

    def recommend_next_papers(self, paper_id: str, top_k: int = 10):
        """
        Recommend papers to read next based on a paper.

        Uses citation chains (papers cited by cited papers).

        Args:
            paper_id: Source paper ID
            top_k: Number of recommendations

        Returns:
            List of recommendations
        """
        recs = []
        try:
            network = self.graph.query_neighbors(paper_id, max_depth=2)
            for n in network[:top_k]:
                recs.append({'id': n.get('name'), 'score': 0.5})
        except Exception:
            logger.exception("Failed to recommend next papers")
        return recs

    # ========== Collaborative Filtering ==========

    def find_similar_users(self, user_id: str, n: int = 10) -> List[Dict]:
        """
        Find users with similar reading patterns and interests.

        Args:
            user_id: User identifier
            n: Number of similar users to find

        Returns:
            [
                {
                    'user_id': str,
                    'similarity_score': float,
                    'shared_papers': [paper_ids],
                    'shared_interests': [topics]
                },
                ...
            ]
        """
        similar_users = []

        try:
            if user_id not in self.user_profiles:
                return similar_users

            user_profile = self.user_profiles[user_id]
            user_papers = user_profile['read_papers']
            user_interests = set(user_profile['interests'])

            # Compare with other users
            for other_id, other_profile in self.user_profiles.items():
                if other_id == user_id:
                    continue

                other_papers = other_profile['read_papers']
                other_interests = set(other_profile['interests'])

                # Calculate similarity (Jaccard similarity)
                paper_overlap = len(user_papers & other_papers)
                paper_union = len(user_papers | other_papers)
                paper_sim = paper_overlap / paper_union if paper_union > 0 else 0

                interest_overlap = len(user_interests & other_interests)
                interest_union = len(user_interests | other_interests)
                interest_sim = interest_overlap / interest_union if interest_union > 0 else 0

                # Combined similarity
                similarity = 0.6 * paper_sim + 0.4 * interest_sim

                if similarity > 0.1:  # Threshold
                    similar_users.append({
                        'user_id': other_id,
                        'similarity_score': similarity,
                        'shared_papers': list(user_papers & other_papers),
                        'shared_interests': list(user_interests & other_interests)
                    })

            # Sort by similarity
            similar_users.sort(key=lambda x: x['similarity_score'], reverse=True)

        except Exception as e:
            logger.error(f"Error finding similar users: {e}")

        return similar_users[:n]

    def recommend_from_similar_users(self, user_id: str, n: int = 10) -> List[Dict]:
        """
        Recommend papers that similar users liked.

        Args:
            user_id: User identifier
            n: Number of recommendations

        Returns:
            [
                {
                    'paper_id': str,
                    'score': float,
                    'similar_users_who_liked': [user_ids],
                    'explanation': str
                },
                ...
            ]
        """
        recommendations = []

        try:
            # Find similar users
            similar_users = self.find_similar_users(user_id, n=20)

            if not similar_users or user_id not in self.user_profiles:
                return recommendations

            user_profile = self.user_profiles[user_id]
            user_read = user_profile['read_papers']

            # Collect papers liked by similar users
            candidate_scores = defaultdict(float)
            candidate_users = defaultdict(list)

            for similar_user in similar_users:
                other_id = similar_user['user_id']
                similarity = similar_user['similarity_score']
                other_profile = self.user_profiles.get(other_id)

                if not other_profile:
                    continue

                # Papers liked by this user
                for paper in other_profile['liked_papers']:
                    if paper not in user_read:
                        candidate_scores[paper] += similarity
                        candidate_users[paper].append(other_id)

            # Rank and format
            ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

            for paper_id, score in ranked[:n]:
                recommendations.append({
                    'paper_id': paper_id,
                    'score': score,
                    'similar_users_who_liked': candidate_users[paper_id],
                    'explanation': f"Liked by {len(candidate_users[paper_id])} similar users"
                })

        except Exception as e:
            logger.error(f"Error generating collaborative recommendations: {e}")

        return recommendations

    def identify_research_communities(self) -> List[Dict]:
        """
        Identify clusters of users with shared interests.

        Returns:
            [
                {
                    'community_id': int,
                    'users': [user_ids],
                    'common_interests': [topics],
                    'representative_papers': [papers],
                    'community_label': str
                },
                ...
            ]
        """
        communities = []

        try:
            if not self.user_profiles or len(self.user_profiles) < 2:
                return communities

            # Simple community detection: cluster users by shared interests
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                logger.warning("sklearn not available for community detection")
                return communities

            user_ids = list(self.user_profiles.keys())

            # Create feature vectors for users (interest embeddings)
            user_vectors = []
            for user_id in user_ids:
                profile = self.user_profiles[user_id]
                if profile.get('embedding') is not None:
                    user_vectors.append(profile['embedding'])
                else:
                    user_vectors.append(np.random.randn(128))

            # Cluster users
            n_clusters = min(5, len(user_ids) // 3)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(user_vectors)

                # Build communities
                for i in range(n_clusters):
                    community_users = [user_ids[j] for j in range(len(labels)) if labels[j] == i]

                    # Find common interests
                    all_interests = []
                    all_papers = set()

                    for user_id in community_users:
                        profile = self.user_profiles[user_id]
                        all_interests.extend(profile['interests'])
                        all_papers.update(profile['read_papers'])

                    # Count interest frequency
                    interest_counts = Counter(all_interests)
                    common_interests = [interest for interest, _ in interest_counts.most_common(5)]

                    communities.append({
                        'community_id': i,
                        'users': community_users,
                        'common_interests': common_interests,
                        'representative_papers': list(all_papers)[:10],
                        'community_label': f"Community {i}: {', '.join(common_interests[:2])}"
                    })

        except Exception as e:
            logger.error(f"Error identifying communities: {e}")

        return communities

    # ========== Utility Methods ==========

    def find_survey_papers(self, topic: str) -> List[str]:
        """
        Find survey papers on a topic.

        Args:
            topic: Topic keyword

        Returns:
            List of survey paper IDs
        """
        surveys = []
        try:
            # heuristic: title contains 'survey' or 'review'
            hits = self.graph.search_entities(topic, limit=200)
            for h in hits:
                name = h.get('name', '').lower()
                if 'survey' in name or 'review' in name:
                    surveys.append(h.get('name'))
        except Exception:
            logger.exception("Failed to find survey papers")
        return surveys

    # ========== Helper Methods ==========

    def _apply_diversity_penalty(
        self,
        candidates: Dict[str, float],
        profile: Dict,
        diversity_weight: float
    ) -> Dict[str, float]:
        """Apply diversity penalty to avoid filter bubble."""
        adjusted = candidates.copy()

        for paper_id in candidates:
            # Check if too similar to read papers
            similarity_to_read = 0.0

            if self.gnn_manager and hasattr(self.gnn_manager, 'embedder'):
                paper_emb = self.gnn_manager.embedder.get_embedding(paper_id)

                if paper_emb is not None:
                    for read_paper in list(profile['read_papers'])[:10]:
                        read_emb = self.gnn_manager.embedder.get_embedding(read_paper)
                        if read_emb is not None:
                            sim = np.dot(paper_emb, read_emb) / (
                                np.linalg.norm(paper_emb) * np.linalg.norm(read_emb) + 1e-8
                            )
                            similarity_to_read = max(similarity_to_read, sim)

            # Apply penalty
            penalty = diversity_weight * similarity_to_read
            adjusted[paper_id] = candidates[paper_id] * (1 - penalty)

        return adjusted

    def _classify_recommendation_type(
        self,
        paper_id: str,
        profile: Dict,
        score: float
    ) -> str:
        """Classify recommendation as similar, trending, or exploratory."""
        if score > 0.7:
            return 'similar'
        elif score > 0.4:
            return 'trending'
        else:
            return 'exploratory'

    def _generate_explanation(
        self,
        paper_id: str,
        profile: Dict,
        score: float,
        rec_type: str
    ) -> str:
        """Generate human-readable explanation for recommendation."""
        if rec_type == 'similar':
            return f"Highly relevant to your interests (score: {score:.2f})"
        elif rec_type == 'trending':
            return f"Related to your reading history (score: {score:.2f})"
        else:
            return f"Exploratory recommendation to broaden your research (score: {score:.2f})"


if __name__ == "__main__":
    print("Unified Recommendation Engine loaded successfully")
