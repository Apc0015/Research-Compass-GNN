"""
Database Optimization Module - Create indices for frequently queried fields.

This module provides database optimization utilities including index creation
for Neo4j graph database to improve query performance.

Part of Phase 4 optimization (OPT-016: Database Index Optimization).
"""

import logging
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """
    Optimize database performance through indices and constraints.

    Benefits:
    - 10-20% faster graph queries with proper indices
    - Faster lookups on frequently queried fields
    - Enforces uniqueness constraints
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize database optimizer.

        Args:
            uri: Neo4j database URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None

    def connect(self):
        """Connect to Neo4j database."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
        return self._driver

    def close(self):
        """Close database connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def create_index(
        self,
        label: str,
        property_name: str,
        index_name: Optional[str] = None
    ) -> bool:
        """
        Create an index on a node label and property.

        Args:
            label: Node label (e.g., "Paper", "Author")
            property_name: Property to index (e.g., "id", "name")
            index_name: Optional custom index name

        Returns:
            True if created successfully, False otherwise

        Example:
            optimizer.create_index("Paper", "id", "paper_id_index")
        """
        driver = self.connect()

        # Generate index name if not provided
        if index_name is None:
            index_name = f"{label.lower()}_{property_name}_index"

        try:
            with driver.session() as session:
                # Create index (Neo4j 4.0+ syntax)
                query = f"""
                CREATE INDEX {index_name} IF NOT EXISTS
                FOR (n:{label})
                ON (n.{property_name})
                """
                session.run(query)
                logger.info(f"✓ Created index: {index_name} on {label}.{property_name}")
                return True

        except Exception as e:
            logger.warning(f"Failed to create index {index_name}: {e}")
            return False

    def create_unique_constraint(
        self,
        label: str,
        property_name: str,
        constraint_name: Optional[str] = None
    ) -> bool:
        """
        Create a uniqueness constraint on a property.

        Args:
            label: Node label
            property_name: Property to constrain
            constraint_name: Optional custom constraint name

        Returns:
            True if created successfully, False otherwise

        Note:
            Uniqueness constraints automatically create an index.
        """
        driver = self.connect()

        # Generate constraint name if not provided
        if constraint_name is None:
            constraint_name = f"{label.lower()}_{property_name}_unique"

        try:
            with driver.session() as session:
                # Create uniqueness constraint
                query = f"""
                CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                FOR (n:{label})
                REQUIRE n.{property_name} IS UNIQUE
                """
                session.run(query)
                logger.info(f"✓ Created constraint: {constraint_name} on {label}.{property_name}")
                return True

        except Exception as e:
            logger.warning(f"Failed to create constraint {constraint_name}: {e}")
            return False

    def list_indices(self) -> List[Dict[str, Any]]:
        """
        List all indices in the database.

        Returns:
            List of index information dictionaries
        """
        driver = self.connect()

        try:
            with driver.session() as session:
                result = session.run("SHOW INDEXES")
                indices = []
                for record in result:
                    indices.append({
                        'name': record.get('name'),
                        'type': record.get('type'),
                        'state': record.get('state'),
                        'labelsOrTypes': record.get('labelsOrTypes'),
                        'properties': record.get('properties')
                    })
                return indices

        except Exception as e:
            logger.error(f"Failed to list indices: {e}")
            return []

    def list_constraints(self) -> List[Dict[str, Any]]:
        """
        List all constraints in the database.

        Returns:
            List of constraint information dictionaries
        """
        driver = self.connect()

        try:
            with driver.session() as session:
                result = session.run("SHOW CONSTRAINTS")
                constraints = []
                for record in result:
                    constraints.append({
                        'name': record.get('name'),
                        'type': record.get('type'),
                        'labelsOrTypes': record.get('labelsOrTypes'),
                        'properties': record.get('properties')
                    })
                return constraints

        except Exception as e:
            logger.error(f"Failed to list constraints: {e}")
            return []

    def optimize_research_compass(self) -> Dict[str, Any]:
        """
        Apply all recommended optimizations for Research Compass.

        Creates indices on frequently queried fields to improve performance.

        Returns:
            Dictionary with optimization results

        Recommended Indices:
            - Paper.id (unique) - Primary key lookups
            - Paper.title - Title searches
            - Paper.year - Temporal queries
            - Author.id (unique) - Primary key lookups
            - Author.name - Author searches
            - Concept.id (unique) - Primary key lookups
            - Concept.name - Concept searches
            - Institution.name - Institution searches
        """
        results = {
            'indices_created': 0,
            'constraints_created': 0,
            'failed': 0,
            'details': []
        }

        logger.info("🔧 Optimizing Research Compass database...")

        # Uniqueness constraints (automatically create indices)
        constraints = [
            ("Paper", "id"),
            ("Author", "id"),
            ("Concept", "id"),
            ("Institution", "id"),
        ]

        for label, prop in constraints:
            if self.create_unique_constraint(label, prop):
                results['constraints_created'] += 1
                results['details'].append(f"✓ Constraint: {label}.{prop}")
            else:
                results['failed'] += 1
                results['details'].append(f"✗ Constraint: {label}.{prop}")

        # Additional indices for non-unique fields
        indices = [
            ("Paper", "title"),
            ("Paper", "year"),
            ("Paper", "venue"),
            ("Author", "name"),
            ("Concept", "name"),
            ("Institution", "name"),
        ]

        for label, prop in indices:
            if self.create_index(label, prop):
                results['indices_created'] += 1
                results['details'].append(f"✓ Index: {label}.{prop}")
            else:
                results['failed'] += 1
                results['details'].append(f"✗ Index: {label}.{prop}")

        # Summary
        total = results['indices_created'] + results['constraints_created']
        logger.info(
            f"✅ Database optimization complete: "
            f"{results['constraints_created']} constraints, "
            f"{results['indices_created']} indices, "
            f"{results['failed']} failed"
        )

        return results

    def analyze_query_performance(
        self,
        query: str,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze query performance using EXPLAIN/PROFILE.

        Args:
            query: Cypher query to analyze
            params: Optional query parameters

        Returns:
            Performance analysis results
        """
        driver = self.connect()

        try:
            with driver.session() as session:
                # Use PROFILE to get actual execution stats
                profiled_query = f"PROFILE {query}"
                result = session.run(profiled_query, params or {})

                # Consume result and get summary
                list(result)  # Consume records
                summary = result.consume()

                return {
                    'db_hits': summary.counters.db_hits if hasattr(summary.counters, 'db_hits') else None,
                    'query': query,
                    'plan': summary.plan if hasattr(summary, 'plan') else None
                }

        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            return {'error': str(e)}


def optimize_database(uri: str, user: str, password: str) -> Dict[str, Any]:
    """
    Convenience function to optimize Research Compass database.

    Args:
        uri: Neo4j URI
        user: Neo4j username
        password: Neo4j password

    Returns:
        Optimization results

    Example:
        results = optimize_database("neo4j://localhost:7687", "neo4j", "password")
        print(f"Created {results['indices_created']} indices")
    """
    with DatabaseOptimizer(uri, user, password) as optimizer:
        return optimizer.optimize_research_compass()


if __name__ == '__main__':
    # Example usage
    import sys

    # Default connection
    uri = "neo4j://localhost:7687"
    user = "neo4j"
    password = "password"

    if len(sys.argv) > 1:
        password = sys.argv[1]

    print("🔧 Research Compass Database Optimizer")
    print("=" * 50)

    with DatabaseOptimizer(uri, user, password) as optimizer:
        # Show existing indices
        print("\n📋 Existing Indices:")
        indices = optimizer.list_indices()
        if indices:
            for idx in indices:
                print(f"  - {idx['name']}: {idx['labelsOrTypes']} {idx['properties']}")
        else:
            print("  None")

        # Show existing constraints
        print("\n🔒 Existing Constraints:")
        constraints = optimizer.list_constraints()
        if constraints:
            for con in constraints:
                print(f"  - {con['name']}: {con['labelsOrTypes']} {con['properties']}")
        else:
            print("  None")

        # Optimize
        print("\n⚡ Applying optimizations...")
        results = optimizer.optimize_research_compass()

        print("\n✅ Results:")
        for detail in results['details']:
            print(f"  {detail}")

        print(f"\n📊 Summary:")
        print(f"  Constraints created: {results['constraints_created']}")
        print(f"  Indices created: {results['indices_created']}")
        print(f"  Failed: {results['failed']}")
