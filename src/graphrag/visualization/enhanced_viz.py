#!/usr/bin/env python3
"""
Enhanced Graph Visualization
Provides filtering, search, highlighting, and interactive features
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from pyvis.network import Network
import json

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

class EnhancedGraphVisualizer:
    """Enhanced graph visualization with filtering and interactivity"""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.colors = {
            'Person': '#FF6B6B',
            'Organization': '#4ECDC4',
            'Location': '#45B7D1',
            'Concept': '#FFA07A',
            'Date': '#98D8C8',
            'Product': '#DDA0DD',
            'Event': '#FFD700',
            'Work': '#90EE90',
            'Law': '#F0E68C',
            'Group': '#FFB6C1'
        }

    def close(self):
        """Close database connection"""
        self.driver.close()

    def create_filtered_visualization(
        self,
        output_file: str = "filtered_graph.html",
        entity_types: Optional[List[str]] = None,
        search_term: Optional[str] = None,
        limit: int = 100,
        highlight_entities: Optional[List[str]] = None
    ) -> Path:
        """Create visualization with filtering options"""

        with self.driver.session() as session:
            # Build query with filters
            where_clauses = []
            params = {"limit": limit}

            if entity_types:
                where_clauses.append("e.type IN $entity_types")
                params["entity_types"] = entity_types

            if search_term:
                where_clauses.append("toLower(e.text) CONTAINS toLower($search_term)")
                params["search_term"] = search_term

            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            # Get entities
            query = f"""
                MATCH (e:Entity)
                WHERE {where_clause}
                RETURN e.id as id, e.text as text, e.type as type
                LIMIT $limit
            """

            result = session.run(query, params)
            entities = [dict(record) for record in result]

            if not entities:
                print("✗ No entities found matching criteria")
                return None

            # Get relationships between filtered entities
            entity_ids = [e['id'] for e in entities]

            query = """
                MATCH (source:Entity)-[r:RELATED]->(target:Entity)
                WHERE source.id IN $entity_ids AND target.id IN $entity_ids
                RETURN source.id as source_id, source.text as source_text,
                       target.id as target_id, target.text as target_text,
                       r.type as type
                LIMIT $rel_limit
            """

            result = session.run(query, {"entity_ids": entity_ids, "rel_limit": limit * 2})
            relationships = [dict(record) for record in result]

        # Create visualization
        net = Network(
            height="800px",
            width="100%",
            bgcolor="#1a1a1a",
            font_color="white",
            notebook=False
        )

        # Enhanced physics for better layout
        net.barnes_hut(
            gravity=-8000,
            central_gravity=0.3,
            spring_length=150,
            spring_strength=0.001,
            damping=0.09
        )

        # Add nodes with enhanced styling
        for entity in entities:
            entity_type = entity.get('type', 'Concept')
            color = self.colors.get(entity_type, '#CCCCCC')
            entity_text = entity['text']

            # Highlight specified entities
            if highlight_entities and entity_text in highlight_entities:
                color = '#FF0000'  # Red for highlighted
                size = 30
                border_width = 4
                border_color = '#FFD700'  # Gold border
            else:
                size = 15
                border_width = 2
                border_color = color

            net.add_node(
                entity['id'],
                label=entity_text,
                title=f"<b>{entity_text}</b><br>Type: {entity_type}",
                color=color,
                size=size,
                borderWidth=border_width,
                borderWidthSelected=4,
                font={'size': 14, 'color': 'white', 'face': 'Arial'},
                shape='dot'
            )

        # Add edges
        for rel in relationships:
            src_id = rel['source_id']
            tgt_id = rel['target_id']
            rel_type = rel.get('type', 'related')

            net.add_edge(
                src_id,
                tgt_id,
                title=rel_type,
                color={'color': '#666666', 'highlight': '#00FF00'},
                width=2
            )

        # Add custom controls
        net.show_buttons(filter_=['physics', 'interaction', 'manipulation'])

        output_path = Path("output") / output_file
        output_path.parent.mkdir(exist_ok=True)

        # Save with enhanced HTML
        net.save_graph(str(output_path))

        # Add custom legend
        self._add_legend_to_html(output_path, entity_types, search_term)

        print(f"✓ Enhanced visualization saved: {output_path}")
        return output_path

    def create_entity_focused_view(
        self,
        entity_text: str,
        hops: int = 2,
        output_file: str = "entity_focused.html"
    ) -> Path:
        """Create visualization focused on a specific entity and its neighborhood"""

        with self.driver.session() as session:
            # Find the entity
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.text) = toLower($text)
                RETURN e.id as id, e.text as text, e.type as type
                LIMIT 1
            """, text=entity_text)

            central_entity = result.single()
            if not central_entity:
                print(f"✗ Entity '{entity_text}' not found")
                return None

            central_id = central_entity['id']

            # Get entities within N hops
            query = f"""
                MATCH path = (center:Entity {{id: $center_id}})-[*1..{hops}]-(neighbor:Entity)
                WITH DISTINCT neighbor
                RETURN neighbor.id as id, neighbor.text as text, neighbor.type as type
                LIMIT 200
            """

            result = session.run(query, center_id=central_id)
            neighbors = [dict(record) for record in result]

            # Include central entity
            all_entities = [dict(central_entity)] + neighbors
            entity_ids = [e['id'] for e in all_entities]
            ids_str = "', '".join(entity_ids)

            # Get relationships
            query = f"""
                MATCH (source:Entity)-[r:RELATED]->(target:Entity)
                WHERE source.id IN ['{ids_str}'] AND target.id IN ['{ids_str}']
                RETURN source.id as source_id, source.text as source_text,
                       target.id as target_id, target.text as target_text,
                       r.type as type
                LIMIT 400
            """

            result = session.run(query)
            relationships = [dict(record) for record in result]

        # Create visualization highlighting the central entity
        output_path = self.create_filtered_visualization(
            output_file=output_file,
            limit=len(all_entities),
            highlight_entities=[entity_text]
        )

        print(f"✓ Entity-focused view created for '{entity_text}'")
        print(f"  Showing {len(neighbors)} connected entities within {hops} hops")

        return output_path

    def create_type_filtered_view(
        self,
        entity_types: List[str],
        output_file: str = "type_filtered.html",
        limit: int = 200
    ) -> Path:
        """Create visualization showing only specific entity types"""

        return self.create_filtered_visualization(
            output_file=output_file,
            entity_types=entity_types,
            limit=limit
        )

    def create_search_view(
        self,
        search_term: str,
        output_file: str = "search_results.html",
        limit: int = 100
    ) -> Path:
        """Create visualization of search results"""

        return self.create_filtered_visualization(
            output_file=output_file,
            search_term=search_term,
            limit=limit,
            highlight_entities=None
        )

    def _add_legend_to_html(self, html_path: Path, filters: Optional[List[str]], search: Optional[str]):
        """Add legend and filter info to HTML file"""

        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Create legend HTML
        legend_html = '''
        <div style="position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; color: white; font-family: Arial; z-index: 1000; max-width: 250px;">
            <h3 style="margin-top: 0; font-size: 16px;">Legend</h3>
        '''

        # Add filter info
        if filters:
            legend_html += f'<p style="font-size: 12px;"><b>Filtered:</b> {", ".join(filters)}</p>'
        if search:
            legend_html += f'<p style="font-size: 12px;"><b>Search:</b> "{search}"</p>'

        # Add color legend
        legend_html += '<div style="font-size: 12px;">'
        for entity_type, color in self.colors.items():
            legend_html += f'<div style="margin: 5px 0;"><span style="display: inline-block; width: 12px; height: 12px; background: {color}; border-radius: 50%; margin-right: 8px;"></span>{entity_type}</div>'

        legend_html += '</div></div>'

        # Insert before closing body tag
        html_content = html_content.replace('</body>', f'{legend_html}</body>')

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def get_entity_types(self) -> List[Dict[str, int]]:
        """Get all entity types and their counts"""

        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.type as type, count(*) as count
                ORDER BY count DESC
            """)

            return [dict(record) for record in result]

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED GRAPH VISUALIZATION")
    print("=" * 80)

    viz = EnhancedGraphVisualizer()

    try:
        # Get available entity types
        print("\n📊 Available Entity Types:")
        print("-" * 80)
        entity_types = viz.get_entity_types()
        for et in entity_types:
            print(f"  {et['type']}: {et['count']} entities")

        # Example 1: Filter by type
        print("\n\n1️⃣ Creating visualization filtered by Person and Organization...")
        viz.create_type_filtered_view(
            entity_types=['Person', 'Organization'],
            output_file="people_and_orgs.html",
            limit=100
        )

        # Example 2: Search view
        print("\n2️⃣ Creating search visualization for entities containing 'AI'...")
        viz.create_search_view(
            search_term="AI",
            output_file="ai_entities.html"
        )

        # Example 3: Entity-focused view
        print("\n3️⃣ Creating entity-focused view...")
        # Try to find a well-connected entity
        with viz.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)-[r]-()
                WITH e, count(r) as connections
                ORDER BY connections DESC
                LIMIT 1
                RETURN e.text as text
            """)
            top_entity = result.single()

            if top_entity:
                entity_text = top_entity['text']
                print(f"  Focusing on '{entity_text}' (most connected entity)")
                viz.create_entity_focused_view(
                    entity_text=entity_text,
                    hops=2,
                    output_file="entity_focused.html"
                )

        print("\n✅ All visualizations created successfully!")
        print("\nGenerated files:")
        print("  - output/people_and_orgs.html")
        print("  - output/ai_entities.html")
        print("  - output/entity_focused.html")

    finally:
        viz.close()
