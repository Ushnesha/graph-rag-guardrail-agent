import os
from neo4j import GraphDatabase
from langchain_ollama import ChatOllama
from hybrid_search_engine import HybridSearchEngine, CORPUS

class GraphRAGPipeline:
    def __init__(self):
        self.search_engine = HybridSearchEngine(CORPUS)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,
            base_url=self.ollama_base_url
        )
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        if self.search_engine.recreate_needed == True:
            self._initialize_knowledge_graph()

    def _initialize_knowledge_graph(self):
        # Programmatically write core entity linkages to Neo4j
        with self.neo4j_driver.session() as session:
            # Clean database first
            session.run("MATCH (n) DETACH DELETE n")
            
            # Seed Entities and Relationships
            session.run("CREATE (sarah:Entity {name: 'sarah', type: 'PERSON'})")
            session.run("CREATE (alpha:Entity {name: 'project alpha', type: 'PROJECT'})")
            session.run("CREATE (infra:Entity {name: 'cloud infrastructure', type: 'DEPARTMENT'})")
            
            session.run("""
                MATCH (s:Entity {name: 'sarah'}), (a:Entity {name: 'project alpha'})
                CREATE (s)-[:LEADS]->(a)
            """)
            session.run("""
                MATCH (s:Entity {name: 'sarah'}), (i:Entity {name: 'cloud infrastructure'})
                CREATE (s)-[:MANAGES]->(i)
            """)

    def query_graph_relationships(self, keywords: list) -> list:
        # Find 1-hop connections for any detected keywords
        # used to fetch structural context to augment the LLM's prompt
        relations = []
        with self.neo4j_driver.session() as session:
            query = """
            MATCH (n:Entity)-[r]->(m:Entity)
            WHERE n.name IN $keywords OR m.name IN $keywords
            RETURN n.name AS source, type(r) AS rel, m.name AS target
            LIMIT 5
            """
            result = session.run(query, keywords=keywords)
            for record in result:
                relations.append(f"({record['source']})-[{record['rel']}]->({record['target']})")
        return relations

    def run_pipeline(self, query:str, model_name: str = "llama3.2:3b") -> str:

        text_context = self.search_engine.search(query)

        # Extract keywords from the query for graph lookup
        words = [word.strip().lower() for word in query.replace("?", "").split(" ")]
        graph_context = self.query_graph_relationships(words)

        context_str = "Text Context:\n" + "\n".join(text_context) + "\n\nGraph Context:\n" + "\n".join(graph_context)
        prompt = f"Using ONLY the context below, answer the query.\n\nContext:\n{context_str}\n\nQuery: {query}"
        
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=self.ollama_base_url
        )
        response = llm.invoke(prompt).content
        return response, context_str


if __name__ == "__main__":
    try:
        from time import time
        start_time = time()
        pipeline = GraphRAGPipeline()
        pipeline.neo4j_driver.verify_connectivity()
        print("Successfully connected to Neo4j database!")
        query = "What department does Sarah manage?"
        answer, formatted_context = pipeline.run_pipeline(query)
        end_time = time()
        print(f"Time taken: {(end_time - start_time) * 1000} ms")
        print(f"\n--- FORMATTED FINAL CONTEXT FOR QUERY : {query} ---")
        print(formatted_context)
        print("\n--- FINAL ANSWER ---")
        print(answer)

        pipeline.search_engine.close()
        pipeline.neo4j_driver.close()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")