from neo4j import GraphDatabase

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))

try:
    with driver.session() as session:
        result = session.run("RETURN 1 AS result")
        print("✅ Connected! Result:", result.single()["result"])
except Exception as e:
    print("❌ Failed to connect:", e)
finally:
    driver.close()
