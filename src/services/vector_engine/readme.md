command to the test codes in examples:

1.process_pipeline (End-to-end test of document URL → vector index)
 cmd: 
-C:\Users\riyai\hackrx>set PYTHONPATH=src\services
-python -m vector_engine.examples.process_pipeline

2.test_vectorizer (	Just test vectorization logic with dummy data)
 cmd: 
-C:\Users\riyai\hackrx>set PYTHONPATH=src\services
-python -m vector_engine.examples.test_vectorizer

3.semantic_search (	Run semantic queries against indexed content)
cmd: 
-C:\Users\riyai\hackrx>set PYTHONPATH=src\services
-python -m vector_engine.examples.semantic_search