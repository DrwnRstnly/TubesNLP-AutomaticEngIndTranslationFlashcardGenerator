from langgraph.graph import StateGraph, END

from src.pipeline.state import PipelineState
from src.task1_ranker.node import SSRNode
from src.task2_intent.node import IntentNode
# from src.task3_cefr.node import CEFRNode
# from src.task4_mt.node import TranslationNode


graph = StateGraph(PipelineState)

graph.add_node(
    "SSR",
    SSRNode(
        model_path="models/e5_large_lora",     
    )
)

graph.add_node(
    "Intent",
    IntentNode(
        model_path="models/distilroberta-base" 
    )
)

# graph.add_node(
#     "CEFR",
#     CEFRNode(
#         model_path="src/task3_cefr/models/cefr_classifier"
#     )
# )

# graph.add_node(
#     "Translation",
#     TranslationNode(
#         model_path="src/task4_mt/models/en_to_id"
#     )
# )

# === ENTRY POINT DAN EDGES ===
graph.set_entry_point("SSR")
graph.add_edge("SSR", END)
# graph.add_edge("SSR", "Intent")
# graph.add_edge("Intent", "CEFR")
# graph.add_edge("CEFR", "Translation")
# graph.add_edge("Translation", END)

# === COMPILE GRAPH ===
flashcard_graph = graph.compile()
