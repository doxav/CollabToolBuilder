import spacy
import networkx as nx
import matplotlib.pyplot as plt

from env.IR_CPS_TechSynthesis.env import SynthesisManager

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

def extract_entities_and_relations(text):
    """
    Extract entities and their relations using spaCy.

    Args:
    text (str): Text to extract entities and relations from.

    Returns:
    list: A list of tuples (entity1, relation, entity2).
    """
    doc = nlp(text)
    relations = []
    for ent in doc.ents:
        if ent.root.head.pos_ in ['VERB', 'ADP']:
            subject = ent.text
            object = ent.root.head.head.text
            relations.append((subject, ent.root.head.text, object))
    return relations

def visualize_key_concept_relation(bot:SynthesisManager, abstract, document_id):
    """
    Visualize the relations using a network graph and integrate with the bot class.

    Args:
    abstract: abstract of the paper
    bot: Instance of Bot class to interact with documents.
    document_id: Identifier of the document for logging.
    """
    
    G = nx.DiGraph()
    
    relations = extract_entities_and_relations(abstract) #  returns relations (list): List of relations (entity1, relation, entity2).

    for ent1, relation, ent2 in relations:
        G.add_node(ent1)
        G.add_node(ent2)
        G.add_edge(ent1, ent2, label=relation)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Key Concept Relationships")
    plt.axis('off')
    
    # Save the plot as an image and add it as a section in the document
    image_path = f"{document_id}_concept_relationships.png"
    plt.savefig(image_path)
    plt.show()

    section_id = bot.create_and_add_section_then_return_id("Concept Relationships", f"See image: {image_path}")
    bot.add_event("concept_relationships", {"document_id": document_id, "section_id": section_id})
