
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
from IPython.display import display_html
from networkx.algorithms import community
from collections import defaultdict
import seaborn as sns
import spacy
import torch
from transformers import Trainer
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)
from collections import Counter
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs   # <-- THIS FIXES IT
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  
    
def foo():
    return "Hello from foo"

def clean_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc 
              if not token.is_stop and token.is_alpha]
    return " ".join(tokens)
    
    




#helper function
def display_side_by_side(dfs, titles=None):
    html_str = "<div style='display:flex;flex-flow:row nowrap;column-gap:20px'>"
    for df, title in zip(dfs, titles):
        html_str += f"""
        <div style="margin:10px">
            <h4 style="text-align:center">{title}</h4>
            {df.to_html()}
        </div>"""
    html_str += "</div>"

    # return display_html(html_str, raw=True)
    return html_str

def top_n_degrees(degree_dict: dict, n: int) -> pd.DataFrame:
    df = pd.DataFrame(list(degree_dict.items()), columns=["Node", "Degree"])
    df["Degree_(%)"] = (df["Degree"] / df["Degree"].sum()) * 100


    df = df.sort_values(by="Degree", ascending=False).head(n).reset_index(drop=True)
    return df.reset_index(drop=True)



def get_island_graph_components(G, threshold=3, remove_unconnected=False):
    strong_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= threshold]
    island_graph = nx.Graph()
    island_graph.add_edges_from(strong_edges)

    if remove_unconnected:
        island_graph.remove_nodes_from(list(nx.isolates(island_graph)))

    components = [island_graph.subgraph(c).copy() for c in nx.connected_components(island_graph)]
    components.sort(key=lambda x: x.number_of_nodes(), reverse=True)

    return island_graph, components


def plot_island_graph(G,threshold, node_size=400):
    if G.number_of_edges() == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No edges to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    pos = nx.kamada_kawai_layout(G)
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    widths = [w / max(weights) * 4 for w in weights] if weights else 1

    fig, ax = plt.subplots(figsize=(15, 10))
    nx.draw(
        G,
        pos,
        node_color="skyblue",
        edge_color="gray",
        width=widths,
        node_size=node_size,
        with_labels=True,
        font_size=8,
        ax=ax
    )
    ax.axis("off")
    ax.set_title(f"Island Graph (edges weight ≥ {threshold})")
    


def compute_metrics(G, top_n=10, graph_label="Island Graph"):
    # Compute clustering and density metrics
    avg_clustering = nx.average_clustering(G)
    avg_weighted_clustering = nx.average_clustering(G, weight="weight")
    density = nx.density(G)

    # Summary dataframe
    clustering_df = pd.DataFrame(
        [
            ["Average Clustering", avg_clustering],
            ["Weighted Average Clustering", avg_weighted_clustering],
            ["Density", density]
        ],
        columns=["Metric Name", "Result"]
    )

    # Compute centrality measures
    deg_centrality = nx.degree_centrality(G)
    bet_centrality = nx.betweenness_centrality(G)
    eig_centrality = nx.eigenvector_centrality(G,max_iter=500)

    # Create top-n tables
    top_degree_df = top_n_degrees(deg_centrality, top_n)
    top_betweenness_df = top_n_degrees(bet_centrality, top_n)
    top_eigen_df = top_n_degrees(eig_centrality, top_n)

    # Display formatted tables
    metrics_view = display_side_by_side(
        [
            clustering_df.style.hide(axis="index"),
            top_degree_df,
            top_betweenness_df,
            top_eigen_df
        ],
        [f"{graph_label} Metrics", f"Degree Centrality (Top {str(top_n)} Nodes)", f"Betweenness Centrality (Top {str(top_n)} Nodes)", 
         f"Eigenvector Centrality (Top {str(top_n)} Nodes)"]
    )

    # Return all relevant data
    return {
        "graph": G,
        "clustering": clustering_df,
        "degree_centrality": top_degree_df,
        "betweenness_centrality": top_betweenness_df,
        "eigenvector_centrality": top_eigen_df,
        "metrics_view": metrics_view
        
    }

    


def analyze_connectivity(G, graph_label="Graph"):
    articulation_points = list(nx.articulation_points(G))
    bridges = list(nx.bridges(G))

    summary = pd.DataFrame({
        "Metric": ["Articulation Points", "Bridges"],
        "Count": [len(articulation_points), len(bridges)],
        "Details": [
            articulation_points if articulation_points else "None",
            bridges if bridges else "None"
        ]
    })

    summary.index = [f"{graph_label} - {m}" for m in summary["Metric"]]
    # summary.drop(columns=["Metric"], inplace=True)

    return summary.style.hide(axis="index")



def summarize_core_numbers(G, top_n=None):
    core_numbers = nx.core_number(G)
    core_groups = defaultdict(list)

    for node, core_val in core_numbers.items():
        core_groups[core_val].append(node)

    # Sort by core number descending (deepest core first)
    core_items = sorted(core_groups.items(), reverse=True)

    if top_n is not None:
        core_items = core_items[:top_n]

    df = pd.DataFrame([
        {"Core_Number": k, "Node_Count": len(v), "Nodes": sorted(v)}
        for k, v in core_items
    ])

    return df

def summarize_component_counts(componentList, top_n=None):

    nodes =  []
    for comp in componentList:
        nodes.append(
            {
                "Component Count": len(comp),
                "Details": list(comp.nodes())
            }
            
            )

    df = pd.DataFrame(nodes).sort_values(by="Component Count",ascending=False).head(top_n)

    return df.style.hide(axis="index")



def plot_island_evolution(graph, weight_attr='weight', title='Island evolution across thresholds'):
    weights = [d.get(weight_attr, 0) for _, _, d in graph.edges(data=True)]
    if not weights:
        raise ValueError("Graph has no edges with the specified weight attribute.")
    
    min_w, max_w = int(min(weights)), int(max(weights))
    thresholds = range(min_w, max_w + 1)
    edges_count, comps_count, largest_size = [], [], []

    for t in thresholds:
        sub_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get(weight_attr, 0) >= t]
        subG = graph.edge_subgraph(sub_edges).copy()
        edges_count.append(subG.number_of_edges())
        comps = list(nx.connected_components(subG))
        comps_count.append(len(comps))
        largest_size.append(max((len(c) for c in comps), default=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, edges_count, label="Edges remaining")
    plt.plot(thresholds, comps_count, label="Number of islands")
    plt.plot(thresholds, largest_size, label="Largest island size")
    plt.xlabel("Threshold (minimum edge weight)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.show()





def plot_island_density_row(results, keys, titles, column="Degree"):
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (key, title) in enumerate(zip(keys, titles)):
        df = results[key]
        sns.kdeplot(df[column].dropna(), fill=True, linewidth=1.5, ax=axes[i])
        axes[i].set_title(title, fontsize=13)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Density")
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



model_metrics = [
        "Set",
        "Accuracy",
        "Precision",
        "Recall",
        "Sensitivity",
        "Specificity",
        "F1"
        ]

def evaluate_model(y_true, y_pred,average):
   

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred,average=average)
    rec = recall_score(y_true, y_pred,average=average)
    f1 = f1_score(y_true, y_pred,average=average)

    cm = confusion_matrix(y_true, y_pred)
    TP, FN, FP, TN = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sensitivity = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1
    }
    
def generate_report(model_instance,trainX,trainY,testX,testY,average=""):
    y_train_pred = model_instance.predict(trainX)
    y_test_pred = model_instance.predict(testX)
    train_set_metrics = evaluate_model(trainY,y_train_pred,average)
    test_set_metrics = evaluate_model(testY,y_test_pred,average)
    train_set_metrics["Set"] = "Training"
    test_set_metrics["Set"] = "Test"
    model_metrics_df = pd.DataFrame(columns=model_metrics,data= [train_set_metrics,test_set_metrics])
    styled_report = model_metrics_df.style.hide(axis="index")
    return model_metrics_df,styled_report



def build_graph(cat,clean_pair_freq_by_cat, max_nodes=None):
    G = nx.Graph()

    for (w1, w2), freq in clean_pair_freq_by_cat[cat].items():
        G.add_edge(w1, w2, weight=freq)

    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        strength = Counter()
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1)
            strength[u] += w
            strength[v] += w

        keep = set(node for node, _ in strength.most_common(max_nodes))
        G = G.subgraph(keep).copy()

    return G


def draw_graph(graph,title, seed=55, figsize=(15, 9)):
    pos = nx.spring_layout(graph, seed=seed)

    fig = Figure(figsize=figsize)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    nx.draw(
        graph,
        pos,
        ax=ax,
        with_labels=False,
        node_size=500,
        font_size=10,
        edge_color="gray",
        alpha=0.8
    )

    ax.set_title(title, fontsize=13)

    return fig, ax



def compute_cliques(graph):

    w_cliques = list(nx.find_cliques(graph))
    clique_data = [{ "Size": len(c),"Members": ", ".join(sorted(c))} for c in w_cliques]
    clique_df = pd.DataFrame(clique_data).sort_values(by="Size", ascending=False)

    styled_df = clique_df.head().style.hide(axis="index")
    return clique_df,styled_df