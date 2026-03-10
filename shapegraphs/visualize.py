# Vibecoded by Antigravity

import pickle
import matplotlib.pyplot as plt
import networkx as nx
import sys
import argparse
import os

def plot_shapegraph(G, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Pitch dimensions (approximate)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-35, 35)

    # Draw pitch outline
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color='black')
    ax.plot([0, 0], [-34, 34], color='black')
    center_circle = plt.Circle((0, 0), 9.15, color='black', fill=False)
    ax.add_patch(center_circle)

    pos = {}
    node_colors = []
    labels = {}

    home_color = 'red'
    away_color = 'blue'

    has_ball_node = None

    for node, data in G.nodes(data=True):
        x = data.get('x', 0)
        y = data.get('y', 0)
        pos[node] = (x, y)
        team = data.get('team', 'unknown')
        if team == 'home':
            node_colors.append(home_color)
        else:
            node_colors.append(away_color)
        
        shirt = data.get('shirt', '')
        role = data.get('inferred_role', '?')
        # labels[node] = f"{shirt}\n{role}"
        labels[node] = str(role)

        if data.get('has_ball', False):
            has_ball_node = node

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500, alpha=0.8)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_color='white', font_weight='bold')

    # Highlight ball
    if has_ball_node is not None:
        ball_x, ball_y = pos[has_ball_node]
        ax.plot(ball_x, ball_y, 'o', color='yellow', markersize=8, markeredgecolor='black', label='Ball Carrier')

    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
        
    ax.axis('off')

def main():
    parser = argparse.ArgumentParser(description="Visualize a Shapegraph from a pickle file")
    parser.add_argument("pkl_file", help="Path to shapegraphs.pkl")
    parser.add_argument("--frame", type=int, default=None, help="Specific frame number to plot")
    parser.add_argument("--output", type=str, default="shapegraph_viz.png", help="Output image file")
    
    args = parser.parse_args()

    if not os.path.exists(args.pkl_file):
        print(f"Error: {args.pkl_file} not found.")
        sys.exit(1)

    print(f"Loading {args.pkl_file}...")
    with open(args.pkl_file, 'rb') as f:
        data = pickle.load(f)

    if not data:
        print("No shapegraphs found in the file.")
        sys.exit(1)

    frames = sorted(list(data.keys()))
    print(f"Loaded {len(frames)} frames. Available range: {frames[0]} to {frames[-1]}")

    target_frame = args.frame if args.frame is not None else frames[0]
    
    if target_frame not in data:
        print(f"Frame {target_frame} not found in the data. Falling back to the first available frame '{frames[0]}'.")
        target_frame = frames[0]

    G = data[target_frame]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_shapegraph(G, ax=ax, title=f"Shapegraph Visualization - Frame {target_frame}")
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
