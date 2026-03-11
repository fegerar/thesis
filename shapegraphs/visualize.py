# Vibecoded by Antigravity

import math
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

    # Resolve overlapping nodes by spreading coincident positions in a small circle.
    # Group nodes that share the same (rounded) coordinate.
    _JITTER_RADIUS = 1.5  # metres
    from collections import defaultdict
    bucket: dict = defaultdict(list)
    for node, (x, y) in pos.items():
        key = (round(x, 3), round(y, 3))
        bucket[key].append(node)

    for key, nodes in bucket.items():
        if len(nodes) == 1:
            continue
        cx, cy = key
        for k, node in enumerate(nodes):
            angle = 2 * math.pi * k / len(nodes)
            pos[node] = (cx + _JITTER_RADIUS * math.cos(angle),
                         cy + _JITTER_RADIUS * math.sin(angle))

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
    parser.add_argument("--frame", type=int, default=None, help="Specific frame number to plot (ignored if --video is set)")
    parser.add_argument("--output", type=str, default="shapegraph_viz.png", help="Output image file (or video file if --video is set)")
    parser.add_argument("--video", action="store_true", help="Generate a video of all frames instead of a single image")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the video (default: 10)")
    
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

    if args.video:
        import cv2
        import numpy as np
        import io
        
        out_path = args.output
        if not out_path.endswith('.mp4'):
            if out_path == "shapegraph_viz.png":
                out_path = "shapegraph_viz.mp4"
            else:
                out_path = out_path.rsplit('.', 1)[0] + '.mp4'
                
        print(f"Generating video with {len(frames)} frames at {args.fps} FPS...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = None

        # Detect whether data contains paired original/nominal graphs
        sample = data[frames[0]]
        has_nominal = isinstance(sample, dict) and "nominal" in sample

        if has_nominal:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, frame_id in enumerate(frames):
            entry = data[frame_id]
            if has_nominal:
                ax1.clear()
                ax2.clear()
                G_orig = entry["original"]
                G_nom  = entry["nominal"]
                plot_shapegraph(G_orig, ax=ax1, title=f"Actual positions — Frame {frame_id}")
                plot_shapegraph(G_nom,  ax=ax2, title=f"Nominal positions — Frame {frame_id}")
            else:
                ax.clear()
                G_orig = entry if not isinstance(entry, dict) else entry.get("original", entry)
                plot_shapegraph(G_orig, ax=ax, title=f"Shapegraph Visualization - Frame {frame_id}")
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            if writer is None:
                height, width, _ = img.shape
                writer = cv2.VideoWriter(out_path, fourcc, args.fps, (width, height))
                
            writer.write(img)
            
            if i > 0 and i % 10 == 0:
                print(f"Processed {i}/{len(frames)} frames...")
                
        if writer is not None:
            writer.release()
        plt.close(fig)
        print(f"Saved video to {out_path}")

    else:
        target_frame = args.frame if args.frame is not None else frames[0]
        
        if target_frame not in data:
            print(f"Frame {target_frame} not found in the data. Falling back to the first available frame '{frames[0]}'.")
            target_frame = frames[0]

        entry = data[target_frame]
        has_nominal = isinstance(entry, dict) and "nominal" in entry

        if has_nominal:
            G_orig = entry["original"]
            G_nom  = entry["nominal"]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
            plot_shapegraph(G_orig, ax=ax1, title=f"Actual positions — Frame {target_frame}")
            plot_shapegraph(G_nom,  ax=ax2, title=f"Nominal positions — Frame {target_frame}")
        else:
            G_orig = entry if not isinstance(entry, dict) else entry.get("original", entry)
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_shapegraph(G_orig, ax=ax, title=f"Shapegraph Visualization - Frame {target_frame}")
        
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
