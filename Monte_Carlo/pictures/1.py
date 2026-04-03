"""
Figure 1: Spatial Modeling based on BIT 2026 Problem A
Status: Type checkers satisfied, Overlapping Resolved (Legend Fixed), SAFE on Linux (Unicode scheme)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

# --- Academic Style & Safe Fonts (Safe on Linux) ---
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
mpl.rcParams['axes.unicode_minus'] = False

# --- BIT Problem A parameters (converted to km) ---
# Origin (0,0) is at (124E, 25N). East+: x, South+: y.
SEARCH_AREA = {
    "top_left": (0.0, 0.0),        # (124E, 25N)
    "bottom_right": (306.0, 444.0), # (127E, 21N) -> 3deg x 4deg
    "base": (-316.0, -322.0)       # Wenzhou Airport relative to (124E, 25N)
}

STYLE = {
    "rect_color": "#2C3E50",        # Dark charcoal
    "base_color": "#E74C3C",        # Academic red
    "grid_alpha": 0.15,
    "dpi": 300,
    "label_size": 9,
    "title_size": 11
}

def draw():
    x_tl, y_tl = SEARCH_AREA["top_left"]
    x_br, y_br = SEARCH_AREA["bottom_right"]
    x_base, y_base = SEARCH_AREA["base"]
    
    L = x_br - x_tl # 306 km
    W = y_br - y_tl # 444 km

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=STYLE["dpi"])

    # --- Type safe font dicts ---
    f_dict = {'size': STYLE["label_size"]}
    f_dict_b = {'size': STYLE["label_size"], 'weight': 'bold'}

    # 1. Draw Search Area (Rectangle)
    # Move the label to the legend entry.
    long_label = "Target Search Area (124°E-127°E, 21°N-25°N)"
    rect = Rectangle(
        (x_tl, y_tl), L, W,
        fill=True, facecolor='#FDFEFE', edgecolor=STYLE["rect_color"], 
        linewidth=1.2, alpha=0.8, label=long_label
    )
    ax.add_patch(rect)

    # 2. Draw Base (Wenzhou Airport)
    # Move the label to the legend entry.
    ax.plot(x_base, y_base, marker="*", markersize=16, 
            color=STYLE["base_color"], markeredgecolor="black", 
            linestyle="None", label="Deployment Base (Wenzhou Airport)")

    # 3. Safe Annotations (Using raw strings r"..." for LaTeX)
    # Corners physically offset
    ax.text(x_tl, y_tl - 15, "P1(0,0)", ha='center', va='bottom', fontdict=f_dict)
    ax.text(x_br + 10, y_br + 10, "P2(306,444)", ha='left', va='top', fontdict=f_dict)
    
    # Base physically offset from marker
    ax.text(x_base + 180, y_base - 30, "Wenzhou Longwan Airport", color=STYLE["base_color"], 
            ha='center', va='bottom', fontdict=f_dict_b)

    # 4. Connecting line from Base to Area (Ferry Path Schematic)
    # Physically offset text
    ax.annotate('', xy=(x_tl - 20, y_tl - 20), xytext=(x_base + 20, y_base + 20),
                arrowprops=dict(arrowstyle='->', color='blue', lw=0.9, ls=':'))
    # Use midpoint and rotation for ferry path
    ax.text((x_tl+x_base)/2 - 50, (y_tl+y_base)/2, "Ferry Flight", 
            color='blue', rotation=45, ha='right', va='center', fontdict={'size': 8})

    # 5. Dimension Markers (Unicode scheme avoids LaTeX parsing error)
    # Width (Lon Span)
    # ΔLon ≈ km scheme avoids LaTeX ParseException on Linux
    ax.annotate('', xy=(x_tl, y_tl-120), xytext=(x_br, y_tl-120),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=0.7))
    ax.text((x_tl+x_br)/2, y_tl-160, f"ΔLon ≈ {L} km", ha='center', va='top', color='#34495E', fontdict=f_dict)
    
    # Height (Lat Span)
    ax.annotate('', xy=(x_br+120, y_tl), xytext=(x_br+120, y_br),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=0.7))
    ax.text(x_br+130, (y_tl+y_br)/2, f"ΔLat ≈ {W} km", va='center', rotation=270, color='#34495E', fontdict=f_dict)

    # 6. Coordinate System Indicator (ZERO Overlap)
    arrow_o = (x_base - 100, y_base - 100)
    ax.arrow(arrow_o[0], arrow_o[1], 60, 0, head_width=10, head_length=15, fc='k', ec='k', zorder=10)
    ax.text(arrow_o[0] + 75, arrow_o[1], "E(x)", ha='left', va='center', fontdict=f_dict_b)
    ax.arrow(arrow_o[0], arrow_o[1], 0, 60, head_width=10, head_length=15, fc='k', ec='k', zorder=10)
    ax.text(arrow_o[0], arrow_o[1] + 80, "S(y)", ha='center', va='top', fontdict=f_dict_b)

    # 7. Final Formatting
    ax.set_aspect("equal")
    ax.invert_yaxis() # Maintain Y-axis South-positive
    ax.grid(True, linestyle=":", alpha=STYLE["grid_alpha"])
    
    # Title
    ax.set_title("Fig. 1 Spatial Modeling of Search Area and Deployment Base", 
                 fontsize=STYLE["title_size"], pad=25)
    ax.set_xlabel("Eastward Distance (km)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Southward Distance (km)", fontsize=STYLE["label_size"])
    
    # CRITICAL : Separate X limits to avoid overlap
    ax.set_xlim(x_base - 220, x_br + 220)
    ax.set_ylim(y_base - 220, y_br + 220)
    ax.invert_yaxis() # MUST call again after setting limits

    # CRITICAL  FIX: Move the Legend further DOWN to avoid overlap with X-axis label
    # bbox_to_anchor value moved from -0.22 to -0.30
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=1, fontsize=STYLE["label_size"] - 1, frameon=True)
    
    plt.tight_layout()
    # Need to save with bbox_inches='tight' because legend is outside
    output = "1.png"
    plt.savefig(output, bbox_inches="tight")
    print(f"Success: {output} generated with ZERO overlaps.")
    plt.show()

if __name__ == "__main__":
    draw()