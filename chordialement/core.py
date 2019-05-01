import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib import colors

import warnings
import seaborn as sns
import math
import numpy as np
import pandas as pd


def get_idx_interv(d, D):
    k = 0
    while(d > D[k]):
        k += 1
    return k-1


def dist(A, B):
    return np.linalg.norm(np.array(A) - np.array(B))


def deCasteljau(b, t):
    N = len(b)
    a = np.copy(b)
    for r in range(1, N):
        a[:N-r, :] = (1-t)*a[:N-r, :] + t*a[1:N-r+1, :]
    return a[0, :]


def BezierCv(b, nr=5):
    t = np.linspace(0, 1, nr)
    return np.array([[deCasteljau(b, t[k]),
                      deCasteljau(b, t[k+1])] for k in range(nr-1)])


def position_circle(x, radius=1):
    """ Return the x,y coordinate of the point at
        angle (360*x)°, in the circle of radius "radius"
        and center (0, 0)
    """
    return np.array([radius*math.cos(x*2*math.pi),
                     radius*math.sin(x*2*math.pi)])


def linear_gradient(start, end, n=10):
    """ Return a gradient between start and end,
        with n elements.
    """
    gradients = np.zeros((len(start), n))
    for i in range(len(start)):
        gradients[i, :] = np.linspace(start[i], end[i], num=n)
    return np.transpose(gradients)


def linear_gradient_color(c1, c2, n=10):
    """ Return a gradient between the two color c1 & c2
    """
    return linear_gradient(colors.to_rgba(c1), colors.to_rgba(c2), n=n)


def draw_chord(A, B, ax=None, color_start="b", color_end="r",
               bezier_prms=([0, 0.765, 1.414, 1.848, 2], [1.2, 1.5, 1.8, 2.]),
               **kwargs):
    """ Draw a Bezier curve between two points """
    d = dist(A, B)

    # Depending on the distance between the points
    # the Bezier curve parameters change
    K = get_idx_interv(d, bezier_prms[0])
    b = [A, A/bezier_prms[1][K], B/bezier_prms[1][K], B]
    bz = BezierCv(b, nr=100)

    lc = mcoll.LineCollection(bz,
                              colors=linear_gradient_color(color_start,
                                                           color_end,
                                                           n=100), **kwargs)
    ax.add_collection(lc)


def draw_arc_circle(start, end, color="b", ax=None,
                    thickness=0.1, precision=0.01):
    ts = np.arange(start, end, precision)
    poly_nodes = ([position_circle(t) for t in ts] +
                  [position_circle(t, radius=1+thickness) for t in ts[::-1]])
    x, y = zip(*poly_nodes)
    ax.fill(x, y, color=color)


def add_text_circle(x, txt, radius=1, ax=None, **kwargs):
    ax.text(*position_circle(x, radius=radius),
            txt, rotation=360*x - 90, ha='center', va='center', **kwargs)


def chord_diagram(source, target, hue="black", data=None, ax=None,
           palette=sns.color_palette(), circle_args={},
           text_args={}, chord_args={}):
    """ Draw a chord diagram
    @ Arguments:
    - source: A unique index for each of the chords. The order
    of these index will determine the order of the chord on the
    circle (starting at (1,0)). If data is not None, can be a column
    name, else should be an array.
    - target: The index to which the source are attached. If data is
    not None can be a column name, else should be an array.
    - hue: Can be either a color name, a column name (if data is not
    None), or an array. In the two last case, the color are given 
    by palette in order.
    - data: None or a dataframe containing columns source and target
    - ax: matplotlib.ax object in which to draw. If None, a figure 
    is created.
    - palette: the color palette used, if categorical data (default,
    seaborn default)
    - circle_args: argument of the ax.fill matplotlib function that
    draws the border of the circle.
    - text_args: argument of the ax.text matplotlib function that 
    draws the text.
    - chords_args: argument of the LineCollection function that
    draws the chords of the diagram.
    """
    if data is not None:
        source = data[source]
        target = data[target]
        if hue in data.keys():
            hue = data[hue]

    # remove duplicate, conserve order
    order = list(dict.fromkeys(hue))
            
    df = pd.DataFrame({"idx": source, "target": target, "categorie": hue})
    df.sort_values(by="idx", inplace=True)
    df["position"] = np.linspace(0, 1, len(df))
    df["cat_nb"] = df.categorie.map(dict(zip(order, range(len(order)))))
    df["tgt_position"] = df.target.map(df.set_index("idx").position.to_dict())
    df["tgt_cat_nb"] = df.target.map(df.set_index("idx").cat_nb.to_dict())
    
    if(len(palette) < len(order)):
        warnings.warn("Not enough colors in the palette, switching "
                      "to Seaborn husl palette.")
        palette = sns.color_palette("husl", len(order))

    if(ax is None):
        fig, ax = plt.subplots()

    start_categorie = df.groupby("cat_nb").position.min().to_dict()
    end_categorie = df.groupby("cat_nb").position.max().to_dict()
    for c, cname in zip(df.cat_nb.unique(), df.categorie.unique()):
        draw_arc_circle(start_categorie[c], end_categorie[c],
                        color=palette[c], ax=ax, **circle_args)
        add_text_circle((start_categorie[c] + end_categorie[c])/2, cname,
                        radius=1.2, ax=ax, color=palette[c], **text_args)

    for index, row in df.iterrows():
        if row["position"] != row["tgt_position"]:
            draw_chord(position_circle(row["position"]),
                       position_circle(row["tgt_position"]), ax=ax,
                       color_start=palette[int(row["cat_nb"])],
                       color_end=palette[int(row["tgt_cat_nb"])],
                       **chord_args)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('equal')
    ax.axis('off')
