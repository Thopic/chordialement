import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib import colors

import warnings
import seaborn as sns
import math
import numpy as np
import pandas as pd


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
               precision=1000, **kwargs):
    """ Draw a Bezier curve between two points """
    d = np.linalg.norm(np.array(A) - np.array(B))

    # Depending on the distance between the points
    # the Bezier curve parameters change
    b = [A, A/(1 + d), B/(1 + d), B]
    bz = BezierCv(b, nr=precision)

    lc = mcoll.LineCollection(bz,
                              colors=linear_gradient_color(color_start,
                                                           color_end,
                                                           n=precision), **kwargs)
    ax.add_collection(lc)


def draw_arc_circle(start, end, color="b", radius=1, ax=None,
                    thickness=0.1, precision=1000):
    ts = np.linspace(start, end, precision)
    poly_nodes = ([position_circle(t, radius=radius) for t in ts] +
                  [position_circle(t, radius=radius+thickness)
                   for t in ts[::-1]])
    x, y = zip(*poly_nodes)
    ax.fill(x, y, color=color)


def add_text_circle(x, txt, radius=1, ax=None, **kwargs):
    ax.text(*position_circle(x, radius=radius),
            txt, rotation=360*x, ha='center', va='center', **kwargs)


def order_data(nodes, links, order=None):
    """
    Return a correctly ordered dataframe, ready to be plotted in chord format
    @ Args:
    - nodes: a dictionary that associates to each unique nodes a category name
    - links: link nodes together
    - order: order of the categories
    """
    categories = list(set(nodes.values()))
    ndsnb = dict(zip(nodes.keys(), range(len(nodes))))
    
    # decide on the order
    if(order is None):
        order = sorted(categories)
    df = pd.Series(nodes).to_frame("categorie")
    df["nbcat"] = df.categorie.map(dict(zip(order, range(len(order)))))

    df = df.rename_axis('source').reset_index()
    df["nbsource"] = df.source.map(ndsnb)
    
    dict_links = {}
    for l1, l2 in links:
        dict_links[l1] = l2
        dict_links[l2] = l1
        
    df["target"] = df.source.map(dict_links)
    df["nbtarget"] = df.target.map(ndsnb)
    df["tgt_nbcat"] = df.target.map(df.set_index("source").nbcat)

    df["tgt_cat_order"] = df.apply(
        lambda r: (len(order)-1+r["nbcat"]-r["tgt_nbcat"]) % (len(order)-1),
        axis=1)
    df["sort_order"] = df.apply(
        lambda r: (r["nbsource"] if r["nbcat"] <= r["tgt_nbcat"]
                   else -r["nbtarget"]),
        axis=1)
    df = df.sort_values(by=["nbcat", "tgt_cat_order", "sort_order"])
    return df
   
    
def chord_diagram(source, target, hue="black", data=None, ax=None,
                  hue_order=None, palette=sns.color_palette(),
                  categorie_internal_chords=False, sub_circle=False,
                  spacing=0, spacing_sub=0, inverted=False,
                  precision_chord=100, precision_circle=100,
                  thickness_circle=0.1, thickness_sub_circle=0.05,
                  radius_text=1.1, no_chords=False, circle_args={},
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
    - hue_order: The order in which the hue should be drawn
    - data: None or a dataframe containing columns source and target
    - ax: matplotlib.ax object in which to draw. If None, a figure 
    is created.
    - palette: the color palette used, if categorical data (default,
    seaborn default)
    - precision_chord: roughly the number of dot used to draw the chord
    bigger = slower but prettier
    - precision_circle: roughly the number of dot used to draw the circle
    bigger = slower but prettier
    - thickness_circle: thickness of the circle at the boundary
    - no_chords: if true don't draw the chords
    - radius_text: distance between the text and the center of the circle
    - categorie_internal_chords: if False does not draw a chord that
    start and end in the same categorie.
    - circle_args: argument of the ax.fill matplotlib function that
    draws the border of the circle.
    - text_args: argument of the ax.text matplotlib function that 
    draws the text.
    - chords_args: argument of the LineCollection function that
    draws the chords of the diagram.
    """
    if data is not None:
        source = data[source].values
        target = data[target].values
        if hue in data.keys():
            hue = data[hue].values

    if hue_order is None:
        hue_order = list(dict.fromkeys(hue))    
    nodes = dict(zip(source, hue))

    links = list(zip(source, target))
    df = order_data(nodes, links, hue_order)
    
    idxs = list(np.where(df.nbcat.values[:-1] != df.nbcat.values[1:])[0])
    x = 0
    positions = []
    for i in range(len(df)):
        positions.append(x)
        if i in idxs:
            x += spacing
        x += (1 - spacing*(len(idxs)+1))/(len(df))
    df["position"] = positions
    df["tgt_position"] = df.target.map(df.set_index("source").position)
        
    if(len(palette) < len(hue_order)):
        warnings.warn("Not enough colors in the palette ({} needed), switching "
                      "to Seaborn husl palette.".format(len(hue_order)))
        palette = sns.color_palette("husl", len(order))

    if(ax is None):
        fig, ax = plt.subplots()

    nb_to_name_cat = dict(enumerate(hue_order))
    positions = df.position.values
    tgt_cat = df.nbcat.values
    idxs = np.where(tgt_cat[:-1] != tgt_cat[1:])[0]
    start_categorie = [0] + list(positions[idxs+1])
    end_categorie = list(positions[idxs]) + [positions[-1]]
    cats = [tgt_cat[0]] + list(tgt_cat[idxs+1])
    
    for s, e, c in zip(start_categorie, end_categorie, cats):
        draw_arc_circle(s - 0.5/len(df), e + 0.5/len(df), color=palette[c], ax=ax,
                        precision=precision_circle, thickness=thickness_circle,
                        radius=(1+thickness_sub_circle + spacing_sub*2*math.pi if inverted else 1),**circle_args)
        add_text_circle((s + e - 1/len(df))/2, nb_to_name_cat[c], ax=ax, color=palette[c], radius=radius_text, **text_args)

    if sub_circle:
        df["both_cat"] = df.apply(lambda r: str(r["nbcat"]) + "_" + str(r["tgt_nbcat"]), axis=1)
        positions = df.position.values
        tgt_cat = df.both_cat.values
        idxs = np.where(tgt_cat[:-1] != tgt_cat[1:])[0]
        start_categorie = [0] + list(positions[idxs+1])
        end_categorie = list(positions[idxs]) + [positions[-1]]
        subcat = [df.tgt_nbcat.values[0]] + list(df.tgt_nbcat.values[idxs+1])
        for s, e, c in zip(start_categorie, end_categorie, subcat):
            draw_arc_circle(s-0.5/len(df), e + 0.5/len(df), color=palette[c], ax=ax, precision=precision_circle,
                            thickness=thickness_sub_circle,
                            radius=(1 if inverted else 1+thickness_circle+spacing_sub*2*math.pi), **circle_args)

        
    for jj, src_p, tgt_p, src_c, tgt_c in zip(range(len(df)), df["position"].values, 
                                          df["tgt_position"].values,
                                          df["nbcat"].values, 
                                          df["tgt_nbcat"].values):
        if not (no_chords or (src_p == tgt_p or (not categorie_internal_chords and src_c == tgt_c))):
            draw_chord(position_circle(src_p),
                        position_circle(tgt_p), ax=ax,
                        color_start=palette[(tgt_c if inverted else src_c)],
                        color_end=palette[(src_c if inverted else tgt_c)],
                        precision=precision_chord,
                        **chord_args)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('equal')
    ax.axis('off')
    return df
